from numbers import Integral
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from sklearn.utils.validation import (  #  type: ignore
    assert_all_finite,  #  type: ignore
    check_consistent_length,  #  type: ignore
)
from xxhash import xxh3_64_intdigest

from ..types._data import ModelData, ModelDesign, SampleData, SampleDataUnchecked
from ..types._defs import LinkFn, LogBaseHazardFn, Trajectory
from ..types._parameters import ModelParameters
from ..utils._cache import Cache
from ..utils._surv import build_possible_buckets, build_remaining_buckets


# NamedTuples
class HazardInfo(NamedTuple):
    """A simple internal `NamedTuple` required for hazard computation.

    Attributes:
        t0 (torch.Tensor): A column vector of previous transition times.
        t1 (torch.Tensor): A matrix of next transition times.
        x (torch.Tensor | None): The covariates.
        psi (torch.Tensor): The individual parameters.
        alpha (torch.Tensor): The baseline hazard parameters.
        beta (torch.Tensor | None): The regression parameters.
        log_base_hazard_fn (LogBaseHazardFn): The log base hazard function.
        link_fn (LinkFn): The link function.
    """

    t0: torch.Tensor
    t1: torch.Tensor
    x: torch.Tensor | None
    psi: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor | None
    log_base_hazard_fn: LogBaseHazardFn
    link_fn: LinkFn

    @classmethod
    def create(
        cls,
        model: "HazardMixin",
        x: torch.Tensor | None,
        psi: torch.Tensor,
        key: tuple[Any, Any],
        idxs: torch.Tensor,
        t0: torch.Tensor,
        t1: torch.Tensor,
    ) -> "HazardInfo":
        """Creates a HazardInfo object from sample data.

        Args:
            model (HazardMixin): The model.
            x (torch.Tensor | None): The covariates.
            psi (torch.Tensor): The individual parameters.
            key (tuple[Any, Any]): The transition key.
            idxs (torch.Tensor): The indices to slice.
            t0 (torch.Tensor): The previous transition times.
            t1 (torch.Tensor): The next transition times.

        Returns:
            HazardInfo: The HazardInfo object.
        """
        return cls(
            t0,
            t1,
            None if x is None else x.index_select(0, idxs),
            psi.index_select(-2, idxs),
            model.params.alphas[str(key)],
            None if model.params.betas is None else model.params.betas[str(key)],
            *model.design.surv_fns[key],
        )


class HazardMixin:
    """Mixin class for hazard model computations."""

    design: ModelDesign
    params: ModelParameters
    n_quad: int
    n_bisect: int
    cache_limit: int | None
    _std_nodes: torch.Tensor
    _std_weights: torch.Tensor
    _cache: Cache

    def __init__(
        self,
        n_quad: int,
        n_bisect: int,
        cache_limit: int | None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initializes the hazard mixin.

        Args:
            n_quad (int): Number of quadrature nodes.
            n_bisect (int): The number of bisection steps.
            cache_limit (int | None): Max length of cache.
        """
        super().__init__(*args, **kwargs)

        self.n_quad = n_quad
        self.n_bisect = n_bisect
        self.cache_limit = cache_limit
        self._std_nodes, self._std_weights = self._legendre_quad(n_quad)
        self._cache = Cache(cache_limit)

    @staticmethod
    def _legendre_quad(n_quad: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the Legendre quadrature nodes and weights.

        Args:
            n_quad (int): The number of quadrature points.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The nodes and weights.
        """
        nodes, weights = cast(
            tuple[
                np.ndarray[Any, np.dtype[np.float64]],
                np.ndarray[Any, np.dtype[np.float64]],
            ],
            np.polynomial.legendre.leggauss(n_quad),  # type: ignore
        )

        dtype = torch.get_default_dtype()
        std_nodes = torch.tensor(nodes, dtype=dtype).unsqueeze(0)
        std_weights = torch.tensor(weights, dtype=dtype)

        return std_nodes, std_weights

    def _log_hazard(self, hazard_info: HazardInfo) -> torch.Tensor:
        """Computes log hazard.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.

        Returns:
            torch.Tensor: The computed log hazard.
        """
        t0, t1, x, psi, alpha, beta, log_base_hazard_fn, link_fn = hazard_info

        # Compute baseline hazard
        def _base_create():
            return log_base_hazard_fn(t0, t1)

        if self.cache_limit != 0:
            try:
                key = (
                    *hazard_info.log_base_hazard_fn.key,
                    xxh3_64_intdigest(t0.detach().contiguous().numpy()),  # type: ignore
                    xxh3_64_intdigest(t1.detach().contiguous().numpy()),  # type: ignore
                )
                base = self._cache.get_cache("base", key, _base_create)
            except RuntimeError:
                base = _base_create()
        else:
            base = _base_create()

        # Compute time-varying effects
        mod = link_fn(t1, psi) @ alpha

        # Compute covariates effect
        if x is None or beta is None:
            return base + mod

        return base + mod + x @ beta.unsqueeze(-1)

    def _cum_hazard(self, hazard_info: HazardInfo) -> torch.Tensor:
        """Computes cumulative hazard.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.

        Returns:
            torch.Tensor: The computed cumulative hazard.
        """
        t0, t1, *_ = hazard_info

        # Transform to quadrature interval
        def _half_create():
            return 0.5 * (t1 - t0)

        def _quad_c_create():
            return (
                (t0 + t1)
                .reshape(-1, 1)
                .addmm(half.reshape(-1, 1), self._std_nodes, beta=0.5)
                .reshape(t1.size(0), -1)
            )

        if self.cache_limit != 0:
            try:
                key = (
                    xxh3_64_intdigest(t0.detach().contiguous().numpy()),  # type: ignore
                    xxh3_64_intdigest(t1.detach().contiguous().numpy()),  # type: ignore
                )
                half = self._cache.get_cache("half", key, _half_create)
                quad_c = self._cache.get_cache("quad_c", key, _quad_c_create)
            except RuntimeError:
                half = _half_create()
                quad_c = _quad_c_create()
        else:
            half = _half_create()
            quad_c = _quad_c_create()

        # Compute hazard at quadrature points
        vals = self._log_hazard(hazard_info._replace(t1=quad_c))
        vals.clamp_(max=50.0).exp_()

        return half.reshape(t1.shape) * (
            vals.reshape(*vals.shape[:-1], -1, self._std_weights.size(-1))
            @ self._std_weights
        )

    def _log_and_cum_hazard(
        self,
        hazard_info: HazardInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes both log and cumulative hazard.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The log and cum hazard.
        """
        t0, t1, *_ = hazard_info

        # Transform to quadrature interval
        def _half_create():
            return 0.5 * (t1 - t0)

        def _quad_lc_create():
            return torch.cat(
                [t1, (t0 + t1).addmm(half, self._std_nodes, beta=0.5)], dim=-1
            )

        if self.cache_limit != 0:
            try:
                key = (
                    xxh3_64_intdigest(t0.detach().contiguous().numpy()),  # type: ignore
                    xxh3_64_intdigest(t1.detach().contiguous().numpy()),  # type: ignore
                )
                half = self._cache.get_cache("half", key, _half_create)
                quad_lc = self._cache.get_cache("quad_c", key, _quad_lc_create)
            except RuntimeError:
                half = _half_create()
                quad_lc = _quad_lc_create()
        else:
            half = _half_create()
            quad_lc = _quad_lc_create()

        # Compute log hazard at all points
        all_vals = self._log_hazard(hazard_info._replace(t1=quad_lc))
        all_vals[..., 1:].clamp_(max=50.0).exp_()

        return all_vals[..., 0], half.reshape(-1) * (
            all_vals[..., 1:] @ self._std_weights
        )

    def _hazard_logliks(self, data: ModelData, psi: torch.Tensor) -> torch.Tensor:
        """Computes the hazard log likelihoods.

        Args:
            data (ModelData): Dataset on which likelihood is computed.
            psi (torch.Tensor): A matrix of individual parameters.

        Returns:
            torch.Tensor: The computed log likelihoods.
        """
        logliks = torch.zeros(psi.shape[:-1])

        for key, bucket in data.buckets.items():
            # Compute log likelihood and scatter add
            idxs, t0, t1, obs = bucket
            hazard_info = HazardInfo.create(self, data.x, psi, key, idxs, t0, t1)
            obs_logliks, alts_logliks = self._log_and_cum_hazard(hazard_info)
            vals = (-alts_logliks).addcmul(obs, obs_logliks)
            logliks.index_add_(-1, idxs, vals)

        return logliks

    @validate_params(
        {
            "sample_data": [SampleData],
            "u": [torch.Tensor],
        },
        prefer_skip_nested_validation=True,
    )
    def compute_surv_logps(
        self, sample_data: SampleData, u: torch.Tensor
    ) -> torch.Tensor:
        r"""Computes log probabilites of remaining event free up to time u.

        A censoring time may also be given. With known individual effects, this computes
        at the times :math:`u` the values of the log survival probabilities given input
        data conditionally to survival up to time :math:`c`:

        .. math::
            \log \mathbb{P}(T^* \geq u \mid T^* > c) = -\int_c^u \lambda(t) \, dt.

        When multiple transitions are allowed, :math:`\lambda(t)` is a sum over all
        possible transitions, that is to say if an individual is in the state :math:`k`
        from time :math:`t_0`, this gives:

        .. math::
            -\int_c^u \sum_{k'} \lambda^{k' \mid k}(t \mid t_0) \, dt.

        Please note this makes use of the Chasles property in order to avoid the
        computation of two integrals and make computations more precise.

        The variable `u` is expected to be a matrix with the same number of rows as
        individuals, and the same number of columns as prediction times.

        Args:
            sample_data (SampleData): The data on which to compute the probabilities.
            u (torch.Tensor): The times at which to evaluate the probabilities.

        Raises:
            ValueError: If `u` contains inf or NaN values.
            ValueError: If `u` has incorrect shape.

        Returns:
            torch.Tensor: The computed survival log probabilities.
        """
        assert_all_finite(u, input_name="u")
        check_consistent_length(u, sample_data)

        x = sample_data.x
        psi = sample_data.psi
        t_trunc = sample_data.t_trunc

        # Get buckets from last states
        buckets = build_remaining_buckets(
            sample_data.trajectories, tuple(self.design.surv_fns.keys())
        )

        # Compute the log probabilities summing over transitions
        nlogps = torch.zeros(*psi.shape[:-1], u.size(1))
        for key, bucket in buckets.items():
            # Compute negative log survival and scatter add
            idxs, t0 = bucket
            t0 = t0 if t_trunc is None else t_trunc.index_select(-2, idxs)
            hazard_info = HazardInfo.create(
                self, x, psi, key, idxs, t0, u.index_select(0, idxs)
            )
            alts_logliks = self._cum_hazard(hazard_info)
            nlogps.index_add_(-2, idxs, alts_logliks)

        return -nlogps.clamp(min=0.0)

    def _sample_transition(
        self,
        hazard_info: HazardInfo,
    ) -> torch.Tensor:
        """Sample survival times using inverse transform sampling.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.

        Returns:
            torch.Tensor: The computed pre transition times.
        """
        t0, t1, *_ = hazard_info

        # Initialize for bisection search
        t_left, t_right = (
            t0.clone(),
            torch.nextafter(t1, torch.tensor(torch.inf)),
        )

        # Generate exponential random variables
        target = -torch.log(torch.rand_like(t_left))

        # Bisection search for survival times
        for _ in range(self.n_bisect):
            t_mid = 0.5 * (t_left + t_right)
            surv_nlogps = self._cum_hazard(hazard_info._replace(t1=t_mid))

            # Update search bounds
            accept_mask = surv_nlogps.reshape(target.shape) < target
            torch.where(accept_mask, t_mid, t_left, out=t_left)
            torch.where(accept_mask, t_right, t_mid, out=t_right)

        return t_right

    def _sample_trajectory_step(self, sample_data: SampleData, c: torch.Tensor) -> bool:
        """Appends the next simulated transition.

        Args:
            sample_data (SampleData): Sampling data
            c (torch.Tensor): Sampling censoring time.

        Returns:
            bool: If the sampling is done.
        """
        x = sample_data.x
        psi = sample_data.psi
        t_trunc = sample_data.t_trunc

        # Get buckets from last states
        current_buckets = build_possible_buckets(
            sample_data.trajectories, c, tuple(self.design.surv_fns.keys())
        )

        if not current_buckets:
            return True

        # Initialize candidate transition times
        n_transitions = len(current_buckets)
        t_candidates = torch.full((len(sample_data), n_transitions), torch.inf)

        for j, (key, bucket) in enumerate(current_buckets.items()):
            # Sample transition times, and condition with c
            idxs, t0, t1 = bucket
            t0 = t0 if t_trunc is None else t_trunc.index_select(-2, idxs)
            hazard_info = HazardInfo.create(self, x, psi, key, idxs, t0, t1)
            t_sample = self._sample_transition(hazard_info)
            t_candidates[idxs, j] = t_sample.reshape(-1)

        # Find earliest transition
        min_times, argmin_idxs = torch.min(t_candidates, dim=1)
        bucket_keys = list(current_buckets.keys())

        for i, (time, arg_idx) in enumerate(zip(min_times, argmin_idxs, strict=True)):
            if torch.isfinite(time):
                sample_data.trajectories[i].append(
                    (time.item(), bucket_keys[int(arg_idx)][1])
                )

        return False

    @validate_params(
        {
            "sample_data": [SampleData],
            "c": [torch.Tensor],
            "max_length": [Interval(Integral, 1, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def sample_trajectories(
        self,
        sample_data: SampleData,
        c: torch.Tensor,
        *,
        max_length: int = 10,
    ) -> list[Trajectory]:
        """Sample trajectories from the multistate joint model.

        This simulates the trajectories up to a censoring time `c` with a maximum length
        of `max_length` to avoid infinite loops. This uses a variant of Gillepsie's
        algorithm.

        The variable `c` is expected to be a column vector with the same number of
        rows as individuals.

        Args:
            sample_data (SampleData): Prediction data.
            c (torch.Tensor): The sampling censoring times.
            max_length (int, optional): Maximum iterations or sampling. Defaults to 10.

        Raises:
            ValueError: If `c` contains inf or NaN values.
            ValueError: If `c` has incorrect shape.

        Returns:
            list[Trajectory]: The sampled trajectories.
        """
        assert_all_finite(c, input_name="c")
        check_consistent_length(c, sample_data)

        # Copy sample data to avoid modifying the original
        trajectories_copied = [
            trajectory.copy() for trajectory in sample_data.trajectories
        ]
        sample_data_copied = SampleDataUnchecked(
            sample_data.x,
            trajectories_copied,
            sample_data.psi,
            sample_data.t_trunc,
        )

        # Sample future transitions iteratively
        for _ in range(max_length):
            if self._sample_trajectory_step(sample_data_copied, c):
                break
            sample_data_copied.t_trunc = None

        return [
            trajectory if trajectory[-1][0] <= c[i] else trajectory[:-1]
            for i, trajectory in enumerate(sample_data_copied.trajectories)
        ]
