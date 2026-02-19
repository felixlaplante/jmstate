from numbers import Integral
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from sklearn.utils.validation import (  #  type: ignore
    assert_all_finite,  #  type: ignore
    check_consistent_length,  #  type: ignore
)

from ..types._data import ModelData, ModelDesign, SampleData, SampleDataUnchecked
from ..types._defs import Trajectory
from ..types._parameters import ModelParameters
from ..utils._surv import build_possible_buckets, build_remaining_buckets


# NamedTuples
class HazardInfo(NamedTuple):
    """A simple internal `NamedTuple` required for hazard computation.

    Attributes:
        key (tuple[Any, Any]): The transition key.
        t0 (torch.Tensor): A column vector of previous transition times.
        t1 (torch.Tensor): A matrix of next transition times.
        x (torch.Tensor): The fixed covariates.
        indiv_params (torch.Tensor): The individual parameters.
    """

    key: tuple[Any, Any]
    t0: torch.Tensor
    t1: torch.Tensor
    x: torch.Tensor
    indiv_params: torch.Tensor

    @classmethod
    def create(
        cls,
        key: tuple[Any, Any],
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor,
        indiv_params: torch.Tensor,
        idxs: torch.Tensor,
    ) -> "HazardInfo":
        """Creates a HazardInfo object from sample data.

        Args:
            key (tuple[Any, Any]): The transition key.
            t0 (torch.Tensor): The previous transition times.
            t1 (torch.Tensor): The next transition times.
            x (torch.Tensor): The fixed covariates.
            indiv_params (torch.Tensor): The individual parameters.
            idxs (torch.Tensor): The indices to slice.

        Returns:
            HazardInfo: The HazardInfo object.
        """
        return cls(
            key,
            t0,
            t1,
            x.index_select(0, idxs),
            indiv_params.index_select(-2, idxs),
        )


class HazardMixin:
    """Mixin class for hazard model computations."""

    design: ModelDesign
    params: ModelParameters
    n_quad: int
    n_bisect: int
    _std_nodes: torch.Tensor
    _std_weights: torch.Tensor

    def __init__(
        self,
        n_quad: int,
        n_bisect: int,
        *args: Any,
        **kwargs: Any,
    ):
        """Initializes the hazard mixin.

        Args:
            n_quad (int): Number of quadrature nodes.
            n_bisect (int): The number of bisection steps.
        """
        super().__init__(*args, **kwargs)

        self.n_quad = n_quad
        self.n_bisect = n_bisect
        self._std_nodes, self._std_weights = self._legendre_quad(n_quad)

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
        key, t0, t1, x, indiv_params = hazard_info
        str_key = str(key)

        # Compute baseline hazard
        base = self.params.base_hazards[str_key](t0, t1)

        # Compute time-varying effects
        mod = (
            self.design.link_fns[key](t1, indiv_params)
            @ self.params.link_coefs[str_key]
        )

        # Compute covariates effect
        var = x @ self.params.x_coefs[str_key].unsqueeze(-1)

        return base + mod + var

    def _cum_hazard(self, hazard_info: HazardInfo) -> torch.Tensor:
        """Computes cumulative hazard.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.

        Returns:
            torch.Tensor: The computed cumulative hazard.
        """
        _, t0, t1, *_ = hazard_info

        # Transform to quadrature interval
        half = 0.5 * (t1 - t0)
        quad = (
            (t0 + t1)
            .reshape(-1, 1)
            .addmm(half.reshape(-1, 1), self._std_nodes, beta=0.5)
            .reshape(t1.size(0), -1)
        )

        # Compute hazard at quadrature points
        vals = self._log_hazard(hazard_info._replace(t1=quad)).clamp(max=50).exp()

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
        _, t0, t1, *_ = hazard_info

        # Transform to quadrature interval
        half = 0.5 * (t1 - t0)
        quad = torch.cat([t1, (t0 + t1).addmm(half, self._std_nodes, beta=0.5)], dim=-1)

        # Compute log hazard at all points
        all_vals = self._log_hazard(hazard_info._replace(t1=quad))
        all_vals[..., 1:].clamp_(max=50).exp_()

        return all_vals[..., 0], half.reshape(-1) * (
            all_vals[..., 1:] @ self._std_weights
        )

    def _hazard_logliks(
        self, data: ModelData, indiv_params: torch.Tensor
    ) -> torch.Tensor:
        """Computes the hazard log likelihoods.

        Args:
            data (ModelData): Dataset on which likelihood is computed.
            indiv_params (torch.Tensor): A matrix of individual parameters.

        Returns:
            torch.Tensor: The computed log likelihoods.
        """
        logliks = torch.zeros(indiv_params.shape[:-1])

        for key, bucket in data.buckets.items():
            # Compute log likelihood and scatter add
            idxs, t0, t1, obs = bucket
            hazard_info = HazardInfo.create(key, t0, t1, data.x, indiv_params, idxs)
            obs_logliks, alts_logliks = self._log_and_cum_hazard(hazard_info)
            vals = (-alts_logliks).addcmul(obs, obs_logliks)
            logliks.index_add_(-1, idxs, vals)

        return logliks

    @torch.no_grad()  # type: ignore
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
        r"""Compute log survival probabilities at specified times.

        Evaluates the log-probability of remaining event-free up to the prediction times
        :math:`u` conditional on individual-level parameters and censoring times. The
        computation uses the hazard function :math:`\lambda(t)` to obtain:

        .. math::
            \log \mathbb{P}(T^* \geq u \mid T^* > c) = -\int_c^u \lambda(t) \, dt,

        where :math:`c` denotes the censoring time for each individual. In cases with
        multiple possible transitions, :math:`\lambda(t)` sums over all
        transition-specific hazards:

        .. math::
            -\int_c^u \sum_{k'} \lambda^{k' \mid k}(t \mid t_0) \, dt,

        exploiting the Chasles property to simplify computation and improve numerical
        precision.

        The input `u` must be a matrix of shape :math:`(n, m)` where :math:`n` is the
        number of individuals and :math:`m` is the number of prediction time points.

        Args:
            sample_data (SampleData): The dataset containing covariates,
                individual-level parameters, trajectories, and censoring information.
            u (torch.Tensor): Matrix of evaluation times of shape `(n, m)`.

        Raises:
            ValueError: If `u` contains NaN or infinite values.
            ValueError: If `u` has a shape inconsistent with the number of individuals.

        Returns:
            torch.Tensor: Computed survival log-probabilities of shape `(n, m)`,
            with rows corresponding to individuals and columns to prediction times.
        """
        assert_all_finite(u, input_name="u")
        check_consistent_length(u, sample_data)

        x = sample_data.x
        indiv_params = sample_data.indiv_params
        t_trunc = sample_data.t_trunc

        # Get buckets from last states
        buckets = build_remaining_buckets(
            sample_data.trajectories, tuple(self.design.link_fns.keys())
        )

        # Compute the log probabilities summing over transitions
        nlogps = torch.zeros(*indiv_params.shape[:-1], u.size(1))
        for key, bucket in buckets.items():
            # Compute negative log survival and scatter add
            idxs, t0 = bucket
            t0 = t0 if t_trunc is None else t_trunc.index_select(-2, idxs)
            hazard_info = HazardInfo.create(
                key, t0, u.index_select(0, idxs), x, indiv_params, idxs
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
        _, t0, t1, *_ = hazard_info

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
            bool: True if the sampling is done.
        """
        x = sample_data.x
        indiv_params = sample_data.indiv_params
        t_trunc = sample_data.t_trunc

        # Get buckets from last states
        current_buckets = build_possible_buckets(
            sample_data.trajectories, c, tuple(self.design.link_fns.keys())
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
            hazard_info = HazardInfo.create(key, t0, t1, x, indiv_params, idxs)
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

    @torch.no_grad()  # type: ignore
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
        r"""Simulate individual trajectories from the multistate joint model.

        Generates sample trajectories for each individual up to the censoring times `c`,
        truncating to a maximum of `max_length` transitions to prevent infinite loops.
        The simulation employs a variant of Gillespie's algorithm adapted for individual
        parameter draws. These sampled trajectories form the basis for posterior
        predictive checks or downstream predictions in the joint model framework.

        The input `c` must be a column vector of shape :math:`(n, 1)` where :math:`n`
        is the number of individuals.

        Args:
            sample_data (SampleData): The dataset containing covariates, trajectories,
                and individual-level parameter used for sampling.
            c (torch.Tensor): Column vector of censoring times for each individual.
            max_length (int, optional): Maximum number of iterations or transitions
                sampled per trajectory. Defaults to 10.

        Raises:
            ValueError: If `c` contains NaN or infinite values.
            ValueError: If `c` has a shape inconsistent with the number of individuals.

        Returns:
            list[Trajectory]: List of sampled trajectories, one per individual, with
            each trajectory truncated at the censoring time.
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
            sample_data.indiv_params,
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
