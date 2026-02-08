from dataclasses import replace
from typing import Any, Final

import torch
from xxhash import xxh3_64_intdigest

from ..typedefs._data import CompleteModelData, ModelDesign, SampleData
from ..typedefs._defs import HazardInfo, Trajectory
from ..typedefs._params import ModelParams
from ..utils._dtype import get_dtype
from ..utils._misc import legendre_quad
from ..utils._surv import build_possible_buckets, build_remaining_buckets
from ._cache import Cache

# Constants
HAZARD_CACHE_KEYS: Final[tuple[str, ...]] = ("base", "half", "quad_c", "quad_lc")


class HazardMixin:
    """Mixin class for hazard model computations."""

    params_: ModelParams
    model_design: ModelDesign
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
        self.n_quad = n_quad
        self.n_bisect = n_bisect
        self.cache_limit = cache_limit
        self._std_nodes, self._std_weights = legendre_quad(n_quad)
        self._cache = Cache(cache_limit, HAZARD_CACHE_KEYS)

        super().__init__(*args, **kwargs)

    def _log_hazard(self, hazard_info: HazardInfo) -> torch.Tensor:
        """Computes log hazard.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.

        Returns:
            torch.Tensor: The computed log hazard.
        """
        # Unpack data
        t0, t1, x, psi, alpha, beta, base_hazard_fn, link_fn = hazard_info

        # Compute baseline hazard
        def _base_create():
            return base_hazard_fn(t0, t1)

        if self.cache_limit != 0:
            try:
                key = (
                    *hazard_info.base_hazard_fn.key,
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

    def _cum_hazard(
        self, hazard_info: HazardInfo, c: torch.Tensor | None
    ) -> torch.Tensor:
        """Computes cumulative hazard.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.
            c (torch.Tensor | None): Integration start or censoring time, t0 if None.

        Returns:
            torch.Tensor: The computed cumulative hazard.
        """
        # Unpack data
        t0, t1, *_ = hazard_info
        c = t0 if c is None else c

        # Transform to quadrature interval
        def _half_create():
            return 0.5 * (t1 - c)

        def _quad_c_create():
            return (
                (c + t1)
                .reshape(-1, 1)
                .addmm(half.reshape(-1, 1), self._std_nodes, beta=0.5)
                .reshape(t1.size(0), -1)
            )

        if self.cache_limit != 0:
            try:
                key = (
                    xxh3_64_intdigest(c.detach().contiguous().numpy()),  # type: ignore
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
        enable_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes both log and cumulative hazard.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.
            enable_cache (bool): Enables caching.

        Raises:
            RuntimeError: If the computation fails.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The log and cum hazard.
        """
        # Unpack data
        t0, t1, *_ = hazard_info

        # Transform to quadrature interval
        def _half_create():
            return 0.5 * (t1 - t0)

        def _quad_lc_create():
            return torch.cat(
                [t1, (t0 + t1).addmm(half, self._std_nodes, beta=0.5)], dim=-1
            )

        if enable_cache:
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

    def _sample_transition(
        self,
        hazard_info: HazardInfo,
        c: torch.Tensor | None,
    ) -> torch.Tensor:
        """Sample survival times using inverse transform sampling.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.
            c (torch.Tensor | None): Conditionning time.

        Returns:
            torch.Tensor: The computed pre transition times.
        """
        # Unpack data
        t0, t1, *_ = hazard_info

        # Initialize for bisection search
        t_left, t_right = (
            t0.clone(),
            torch.nextafter(t1, torch.tensor(torch.inf, dtype=t1.dtype)),
        )

        # Generate exponential random variables
        target = -torch.log(torch.rand_like(t_left))

        # Bisection search for survival times
        for _ in range(self.n_bisect):
            t_mid = 0.5 * (t_left + t_right)

            surv_logps = self._cum_hazard(hazard_info._replace(t1=t_mid), c)

            # Update search bounds
            accept_mask = surv_logps.reshape(target.shape) < target
            torch.where(accept_mask, t_mid, t_left, out=t_left)
            torch.where(accept_mask, t_right, t_mid, out=t_right)

        return t_right

    def _sample_trajectory_step(
        self, sample_data: SampleData, c_max: torch.Tensor
    ) -> bool:
        """Appends the next simulated transition.

        Args:
            sample_data (SampleData): Sampling data
            c_max (torch.Tensor): Max sampling time.

        Returns:
            bool: If the sampling is done.
        """
        # Unpack data
        x = sample_data.x
        psi = sample_data.psi
        c = sample_data.c

        # Get buckets from last states
        current_buckets = build_possible_buckets(
            sample_data.trajectories, c_max, tuple(self.model_design.surv.keys())
        )

        if not current_buckets:
            return False

        # Initialize candidate transition times
        n_transitions = len(current_buckets)
        t_candidates = torch.full(
            (sample_data.size, n_transitions), torch.inf, dtype=get_dtype()
        )

        for j, (key, bucket) in enumerate(current_buckets.items()):
            idxs, t0, t1 = bucket

            # Create info
            hazard_info = HazardInfo(
                t0,
                t1,
                None if x is None else x.index_select(-2, idxs),
                psi.index_select(-2, idxs),
                self.params_.alphas[key],
                None if self.params_.betas is None else self.params_.betas[key],
                *self.model_design.surv[key],
            )

            # Sample transition times
            t_sample = self._sample_transition(
                hazard_info,
                None if c is None else c.index_select(-2, idxs),
            )

            t_candidates[idxs, j] = t_sample.reshape(-1)

        # Find earliest transition
        min_times, argmin_idxs = torch.min(t_candidates, dim=1)
        bucket_keys = list(current_buckets.keys())

        for i, (time, arg_idx) in enumerate(zip(min_times, argmin_idxs, strict=True)):
            if torch.isfinite(time):
                sample_data.trajectories[i].append(
                    (time.item(), bucket_keys[int(arg_idx)][1])
                )

        return True

    def _sample_trajectories(
        self,
        sample_data: SampleData,
        c_max: torch.Tensor,
        max_length: int,
    ) -> list[Trajectory]:
        """Sample future trajectories from the fitted joint model.

        Args:
            sample_data (SampleData): Prediction data.
            c_max (torch.Tensor): The maximum trajectory censoring times.
            max_length (int): Maximum iterations or sampling.

        Returns:
            list[Trajectory]: The sampled trajectories.
        """
        sample_data_copied = replace(sample_data, skip_validation=True)

        # Sample future transitions iteratively
        for _ in range(max_length):
            if not self._sample_trajectory_step(
                sample_data_copied,
                c_max,
            ):
                break
            object.__setattr__(sample_data_copied, "c", None)

        return [
            trajectory if trajectory[-1][0] <= c_max[i] else trajectory[:-1]
            for i, trajectory in enumerate(sample_data_copied.trajectories)
        ]

    def _compute_surv_logps(
        self, sample_data: SampleData, u: torch.Tensor
    ) -> torch.Tensor:
        """Computes log probabilites of remaining event free up to time u.

        Args:
            sample_data (SampleData): The data on which to compute the probabilities.
            u (torch.Tensor): The time at which to evaluate the probabilities.

        Returns:
            torch.Tensor: The computed survival log probabilities.
        """
        # Unpack data
        x = sample_data.x
        psi = sample_data.psi
        c = sample_data.c

        # Get buckets from last states
        dtype = get_dtype()
        buckets = build_remaining_buckets(
            sample_data.trajectories, tuple(self.model_design.surv.keys())
        )

        nlogps = torch.zeros(*psi.shape[:-1], u.size(1), dtype=dtype)

        # Compute the log probabilities summing over transitions
        for key, bucket in buckets.items():
            idxs, t0 = bucket

            # Create info
            hazard_info = HazardInfo(
                t0,
                u.index_select(0, idxs),
                None if x is None else x.index_select(-2, idxs),
                psi.index_select(-2, idxs),
                self.params_.alphas[key],
                None if self.params_.betas is None else self.params_.betas[key],
                *self.model_design.surv[key],
            )

            # Compute negative log survival
            alts_logliks = self._cum_hazard(
                hazard_info, None if c is None else c.index_select(0, idxs)
            )

            nlogps.index_add_(-2, idxs, alts_logliks)

        return -nlogps.clamp(min=0.0)

    def _hazard_logliks(
        self, params: ModelParams, psi: torch.Tensor, data: CompleteModelData
    ) -> torch.Tensor:
        """Computes the hazard log likelihoods.

        Args:
            params (ModelParams): The model parameters.
            psi (torch.Tensor): A matrix of individual parameters.
            data (CompleteModelData): Dataset on which likelihood is computed.
            enable_cache (bool): Enable caching

        Returns:
            torch.Tensor: The computed log likelihoods.
        """
        logliks = torch.zeros(psi.shape[:-1], dtype=get_dtype())

        for key, bucket in data.buckets.items():
            idxs, t0, t1, obs = bucket

            # Create info
            hazard_info = HazardInfo(
                t0,
                t1,
                None if data.x is None else data.x.index_select(-2, idxs),
                psi.index_select(-2, idxs),
                params.alphas[key],
                None if params.betas is None else params.betas[key],
                *self.model_design.surv[key],
            )

            obs_logliks, alts_logliks = self._log_and_cum_hazard(
                hazard_info,
                self.cache_limit != 0,
            )

            vals = (-alts_logliks).addcmul(obs, obs_logliks)
            logliks.index_add_(-1, idxs, vals)

        return logliks
