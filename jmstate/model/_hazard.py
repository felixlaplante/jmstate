from collections import OrderedDict
from dataclasses import replace
from typing import Any, Callable

import torch
from beartype import beartype

from ..typedefs._data import CompleteModelData, ModelDesign, SampleData
from ..typedefs._defs import (
    HazardInfo,
    Tensor1D,
    Tensor2D,
    Tensor3D,
    TensorCol,
    TensorRow,
    Trajectory,
)
from ..typedefs._params import ModelParams
from ..utils._checks import check_consistent_size
from ..utils._misc import legendre_quad
from ..utils._surv import build_vec_rep


class HazardMixin:
    """Mixin class for hazard model computations."""

    params_: ModelParams
    model_design: ModelDesign
    n_quad: int
    n_bissect: int
    cache_limit: int | None
    _std_nodes: TensorRow
    _std_weights: Tensor1D

    def __init__(
        self,
        n_quad: int,
        n_bissect: int,
        cache_limit: int | None,
        **kwargs: Any,
    ):
        """Initializes the class.

        Args:
            n_quad (int): Number of quadrature nodes.
            n_bissect (int): The number of bissection steps.
            enable_cache (bool): Enables caching.
            cache_limit (int | None): Max length of cache.
            kwargs (Any): Additional kwargs.

        Raises:
            ValueError: If n_quad is not greater than 0.
            ValueError: If n_bissect is not greater than 0.
            ValueError: If cache_limit is not None and negative.
        """
        self.n_quad = n_quad
        self.n_bissect = n_bissect
        self.cache_limit = cache_limit

        if self.n_quad <= 0:
            raise ValueError(f"n_quad must be greater than 0, got {self.n_quad}")
        if self.n_bissect <= 0:
            raise ValueError(f"n_bissect must be greater than 0, got {self.n_bissect}")
        if self.cache_limit is not None and self.cache_limit < 0:
            raise ValueError(
                f"cache_limit must be None or positive integer, got {self.cache_limit}"
            )

        self._std_nodes, self._std_weights = legendre_quad(n_quad)

        self._cache: dict[str, OrderedDict[tuple[int, ...], torch.Tensor]] = {
            "base": OrderedDict(),
            "half": OrderedDict(),
            "quad_c": OrderedDict(),
            "quad_lc": OrderedDict(),
        }

        super().__init__(**kwargs)

    def _get_cache(
        self, name: str, key: Any, missing: Callable[[], torch.Tensor]
    ) -> torch.Tensor:
        """Gets the cache [name][key].

        Args:
            name (str): The name of the cache info.
            key (Any): The key inside the current info.
            missing (Callable[[], torch.Tensor]): The fallback function.

        Returns:
            torch.Tensor: The cached tensor.
        """
        cache = self._cache[name]
        if key in cache:
            cache.move_to_end(key)
        else:
            cache[key] = missing()

        if self.cache_limit is not None and len(cache) > self.cache_limit:
            cache.popitem(last=False)

        return cache[key]

    def clear_cache(self):
        """Clears the cached tensors."""
        self._cache = {
            "base": OrderedDict(),
            "half": OrderedDict(),
            "quad_c": OrderedDict(),
            "quad_lc": OrderedDict(),
        }

    def _log_hazard(
        self,
        hazard_info: HazardInfo,
        enable_cache: bool,
    ) -> Tensor2D | Tensor3D:
        """Computes log hazard.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.
            enable_cache (bool): Enables caching.

        Returns:
            Tensor2D | Tensor3D: The computed log hazard.
        """
        # Unpack data
        t0, t1, x, psi, alpha, beta, base_hazard_fn, link_fn = hazard_info

        # Compute baseline hazard
        key = (
            id(hazard_info.base_hazard_fn),
            hash(t0.numpy().tobytes()),  # type: ignore
            hash(t1.numpy().tobytes()),  # type: ignore
        )

        def _base_create():
            return base_hazard_fn(t0, t1)

        base = (
            self._get_cache("base", key, _base_create)
            if enable_cache
            else _base_create()
        )

        # Compute time-varying effects
        mod = link_fn(t1, psi) @ alpha

        # Compute covariates effect
        if x is None or beta is None:
            return base + mod

        return base + mod + x @ beta.unsqueeze(1)

    def _cum_hazard(
        self,
        hazard_info: HazardInfo,
        c: TensorCol | None,
        enable_cache: bool,
    ) -> Tensor1D | Tensor2D:
        """Computes cumulative hazard.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.
            c (TensorCol | None): Integration start or censoring time, t0 if None.
            enable_cache (bool): Enables caching.

        Returns:
            Tensor1D | Tensor2D: The computed cumulative hazard.
        """
        # Unpack data
        t0, t1, *_ = hazard_info
        c = t0 if c is None else c

        # Transform to quadrature interval
        key = (hash(c.numpy().tobytes()), hash(t1.numpy().tobytes()))  # type: ignore

        def _half_create():
            return 0.5 * (t1 - c)

        half = (
            self._get_cache("half", key, _half_create)
            if enable_cache
            else _half_create()
        )

        def _quad_c_create():
            return 0.5 * (c + t1) + half * self._std_nodes

        quad_c = (
            self._get_cache("quad_c", key, _quad_c_create)
            if enable_cache
            else _quad_c_create()
        )

        # Compute hazard at quadrature points
        vals = self._log_hazard(hazard_info._replace(t1=quad_c), enable_cache)
        vals.clamp_(min=-50.0, max=50.0).exp_()

        return half.view(-1) * (vals @ self._std_weights)

    def _log_and_cum_hazard(
        self,
        hazard_info: HazardInfo,
        enable_cache: bool,
    ) -> tuple[Tensor1D | Tensor2D, Tensor1D | Tensor2D]:
        """Computes both log and cumulative hazard.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.
            enable_cache (bool): Enables caching.

        Raises:
            RuntimeError: If the computation fails.

        Returns:
            tuple[Tensor1D | Tensor2D, Tensor1D | Tensor2D]: The log and cum hazard.
        """
        # Unpack data
        t0, t1, *_ = hazard_info

        # Transform to quadrature interval
        key = (hash(t0.numpy().tobytes()), hash(t1.numpy().tobytes()))  # type: ignore

        def _half_create():
            return 0.5 * (t1 - t0)

        half = (
            self._get_cache("half", key, _half_create)
            if enable_cache
            else _half_create()
        )

        def _quad_lc_create():
            return torch.cat([t1, 0.5 * (t0 + t1) + half * self._std_nodes], dim=-1)

        quad_lc = (
            self._get_cache("quad_lc", key, _quad_lc_create)
            if enable_cache
            else _quad_lc_create()
        )

        # Compute log hazard at all points
        all_vals = self._log_hazard(hazard_info._replace(t1=quad_lc), enable_cache)
        all_vals[..., 1:].clamp_(min=-50.0, max=50.0).exp_()

        return all_vals[..., 0], half.view(-1) * (all_vals[..., 1:] @ self._std_weights)

    def _sample_transition(
        self,
        hazard_info: HazardInfo,
        c: TensorCol | None,
    ) -> TensorCol:
        """Sample survival times using inverse transform sampling.

        Args:
            hazard_info (HazardInfo): All necessary information for computation.
            c (TensorCol | None): Conditionning time.

        Returns:
            Tensor1D: The computed pre transition times.
        """
        # Unpack data
        t0, t1, *_ = hazard_info

        # Initialize for bisection search
        t_left, t_right = t0.clone(), t1.clone()

        # Generate exponential random variables
        target = -torch.log(torch.rand_like(t_left))

        # Bisection search for survival times
        for _ in range(self.n_bissect):
            t_mid = 0.5 * (t_left + t_right)

            surv_logps = self._cum_hazard(
                hazard_info._replace(t1=t_mid),
                c,
                self.cache_limit != 0,
            )

            # Update search bounds
            accept_mask = surv_logps.view(target.shape) < target
            torch.where(accept_mask, t_mid, t_left, out=t_left)
            torch.where(accept_mask, t_right, t_mid, out=t_right)

        return t_right

    def _sample_trajectory_step(
        self, sample_data: SampleData, c_max: TensorCol
    ) -> bool:
        """Appends the next simulated transition.

        Args:
            sample_data (SampleData): Sampling data
            c_max (TensorCol): Max sampling time.

        Returns:
            bool: If the sampling is done.
        """
        # Unpack data
        x = sample_data.x
        trajectories = sample_data.trajectories
        psi = sample_data.psi
        c = sample_data.c

        # Get initial buckets from last states
        last_states = [trajectory[-1:] for trajectory in trajectories]
        current_buckets = build_vec_rep(last_states, c_max, self.model_design.surv)

        if not current_buckets:
            return False

        # Initialize candidate transition times
        n_transitions = len(current_buckets)
        t_candidates = torch.full(
            (sample_data.size, n_transitions), torch.inf, dtype=torch.float32
        )

        for j, (key, bucket) in enumerate(current_buckets.items()):
            idx, t0, t1, _ = bucket
            t1 = torch.nextafter(t1, torch.tensor(torch.inf, dtype=torch.float32))

            # Create info
            hazard_info = HazardInfo(
                t0,
                t1,
                None if x is None else x.index_select(0, idx),
                psi.index_select(-2, idx),
                self.params_.alphas[key],
                None if self.params_.betas is None else self.params_.betas[key],
                *self.model_design.surv[key],
            )

            # Sample transition times
            t_sample = self._sample_transition(
                hazard_info,
                None if c is None else c.index_select(0, idx),
            )

            t_candidates[idx, j] = t_sample.view(-1)

        # Find earliest transition
        min_times, argmin_idxs = torch.min(t_candidates, dim=1)
        bucket_keys = list(current_buckets.keys())

        for i, (time, arg_idx) in enumerate(zip(min_times, argmin_idxs)):
            if torch.isfinite(time):
                trajectories[i].append((time.item(), bucket_keys[int(arg_idx)][1]))

        return True

    @beartype
    def sample_trajectories(
        self,
        sample_data: SampleData,
        c_max: TensorCol,
        *,
        max_length: int = 10,
    ) -> list[Trajectory]:
        """Sample future trajectories from the fitted joint model.

        Args:
            sample_data (SampleData): Prediction data.
            c_max (TensorCol): The maximum trajectory censoring times.
            max_length (int, optional): Maximum iterations or sampling. Defaults to 10.

        Raises:
            ValueError: If psi has incorrect shape.

        Returns:
            list[Trajectory]: The sampled trajectories.
        """
        # Convert and check if c_max matches the right shape
        c_max = c_max.to(torch.float32)
        check_consistent_size((c_max,), (0,), sample_data.size)

        # Initialize with copies of current trajectories
        trajectories_copied = [
            list(trajectory) for trajectory in sample_data.trajectories
        ]
        sample_data_copied = replace(
            sample_data, trajectories=trajectories_copied, skip_validation=True
        )

        # Sample future transitions iteratively
        for _ in range(max_length):
            if not self._sample_trajectory_step(
                sample_data_copied,
                c_max,
            ):
                break

        return [
            trajectory[:-1] if trajectory[-1][0] > c_max[i] else trajectory
            for i, trajectory in enumerate(trajectories_copied)
        ]

    @beartype
    def compute_surv_logps(
        self, sample_data: SampleData, u: Tensor2D
    ) -> Tensor2D | Tensor3D:
        """Computes log probabilites of remaining event free up to time u.

        Args:
            sample_data (SampleData): The data on which to compute the probabilities.
            u (Tensor2D): The time at which to evaluate the probabilities.

        Raises:
            ValueError: If u is of incorrect shape.

        Returns:
            Tensor2D | Tensor3D: The computed survival log probabilities.
        """
        # Unpack data
        x = sample_data.x
        trajectories = sample_data.trajectories
        psi = sample_data.psi
        c = sample_data.c

        # Convert to float32
        u = u.to(torch.float32)
        check_consistent_size((u,), (0,), sample_data.size)

        last_states = [trajectory[-1:] for trajectory in trajectories]
        buckets = build_vec_rep(
            last_states,
            torch.full((sample_data.size,), torch.inf),
            self.model_design.surv,
        )

        nlogps = torch.zeros((*sample_data.psi.shape[:-1], u.size(1)))

        # Compute the log probabilities summing over transitions
        for key, bucket in buckets.items():
            for k in range(u.shape[1]):
                idx, t0, _, _ = bucket

                # Create info
                hazard_info = HazardInfo(
                    t0,
                    u[:, k].view(-1, 1),
                    None if x is None else x.index_select(0, idx),
                    psi.index_select(-2, idx),
                    self.params_.alphas[key],
                    None if self.params_.betas is None else self.params_.betas[key],
                    *self.model_design.surv[key],
                )

                # Compute negative log survival
                alts_logliks = self._cum_hazard(
                    hazard_info,
                    c,
                    self.cache_limit != 0,
                )

                nlogps[..., k].index_add_(-1, idx, alts_logliks)

        return -nlogps.clamp(min=0.0)

    def _hazard_logliks(self, psi: Tensor3D, data: CompleteModelData) -> Tensor1D:
        """Computes the hazard log likelihood.

        Args:
            psi (Tensor3D): A matrix of individual parameters.
            data (CompleteModelData): Dataset on which likelihood is computed.
            enable_cache (bool): Enable caching

        Returns:
            Tensor1D: The computed log likelihood.
        """
        logliks = torch.zeros((data.n_chains, data.size), dtype=torch.float32)

        for key, bucket in data.buckets.items():
            idx, t0, t1, obs = bucket

            # Create info
            hazard_info = HazardInfo(
                t0,
                t1,
                None if data.x is None else data.x.index_select(0, idx),
                psi.index_select(-2, idx),
                self.params_.alphas[key],
                None if self.params_.betas is None else self.params_.betas[key],
                *self.model_design.surv[key],
            )

            obs_logliks, alts_logliks = self._log_and_cum_hazard(
                hazard_info,
                self.cache_limit != 0,
            )

            vals = obs * obs_logliks - alts_logliks
            logliks = logliks.index_add(-1, idx, vals)

        return logliks
