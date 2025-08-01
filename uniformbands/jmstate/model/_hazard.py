from collections import OrderedDict
from typing import Any, Callable

import torch
from beartype import beartype

from ..typedefs._data import CompleteModelData, ModelDesign, SampleData
from ..typedefs._defs import (
    BaseHazardFn,
    LinkFn,
    Tensor1D,
    Tensor2D,
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
    _std_nodes: Tensor2D
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
        t0: Tensor1D | Tensor2D,
        t1: Tensor1D | Tensor2D,
        x: Tensor2D | None,
        psi: Tensor2D,
        alpha: Tensor1D,
        beta: Tensor1D | None,
        base_hazard_fn: BaseHazardFn,
        link_fn: LinkFn,
        enable_cache: bool,
    ) -> Tensor2D:
        """Computes log hazard.

        Args:
            t0 (Tensor1D | Tensor2D): Start time.
            t1 (Tensor1D | Tensor2D): End time.
            x (Tensor2D | None): Covariates.
            psi (Tensor2D): Inidivual parameters.
            alpha (Tensor1D): Link linear parameters.
            beta (Tensor1D): Covariate linear parameters.
            base_hazard_fn (BaseHazardFn): Base hazard function.
            link_fn (LinkFn): Link function.
            enable_cache (bool): Enables caching.

        Returns:
            Tensor2D: The computed log hazard.
        """
        # Compute baseline hazard
        key = (
            id(base_hazard_fn),
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
        t0: Tensor1D | Tensor2D,
        t1: Tensor1D | Tensor2D,
        c: Tensor1D | Tensor2D | None,
        x: Tensor2D | None,
        psi: Tensor2D,
        alpha: Tensor1D,
        beta: Tensor1D | None,
        base_hazard_fn: BaseHazardFn,
        link_fn: LinkFn,
        enable_cache: bool,
    ) -> Tensor1D:
        """Computes cumulative hazard.

        Args:
            t0 (Tensor1D | Tensor2D): Start time.
            t1 (Tensor1D | Tensor2D): End time.
            c (Tensor1D | Tensor2D | None): Integration start or censoring time, t0 if None.
            x (Tensor2D | None): Covariates.
            psi (Tensor2D): Inidivual parameters.
            alpha (Tensor1D): Link linear parameters.
            beta (Tensor1D): Covariate linear parameters.
            base_hazard_fn (BaseHazardFn): Base hazard function.
            link_fn (LinkFn): Link function.
            enable_cache (bool): Enables caching.

        Returns:
            Tensor1D: The computed cumulative hazard.
        """
        # Reshape for broadcasting
        t0, t1, c = (
            t0.view(-1, 1),
            t1.view(-1, 1),
            t0.view(-1, 1) if c is None else c.view(-1, 1),
        )

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
            return torch.addmm(0.5 * (c + t1), half, self._std_nodes)

        quad_c = (
            self._get_cache("quad_c", key, _quad_c_create)
            if enable_cache
            else _quad_c_create()
        )

        # Compute hazard at quadrature points
        vals = self._log_hazard(
            t0, quad_c, x, psi, alpha, beta, base_hazard_fn, link_fn, enable_cache
        )
        vals.clamp_(min=-50.0, max=50.0).exp_()

        return half.view(-1) * (vals @ self._std_weights)

    def _log_and_cum_hazard(
        self,
        t0: Tensor1D | Tensor2D,
        t1: Tensor1D | Tensor2D,
        x: Tensor2D | None,
        psi: Tensor2D,
        alpha: Tensor1D,
        beta: Tensor1D | None,
        base_hazard_fn: BaseHazardFn,
        link_fn: LinkFn,
        enable_cache: bool,
    ) -> tuple[Tensor1D, Tensor1D]:
        """Computes both log and cumulative hazard.

        Args:
            t0 (Tensor1D | Tensor2D): Start time.
            t1 (Tensor1D | Tensor2D): End time.
            x (Tensor2D | None): Covariates.
            psi (Tensor2D): Inidivual parameters.
            alpha (Tensor1D): Link linear parameters.
            beta (Tensor1D): Covariate linear parameters.
            base_hazard_fn (BaseHazardFn): Base hazard function.
            link_fn (LinkFn): Link function.
            enable_cache (bool): Enables caching.

        Raises:
            RuntimeError: If the computation fails.

        Returns:
            tuple[Tensor1D, Tensor1D]: The log and cumulative hazard.
        """
        # Reshape for broadcasting
        t0, t1 = t0.view(-1, 1), t1.view(-1, 1)

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
            return torch.cat(
                [t1, torch.addmm(0.5 * (t0 + t1), half, self._std_nodes)], dim=1
            )

        quad_lc = (
            self._get_cache("quad_lc", key, _quad_lc_create)
            if enable_cache
            else _quad_lc_create()
        )

        # Compute log hazard at all points
        all_vals = self._log_hazard(
            t0, quad_lc, x, psi, alpha, beta, base_hazard_fn, link_fn, enable_cache
        )
        all_vals[:, 1:].clamp_(min=-50.0, max=50.0).exp_()

        return all_vals[:, 0], half.view(-1) * (all_vals[:, 1:] @ self._std_weights)

    def _sample_transition(
        self,
        t0: Tensor1D | Tensor2D,
        c_max: Tensor1D | Tensor2D,
        x: Tensor2D | None,
        psi: Tensor2D,
        alpha: Tensor1D,
        beta: Tensor1D | None,
        base_hazard_fn: BaseHazardFn,
        link_fn: LinkFn,
        *,
        c: Tensor1D | Tensor2D | None = None,
    ) -> Tensor1D:
        """Sample survival times using inverse transform sampling.

        Args:
            t0 (Tensor1D | Tensor2D): Starting time.
            c_max (Tensor1D | Tensor2D): Right censoring sampling time.
            x (Tensor2D | None): Covariates.
            psi (Tensor2D): Inidivual parameters.
            alpha (Tensor1D): Link linear parameters.
            beta (Tensor1D | None): Covariate linear parameters.
            base_hazard_fn (BaseHazardFn): Base hazard function.
            link_fn (LinkFn): Link function.
            c (Tensor1D | Tensor2D | None, optional): Conditionning time. Defaults to None.

        Returns:
            Tensor1D: The computed pre transition times.
        """
        # Initialize for bisection search
        t0 = t0.view(-1, 1)
        t_left, t_right = t0.clone().view(-1, 1), c_max.clone().view(-1, 1)

        # Generate exponential random variables
        target = -torch.log(torch.rand(t_left.numel()))

        # Bisection search for survival times
        for _ in range(self.n_bissect):
            t_mid = 0.5 * (t_left + t_right)

            surv_logps = self._cum_hazard(
                t0,
                t_mid,
                c,
                x,
                psi,
                alpha,
                beta,
                base_hazard_fn,
                link_fn,
                self.cache_limit != 0,
            )

            # Update search bounds
            accept_mask = surv_logps < target
            t_left[accept_mask] = t_mid[accept_mask]
            t_right[~accept_mask] = t_mid[~accept_mask]

        return t_right.view(-1)

    def _sample_trajectory_step(
        self,
        trajectories: list[Trajectory],
        sample_data: SampleData,
        c_max: Tensor1D | Tensor2D,
    ) -> bool:
        """Appends the next simulated transition.

        Args:
            trajectories (list[Trajectory]): The current trajectories.
            sample_data (SampleData): The sampling data.
            c_max (Tensor1D | Tensor2D): The censoring time.

        Returns:
            bool: Returns False if simulations are left.
        """
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

            x = sample_data.x
            psi = sample_data.psi
            alphas = self.params_.alphas
            betas = self.params_.betas
            surv = self.model_design.surv
            c = sample_data.c

            # Sample transition times
            t_sample = self._sample_transition(
                t0,
                t1,
                None if x is None else x.index_select(0, idx),
                psi.index_select(0, idx),
                alphas[key],
                None if betas is None else betas[key],
                *surv[key],
                c=None if c is None else c.index_select(0, idx),
            )

            t_candidates[idx, j] = t_sample

        # Find earliest transition
        min_times, argmin_indices = torch.min(t_candidates, dim=1)
        bucket_keys = list(current_buckets.keys())

        for i, (time, arg_idx) in enumerate(zip(min_times, argmin_indices)):
            if torch.isfinite(time):
                trajectories[i].append((float(time), bucket_keys[int(arg_idx)][1]))

        return True

    @beartype
    def compute_surv_logps(self, sample_data: SampleData, u: Tensor2D) -> Tensor2D:
        """Computes log probabilites of remaining event free up to time u.

        Args:
            sample_data (SampleData): The data on which to compute the probabilities.
            u (Tensor2D): The time at which to evaluate the probabilities.

        Raises:
            ValueError: If u is of incorrect shape.

        Returns:
            Tensor2D: The computed survival log probabilities.
        """
        # Convert to float32
        u = u.to(torch.float32)
        check_consistent_size([u], sample_data.size)

        last_states = [trajectory[-1:] for trajectory in sample_data.trajectories]
        buckets = build_vec_rep(
            last_states,
            torch.full((sample_data.size,), torch.inf),
            self.model_design.surv,
        )

        nlogps = torch.zeros_like(u)

        # Compute the log probabilities summing over transitions
        for key, bucket in buckets.items():
            for k in range(u.shape[1]):
                idx, t0, _, _ = bucket
                t1 = u[:, k]

                x = sample_data.x
                psi = sample_data.psi
                alphas = self.params_.alphas
                betas = self.params_.betas
                surv = self.model_design.surv
                c = sample_data.c

                # Compute negative log survival
                alts_logliks = self._cum_hazard(
                    t0,
                    t1,
                    c,
                    None if x is None else x.index_select(0, idx),
                    psi.index_select(0, idx),
                    alphas[key],
                    None if betas is None else betas[key],
                    *surv[key],
                    self.cache_limit != 0,
                )

                nlogps[:, k].index_add_(0, idx, alts_logliks)

        return torch.clamp(-nlogps, max=0.0)

    @beartype
    def sample_trajectories(
        self,
        sample_data: SampleData,
        c_max: Tensor1D | Tensor2D,
        max_length: int = 100,
    ) -> list[Trajectory]:
        """Sample future trajectories from the fitted joint model.

        Args:
            sample_data (SampleData): Prediction data.
            c_max (Tensor1D | Tensor2D): The maximum trajectory sampling time (censoring time).
            max_length (int, optional): Maximum iterations or sampling. Defaults to 100.

        Raises:
            ValueError: If all the parameters are not set.
            ValueError: If the shape of c_max is not compatible.

        Returns:
            list[Trajectory]: The sampled trajectories.
        """
        # Convert and check if c_max matches the right shape
        c_max = c_max.to(torch.float32)
        check_consistent_size([c_max], sample_data.size)

        # Initialize with copies of current trajectories
        trajectories = [list(trajectory) for trajectory in sample_data.trajectories]

        # Sample future transitions iteratively
        for _ in range(max_length):
            if not self._sample_trajectory_step(trajectories, sample_data, c_max):
                break

        return [
            trajectory[:-1] if trajectory[-1][0] > c_max[i] else trajectory
            for i, trajectory in enumerate(trajectories)
        ]

    def _hazard_logliks(self, psi: Tensor2D, data: CompleteModelData) -> Tensor1D:
        """Computes the hazard log likelihood.

        Args:
            psi (Tensor2D): A matrix of individual parameters.
            data (ModelData): Dataset on which likelihood is computed.
            enable_cache (bool): Enable caching

        Returns:
            Tensor1D: The computed log likelihood.
        """
        logliks = torch.zeros(data.size, dtype=torch.float32)

        for key, bucket in data.buckets.items():
            idx, t0, t1, obs = bucket

            x = data.x
            alphas = self.params_.alphas
            betas = self.params_.betas
            surv = self.model_design.surv

            obs_logliks, alts_logliks = self._log_and_cum_hazard(
                t0,
                t1,
                None if x is None else x.index_select(0, idx),
                psi.index_select(0, idx),
                alphas[key],
                None if betas is None else betas[key],
                *surv[key],
                self.cache_limit != 0,
            )

            vals = obs * obs_logliks - alts_logliks
            logliks = logliks.index_add(0, idx, vals)

        return logliks
