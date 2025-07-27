from collections import OrderedDict
from typing import Any

import torch

from ..utils._utils import build_vec_rep, legendre_quad
from ..utils.structures import ModelData, ModelDesign, ModelParams, SampleData
from ..utils.types import BaseHazardFn, LinkFn, Trajectory


class HazardMixin:
    """Mixin class for hazard model computations."""

    params_: ModelParams
    model_design: ModelDesign

    def __init__(
        self,
        n_quad: int,
        n_bissect: int,
        enable_likelihood_cache: bool,
        enable_predict_cache: bool,
        cache_limit: int,
        **kwargs: Any,
    ):
        """Initializes the class.

        Args:
            n_quad (int): Number of quadrature nodes.
            n_bissect (int): The number of bissection steps.
            enable_likelihood_cache (bool, optional): Enables cache during fit, and MCMC loops and likelihood computations in general. Defaults to True.
            enable_predict_cache (bool, optional): Enables cache during predicting steps. Defaults to False.
            cache_limit (int): Max length of cache.
            kwargs (Any): Additional kwargs.
        """
        self.n_quad = n_quad
        self.n_bissect = n_bissect
        self.enable_likelihood_cache = enable_likelihood_cache
        self.enable_predict_cache = enable_predict_cache
        self.cache_limit = cache_limit

        self._std_nodes, self._std_weights = legendre_quad(n_quad)

        self._cache: dict[str, OrderedDict[Any, torch.Tensor]] = {
            "half": OrderedDict(),
            "quad": OrderedDict(),
            "base": OrderedDict(),
        }

        super().__init__(**kwargs)

    def _get_cache(self, name: str, key: Any) -> torch.Tensor:
        """Gets the cache [name][key].

        Args:
            name (str): The name of the cache info.
            key (Any): The key inside the current info.

        Returns:
            torch.Tensor: The cached tensor.
        """
        cache = self._cache[name]
        cache.move_to_end(key)
        return cache[key]

    def _add_cache(self, name: str, key: Any, val: torch.Tensor) -> None:
        """Add value to cache if not exceeding the limit.

        Args:
            name (str): name (str): The name of the cache info.
            key (Any): The key inside the current info.
            val (torch.Tensor): The tensor to cache.
        """
        cache = self._cache[name]
        cache[key] = val
        if len(cache) > self.cache_limit:
            cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Clears the cached tensors."""
        self._cache = {
            "half": OrderedDict(),
            "quad": OrderedDict(),
            "base": OrderedDict(),
        }

    def _log_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor | None,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor | None,
        log_base_hazard_fn: BaseHazardFn,
        link: LinkFn,
        enable_cache: bool,
    ) -> torch.Tensor:
        """Computes log hazard.

        Args:
            t0 (torch.Tensor): Start time.
            t1 (torch.Tensor): End time.
            x (torch.Tensor | None): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_base_hazard_fn (BaseHazardFn): Base hazard function.
            link (LinkFn): Link function.
            enable_cache (bool): Enables caching.

        Returns:
            torch.Tensor: The computed log hazard.
        """
        # Compute baseline hazard
        key = (
            id(log_base_hazard_fn),
            hash(t0.numpy().tobytes()),  # type: ignore
            hash(t1.numpy().tobytes()),  # type: ignore
        )
        try:
            base = self._get_cache("base", key)

        except Exception:
            base = log_base_hazard_fn(t0, t1)
            if enable_cache:
                self._add_cache("base", key, base)

        # Compute time-varying effects
        mod = link(t1, x, psi) @ alpha

        # Compute covariates effect
        cov = torch.tensor(0.0, dtype=torch.float32)
        if x is not None and beta is not None:
            cov = x @ beta.unsqueeze(1)

        # Compute the total
        log_hazard_vals = base + mod + cov

        return log_hazard_vals

    def _cum_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        c: torch.Tensor | None,
        x: torch.Tensor | None,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor | None,
        log_base_hazard_fn: BaseHazardFn,
        link: LinkFn,
        enable_cache: bool,
    ) -> torch.Tensor:
        """Computes cumulative hazard.

        Args:
            t0 (torch.Tensor): Start time.
            t1 (torch.Tensor): End time.
            c (torch.Tensor | None): Integration start or censoring time, if None, means t0.
            x (torch.Tensor | None): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_base_hazard_fn (BaseHazardFn): Base hazard function.
            link (LinkFn): Link function.
            enable_cache (bool): Enables caching.

        Returns:
            torch.Tensor: The computed cumulative hazard.
        """
        # Reshape for broadcasting
        t0, t1, c = (
            t0.view(-1, 1),
            t1.view(-1, 1),
            c.view(-1, 1) if c is not None else t0.view(-1, 1),
        )

        # Transform to quadrature interval
        key = (hash(c.numpy().tobytes()), hash(t1.numpy().tobytes()))  # type: ignore
        try:
            half = self._cache["half"][key]
            quad = self._get_cache("quad", key)

        except Exception:
            half = 0.5 * (t1 - c)
            mid = 0.5 * (c + t1)
            quad = torch.addmm(mid, half.view(-1, 1), self._std_nodes.view(1, -1))

            self._add_cache("half", key, half)
            self._add_cache("quad", key, quad)

        # Compute hazard at quadrature points
        log_hazard_vals = self._log_hazard(
            t0, quad, x, psi, alpha, beta, log_base_hazard_fn, link, enable_cache
        )

        # Numerical integration using Gaussian quadrature
        hazard_vals = torch.exp(torch.clamp(log_hazard_vals, min=-50.0, max=50.0))

        cum_hazard_vals = half.view(-1) * (hazard_vals @ self._std_weights)

        return cum_hazard_vals

    def _log_and_cum_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor | None,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor | None,
        log_base_hazard_fn: BaseHazardFn,
        link: LinkFn,
        enable_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes both log and cumulative hazard.

        Args:
            t0 (torch.Tensor): Start time.
            t1 (torch.Tensor): End time.
            x (torch.Tensor | None): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_base_hazard_fn (BaseHazardFn): Base hazard function.
            link (LinkFn): Link function.
            enable_cache (bool): Enables caching.

        Raises:
            RuntimeError: If the computation fails.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing log and cumulative hazard.
        """
        # Reshape for broadcasting
        t0, t1 = t0.view(-1, 1), t1.view(-1, 1)

        # Transform to quadrature interval
        key = (hash(t0.numpy().tobytes()), hash(t1.numpy().tobytes()))  # type: ignore
        try:
            half = self._cache["half"][key]
            quad = self._get_cache("quad", key)

        except Exception:
            half = 0.5 * (t1 - t0)
            mid = 0.5 * (t0 + t1)
            quad = torch.addmm(mid, half.view(-1, 1), self._std_nodes.view(1, -1))

            if enable_cache:
                self._add_cache("half", key, half)
                self._add_cache("quad", key, quad)

        # Combine with t1
        t1_and_quad = torch.cat([t1, quad], dim=1)

        # Compute log hazard at all points
        temp = self._log_hazard(
            t0, t1_and_quad, x, psi, alpha, beta, log_base_hazard_fn, link, enable_cache
        )

        # Extract log hazard at endpoint and quadrature points
        log_hazard_vals = temp[:, :1]  # Log hazard at t1
        hazard_vals = torch.exp(
            torch.clamp(temp[:, 1:], min=-50.0, max=50.0)
        )  # Hazard at quadrature points

        # Compute cumulative hazard using quadrature
        cum_hazard_vals = half.view(-1) * (hazard_vals @ self._std_weights)

        return log_hazard_vals.view(-1), cum_hazard_vals

    def _sample_transition(
        self,
        t0: torch.Tensor,
        c_max: torch.Tensor,
        x: torch.Tensor | None,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor | None,
        log_base_hazard_fn: BaseHazardFn,
        link: LinkFn,
        *,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample survival times using inverse transform sampling.

        Args:
            t0 (torch.Tensor): Starting time.
            c_max (torch.Tensor): Right censoring sampling time.
            x (torch.Tensor | None): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_base_hazard_fn (BaseHazardFn): Base hazard function.
            link (LinkFn): Link function.
            c (torch.Tensor | None, optional): Conditionning time. Defaults to None.

        Returns:
            torch.Tensor: The computed pre transition times.
        """
        # Initialize for bisection search
        t0 = t0.view(-1, 1)
        t_left, t_right = t0.clone().view(-1, 1), c_max.clone().view(-1, 1)

        # Generate exponential random variables
        target = -torch.log(torch.clamp(torch.rand(t_left.numel()), min=1e-8))

        # Bisection search for survival times
        for _ in range(self.n_bissect):
            t_mid = 0.5 * (t_left + t_right)

            cumulative = self._cum_hazard(
                t0, t_mid, c, x, psi, alpha, beta, log_base_hazard_fn, link, False
            )

            # Update search bounds
            accept_mask = cumulative < target
            t_left[accept_mask] = t_mid[accept_mask]
            t_right[~accept_mask] = t_mid[~accept_mask]

        return t_right.view(-1)

    def _sample_trajectory_step(
        self,
        trajectories: list[Trajectory],
        sample_data: SampleData,
        c_max: torch.Tensor,
    ) -> bool:
        """Appends the next simulated transition.

        Args:
            trajectories (list[Trajectory]): The current trajectories.
            sample_data (SampleData): The sampling data.
            c_max (torch.Tensor): The censoring time.

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
                x.index_select(0, idx) if x is not None else None,
                psi.index_select(0, idx),
                alphas[key],
                betas[key] if betas is not None else None,
                *surv[key],
                c=c.index_select(0, idx) if c is not None else None,
            )

            t_candidates[idx, j] = t_sample

        # Find earliest transition
        min_times, argmin_indices = torch.min(t_candidates, dim=1)
        bucket_keys = list(current_buckets.keys())

        for i, (time, arg_idx) in enumerate(zip(min_times, argmin_indices)):
            if torch.isfinite(time):
                trajectories[i].append((float(time), bucket_keys[int(arg_idx)][1]))

        return True

    def compute_surv_log_probs(
        self, sample_data: SampleData, u: torch.Tensor
    ) -> torch.Tensor:
        """Computes log probabilites of remaining event free up to time u.

        Args:
            sample_data (SampleData): The data on which to compute the probabilities.
            u (torch.Tensor): The time at which to evaluate the probabilities.

        Raises:
            ValueError: If u is of incorrect shape.

        Returns:
            torch.Tensor: The computed survival log probabilities.
        """
        # Convert to float32
        u = torch.as_tensor(u, dtype=torch.float32)

        # Check dims
        if u.ndim != 2 or u.shape[0] != sample_data.size:
            raise ValueError(
                f"u must have shape ({sample_data.size}, eval_points), got {u.shape}"
            )

        last_states = [trajectory[-1:] for trajectory in sample_data.trajectories]
        buckets = build_vec_rep(
            last_states,
            torch.full((sample_data.size,), torch.inf),
            self.model_design.surv,
        )

        nlog_probs = torch.zeros_like(u)

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
                enable_predict_cache = self.enable_predict_cache

                # Compute negative log survival
                alts_loglik = self._cum_hazard(
                    t0,
                    t1,
                    c,
                    x.index_select(0, idx) if x is not None else None,
                    psi.index_select(0, idx),
                    alphas[key],
                    betas[key] if betas is not None else None,
                    *surv[key],
                    enable_predict_cache,
                )

                nlog_probs[:, k].index_add_(0, idx, alts_loglik)

        log_probs = torch.clamp(-nlog_probs, max=0.0)

        return log_probs

    def sample_trajectories(
        self,
        sample_data: SampleData,
        c_max: torch.Tensor,
        max_length: int = 100,
    ) -> list[Trajectory]:
        """Sample future trajectories from the fitted joint model.

        Args:
            sample_data (SampleData): Prediction data.
            c_max (torch.Tensor): The maximum trajectory sampling time (censoring time).
            max_length (int, optional): Maximum iterations or sampling (prevents infinite loops). Defaults to 100.

        Raises:
            ValueError: If all the parameters are not set.
            ValueError: If the shape of c_max is not compatible.

        Returns:
            list[Trajectory]: The sampled trajectories.
        """
        # Convert and check if c_max matches the right shape
        c_max = torch.as_tensor(c_max, dtype=torch.float32)
        if c_max.shape != (sample_data.size,):
            raise ValueError(
                "c_max has incorrect shape, got {c_max.shape}, expected {(sample_data.size,)}"
            )

        # Initialize with copies of current trajectories
        trajectories = [list(trajectory) for trajectory in sample_data.trajectories]

        # Sample future transitions iteratively
        for _ in range(max_length):
            if not self._sample_trajectory_step(trajectories, sample_data, c_max):
                break

        # Remove transitions that exceed censoring times
        trajectories = [
            trajectory[:-1] if trajectory[-1][0] > c_max[i] else trajectory
            for i, trajectory in enumerate(trajectories)
        ]

        return trajectories

    def _hazard_loglik(self, psi: torch.Tensor, data: ModelData) -> torch.Tensor:
        """Computes the hazard log likelihood.

        Args:
            psi (torch.Tensor): A matrix of individual parameters.
            data (ModelData): Dataset on which likelihood is computed.
            enable_cache (bool): Enable caching

        Returns:
            torch.Tensor: The computed log likelihood.
        """
        ll = torch.zeros(data.size, dtype=torch.float32)

        for key, bucket in data.extra_["buckets"].items():
            idx, t0, t1, obs = bucket

            x, alphas, betas, surv = (
                data.x,
                self.params_.alphas,
                self.params_.betas,
                self.model_design.surv,
            )

            obs_loglik, alts_loglik = self._log_and_cum_hazard(
                t0,
                t1,
                x.index_select(0, idx) if x is not None else None,
                psi.index_select(0, idx),
                alphas[key],
                betas[key] if betas is not None else None,
                *surv[key],
                self.enable_likelihood_cache,
            )

            vals = obs * obs_loglik - alts_loglik
            ll.index_add_(0, idx, vals)

        return ll
