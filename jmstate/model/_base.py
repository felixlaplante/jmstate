import copy
from typing import Any, Callable, SupportsFloat, cast

import torch
from beartype import beartype

from ..typedefs._data import CompleteModelData, ModelData, ModelDesign
from ..typedefs._defaults import DEFAULT_HYPERPARAMETERS
from ..typedefs._defs import (
    LOGTWOPI,
    Info,
    Job,
    Metrics,
    Tensor0D,
    Tensor2D,
    Tensor3D,
)
from ..typedefs._params import ModelParams
from ..utils._linalg import get_cholesky_and_log_eigvals
from ..utils._misc import params_like_from_flat, run_jobs
from ._hazard import HazardMixin
from ._longitudinal import LongitudinalMixin
from ._sampler import MetropolisHastingsSampler


class MultiStateJointModel(LongitudinalMixin, HazardMixin):
    """A class of the nonlinear multistate joint model.

    It feature possibility
    to simulate data, fit based on stochastic gradient with any torch.optim
    optimizer of choice.

    Args:
        LongitudinalMixin (_type_): The longitudinal part of the model.
        HazardMixin (_type_): The hazard part of the model.
    """

    model_design: ModelDesign
    params_: ModelParams
    pen: Callable[[ModelParams], Tensor0D] | None
    n_quad: int
    n_bissect: int
    cache_limit: int | None
    data: ModelData | None
    metrics_: Metrics | None
    fit_: bool

    @beartype
    def __init__(
        self,
        model_design: ModelDesign,
        init_params: ModelParams,
        *,
        pen: Callable[[ModelParams], Tensor0D] | None = None,
        n_quad: int = 32,
        n_bissect: int = 32,
        cache_limit: int | None = 256,
    ):
        """Initializes the joint model based on the user defined design.

        Args:
            model_design (ModelDesign): Model design containing modeling information.
            init_params (ModelParams): Initial values for the parameters.
            pen (Callable[[ModelParams], Tensor0D] | None, optional): The penalization function. Defaults to None.
            n_quad (int, optional): The used numnber of points for Gauss-Legendre quadrature. Defaults to 32.
            n_bissect (int, optional): The number of bissection steps used in transition sampling. Defaults to 32.
            cache_limit (int | None, optional): The max length of cache. Defaults to 256.

        Raises:
            TypeError: If pen is not None and is not callable.
        """
        # Store model components
        self.model_design = model_design
        self.params_ = copy.deepcopy(init_params)

        # Store penalization
        self.pen = pen
        if self.pen is not None and not callable(self.pen):
            raise TypeError("pen must be callable or None")

        # Info of the Mixin Classes
        super().__init__(
            n_quad=n_quad,
            n_bissect=n_bissect,
            cache_limit=cache_limit,
        )

        # Initialize attributes that will be set later
        self.data = None
        self.metrics_ = None
        self.fit_ = False

    def _logpdfs_and_aux_fn(
        self, params: ModelParams, b: Tensor3D, data: CompleteModelData
    ) -> tuple[Tensor2D, tuple[Tensor2D, Tensor3D]]:
        """Computes the total log likelihood up to a constant.

        Args:
            params (ModelParams): The model parameters.
            b (Tensor3D): The individual random effects.
            data (CompleteModelData): Dataset on which the likeihood is computed.

        Returns:
            tuple[Tensor2D, Tensor2D, Tensor3D]: The computed quantities.
        """

        def _prior_logliks(b: Tensor3D) -> Tensor2D:
            Q_inv_cholesky, Q_nlog_eigvals = get_cholesky_and_log_eigvals(params, "Q")
            Q_quad_form = (b @ Q_inv_cholesky).pow(2).sum(dim=-1)
            Q_norm_factor = (Q_nlog_eigvals - LOGTWOPI).sum()

            return 0.5 * (Q_norm_factor - Q_quad_form)

        # Transform random effects to individual-specific parameters
        psi = self.model_design.individual_effects_fn(params.gamma, data.x, b)

        # Compute individual likelihood components
        long_logliks = super()._long_logliks(params, psi, data)
        hazard_logliks = super()._hazard_logliks(params, psi, data)
        prior_logliks = _prior_logliks(b)

        # Sum all likelihood components
        logliks = long_logliks + hazard_logliks
        logpdfs = logliks + prior_logliks

        return logpdfs, (logliks, psi)

    def _setup_mcmc(
        self,
        data: CompleteModelData,
        n_chains: int,
        init_step_size: SupportsFloat,
        adapt_rate: SupportsFloat,
        target_accept_rate: SupportsFloat,
    ) -> MetropolisHastingsSampler:
        """Setup the MCMC kernel and hyperparameters.

        Args:
            data (CompleteModelData): The complete dataset.
            n_chains (int): The number of parallel MCMC chains.
            init_step_size (SupportsFloat): Kernel standard error in Metropolis.
            adapt_rate (SupportsFloat): Adaptation rate for the step_size.
            target_accept_rate (SupportsFloat): Mean acceptance target.

        Returns:
            MetropolisHastingsSampler: The intialized Markov kernel.
        """
        # Initialize random effects
        init_b = torch.zeros(
            (n_chains, data.size, self.params_.Q_repr.dim), dtype=torch.float32
        )

        return MetropolisHastingsSampler(
            lambda b: self._logpdfs_and_aux_fn(self.params_, b, data),
            init_b,
            n_chains,
            init_step_size,
            adapt_rate,
            target_accept_rate,
        )

    def _setup_jobs(
        self, jobs: Job | list[Job]
    ) -> tuple[list[Job], dict[str, Any] | None]:
        """Sets up jobs, and gets default hyperparameters.

        Args:
            jobs (Job | list[Job]): The job(s).

        Returns:
            tuple[list[Job], dict[str, Any] | None]: The jobs and default hyperparameters.
        """
        # Initialize jobs
        if isinstance(jobs, Job):
            jobs = [jobs]

        hyperparameters: dict[str, Any] | None = None
        for job in jobs:
            val = DEFAULT_HYPERPARAMETERS.get(type(job))
            if val is not None:
                hyperparameters = val

        return jobs, hyperparameters

    @beartype
    def do(
        self,
        new_data: ModelData | None = None,
        *,
        jobs: Job | list[Job],
        max_iterations: int | None = None,
        n_chains: int | None = None,
        warmup: int | None = None,
        n_steps: int | None = None,
        init_step_size: SupportsFloat = 0.1,
        adapt_rate: SupportsFloat = 0.1,
        accept_target: SupportsFloat = 0.234,
        verbose: bool = True,
    ) -> Metrics | Any | None:
        """Runs the MultiStateJointModel loop and some jobs.

        Args:
            new_data (ModelData): The dataset to learn from.
            jobs (Job | list[Job]): A list of jobs to execute in order.
            max_iterations (int | None, optional): Maximum number of iterations. Defaults to None.
            n_chains (int | None, optional): Batch size used. Defaults to None.
            warmup (int | None, optional): The number of iteration steps used in the warmup. Defaults to None.
            n_steps (int | None, optional): The steps to do at each iteration. Defaults to None.
            init_step_size (SupportsFloat, optional): Kernel step in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (SupportsFloat, optional): Adaptation rate for the step_size. Defaults to 0.01.
            accept_target (SupportsFloat, optional): Mean acceptation target. Defaults to 0.234.
            verbose (bool, optional): Wheter or not to show progress. Defaults to True.

        Raises:
            ValueError: If both new_data and self.data are None.
            TypeError: If some attribute is left unset.

        Returns:
            Metrics | Any | None: The metrics, a single element, or None.
        """
        if new_data is None and self.data is None:
            raise ValueError("data must not be None if self.data is also None; use fit")

        # Load and complete data
        data = cast(ModelData, new_data if new_data is not None else self.data)
        complete_data = CompleteModelData(
            data.x, data.t, data.y, data.trajectories, data.c, skip_validation=True
        )
        complete_data.init(self.model_design, self.params_)

        # Set up jobs and hyperparameters.
        jobs, hyperparameters = self._setup_jobs(jobs)
        if hyperparameters is not None:
            max_iterations = (
                hyperparameters["max_iterations"]
                if max_iterations is None
                else max_iterations
            )
            n_chains = hyperparameters["n_chains"] if n_chains is None else n_chains
            warmup = hyperparameters["warmup"] if warmup is None else warmup
            n_steps = hyperparameters["n_steps"] if n_steps is None else n_steps

        # Check everything is there
        for field in ("max_iterations", "n_chains", "warmup", "n_steps"):
            if locals()[field] is None:
                raise TypeError(f"Missing required argument: '{field}'")

        # Set up MCMC
        sampler = self._setup_mcmc(
            complete_data,
            cast(int, n_chains),
            init_step_size,
            adapt_rate,
            accept_target,
        )
        sampler.run(cast(int, warmup))

        def _logpdfs_fn(params: ModelParams, b: Tensor3D) -> Tensor2D:
            logpdfs, _ = self._logpdfs_and_aux_fn(params, b, complete_data)
            return logpdfs

        def _logliks_fn(params: ModelParams, b: Tensor3D) -> Tensor2D:
            psi = self.model_design.individual_effects_fn(params.gamma, data.x, b)
            long_logliks = super(type(self), self)._long_logliks(
                params, psi, complete_data
            )
            hazard_logliks = super(type(self), self)._hazard_logliks(
                params, psi, complete_data
            )
            return long_logliks + hazard_logliks

        info = Info(
            data=data,
            logpdfs_fn=_logpdfs_fn,
            logliks_fn=_logliks_fn,
            iteration=-1,
            model=self,
            sampler=sampler,
        )

        for job in jobs:
            job.init(info)

        def _do(state: Tensor3D, aux: tuple[torch.Tensor, ...]):
            nonlocal info

            info.b, (info.logliks, info.psi) = state, aux
            info.iteration += 1

            return run_jobs(jobs, info)

        # Main loop
        sampler.loop(
            cast(int, max_iterations),
            cast(int, n_steps),
            _do,
            desc="Running joint model",
            verbose=verbose,
        )

        # End things
        is_known = new_data in (self.data, None)
        metrics = self.metrics_ if is_known and self.metrics_ is not None else Metrics()

        info.iteration += 1
        for job in jobs:
            job.end(info, metrics)

        if is_known:
            self.metrics_ = metrics

        self._cache.clear_cache()

        match len(vars(metrics)):
            case 0:
                return None
            case 1:
                return next(iter(vars(metrics).values()))
            case _:
                return metrics

    @beartype
    def sample_params(self, sample_size: int) -> list[ModelParams]:
        """Sample parameters based on asymptotic behavior of the MLE.

        Args:
            sample_size (int): The desired sample size.

        Raises:
            ValueError: If the model has not been fitted, or FIM not computed.

        Returns:
            list[ModelParams]: A list of model parameters.
        """
        if not self.fit_:
            raise ValueError("Model must be fit")

        dist = torch.distributions.MultivariateNormal(
            self.params_.as_flat_tensor, self.fim.inverse()
        )
        flat_samples = dist.sample((sample_size,))

        return [params_like_from_flat(self.params_, sample) for sample in flat_samples]

    @property
    def fim(self) -> Tensor2D:
        """Returns the Fisher Information Matrix.

        Raises:
            ValueError: If Fisher Information Matrix has not yet been computed.

        Returns:
            Tensor2D: The Fisher Information Matrix.
        """
        if self.metrics_ is None or not hasattr(self.metrics_, "fim"):
            raise ValueError("Fisher Information Matrix must be previously computed.")

        return self.metrics_.fim

    @property
    def stderror(self) -> ModelParams:
        """Returns the standard error of the parameters.

        They can be used to draw confidence intervals.

        Returns:
            ModelParams: The standard error in the same format as the parameters.
        """
        return params_like_from_flat(self.params_, self.fim.inverse().diagonal().sqrt())
