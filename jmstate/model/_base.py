import copy
from typing import Any, Callable, SupportsFloat, cast

import torch
from beartype import beartype

from ..typedefs._data import CompleteModelData, ModelData, ModelDesign
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
from ..utils._misc import do_jobs, params_like_from_flat
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
        self.data: ModelData | None = None
        self.metrics_: Metrics | None = None
        self.fit_ = False

    def _logliks_and_aux(
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

        def _prior_logliks(b: torch.Tensor) -> torch.Tensor:
            Q_inv_cholesky, Q_log_eigvals = get_cholesky_and_log_eigvals(params, "Q")
            Q_quad_form = (b @ Q_inv_cholesky).pow(2).sum(dim=-1)
            Q_log_det = (Q_log_eigvals - LOGTWOPI).sum()

            return 0.5 * (Q_log_det - Q_quad_form)

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
            (n_chains, data.size, self.params_.Q_dim_), dtype=torch.float32
        )

        return MetropolisHastingsSampler(
            lambda b: self._logliks_and_aux(self.params_, b, data),
            init_b,
            n_chains,
            init_step_size,
            adapt_rate,
            target_accept_rate,
        )

    @beartype
    def do(
        self,
        new_data: ModelData | None = None,
        *,
        jobs: Job | list[Job],
        n_iterations: int = 200,
        n_chains: int = 10,
        init_step_size: SupportsFloat = 0.1,
        adapt_rate: SupportsFloat = 0.1,
        accept_target: SupportsFloat = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
        verbose: bool = True,
    ) -> Metrics | Any | None:
        """Runs the MultiStateJointModel loop and some jobs.

        Args:
            new_data (ModelData): The dataset to learn from.
            jobs (Job | list[Job]): A list of jobs to execute in order.
            n_iterations (int, optional): Number of iterations for optimization. Defaults to 200.
            n_chains (int, optional): Batch size used. Defaults to 10.
            init_step_size (SupportsFloat, optional): Kernel step in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (SupportsFloat, optional): Adaptation rate for the step_size. Defaults to 0.01.
            accept_target (SupportsFloat, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
            verbose (bool, optional): Wheter or not to show progress. Defaults to True.

        Raises:
            ValueError: If n_iterations is not greater than 0.
            ValueError: If n_chains is not greater than 0.
            TypeError: If some callback is not callable.

        Returns:
            dict[str, Any] | Any | None: The metrics dict, its single element, pr None.
        """
        # Verify n_chains
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be greater than 0, got {n_iterations}")
        if n_chains < 1:
            raise ValueError(f"n_chains must be greater than 0, got {n_chains}")
        # Load and complete data
        if new_data is None and self.data is None:
            raise ValueError("data must not be None if self.data is None")

        # Repeat data to do minibatch in a vectorized fashion
        data = cast(ModelData, new_data if new_data is not None else self.data)
        complete_data = CompleteModelData(
            data.x, data.t, data.y, data.trajectories, data.c, skip_validation=True
        )
        complete_data.init(self.model_design)

        # Set up MCMC
        sampler = self._setup_mcmc(
            complete_data, n_chains, init_step_size, adapt_rate, accept_target
        )
        sampler.warmup(init_warmup)

        # Initialize metrics
        metrics = Metrics()

        def _logpdfs_fn(params: ModelParams, b: Tensor3D):
            return self._logliks_and_aux(params, b, complete_data)[0]

        info = Info(
            data=data,
            logpdfs_fn=_logpdfs_fn,
            iteration=-1,
            n_iterations=n_iterations,
            model=self,
            sampler=sampler,
        )
        do_jobs("init", jobs, info, metrics)

        def _do(iteration: int):
            nonlocal info

            info.b, (info.logliks, info.psi) = sampler.step()
            info.iteration = iteration

            do_jobs("run", jobs, info, metrics)

        # Main loop
        sampler.loop(
            n_iterations, cont_warmup, _do, desc="Running joint model", verbose=verbose
        )

        info.iteration += 1
        do_jobs("end", jobs, info, metrics)

        self._cache.clear_cache()

        if new_data is None:
            self.metrics_ = metrics

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
            ValueError: If the model has not been fitted, or Fisher Information Matrix not computed.

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
        return params_like_from_flat(
            self.params_, torch.sqrt(self.fim.inverse().diag())
        )
