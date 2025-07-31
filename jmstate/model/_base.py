import copy
from typing import Any, Callable, cast

import torch

from ..typedefs._data import CompleteModelData, ModelData
from ..typedefs._defs import LOGTWOPI, Info, Job, Metrics, ModelDesign
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
    pen: Callable[[ModelParams], torch.Tensor] | None
    n_quad: int
    n_bissect: int
    enable_cache: bool
    cache_limit: int | None
    data: ModelData | None
    metrics_: Metrics | None
    fit_: bool

    def __init__(
        self,
        model_design: ModelDesign,
        init_params: ModelParams,
        *,
        pen: Callable[[ModelParams], torch.Tensor] | None = None,
        n_quad: int = 32,
        n_bissect: int = 32,
        enable_cache: bool = True,
        cache_limit: int | None = 256,
    ):
        """Initializes the joint model based on the user defined design.

        Args:
            model_design (ModelDesign): Model design containing modeling information.
            init_params (ModelParams): Initial values for the parameters.
            pen (Callable[[ModelParams], torch.Tensor] | None, optional): The penalization function. Defaults to None.
            n_quad (int, optional): The used numnber of points for Gauss-Legendre quadrature. Defaults to 32.
            n_bissect (int, optional): The number of bissection steps used in transition sampling. Defaults to 32.
            enable_cache (bool, optional): Enables caching. Defaults to True.
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
            enable_cache=enable_cache,
            cache_limit=cache_limit,
        )

        # Initialize attributes that will be set later
        self.data: ModelData | None = None
        self.metrics_: Metrics | None = None
        self.fit_ = False

    def _logliks(
        self, b: torch.Tensor, data: CompleteModelData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the total log likelihood up to a constant.

        Args:
            b (torch.Tensor): The individual random effects.
            data (CompleteModelData): Dataset on which the likeihood is computed.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The computed total log logpdf/likelihood.
        """

        def _prior_logliks(b: torch.Tensor) -> torch.Tensor:
            Q_inv_cholesky, Q_log_eigvals = get_cholesky_and_log_eigvals(
                self.params_, "Q"
            )

            Q_quad_form = (b @ Q_inv_cholesky).pow(2).sum(dim=1)
            Q_log_det = (Q_log_eigvals - LOGTWOPI).sum()

            return 0.5 * (Q_log_det - Q_quad_form)

        # Transform random effects to individual-specific parameters
        psi = self.model_design.individual_effects_fn(self.params_.gamma, data.x, b)

        # Compute individual likelihood components
        long_logliks = super()._long_logliks(psi, data)
        hazard_logliks = super()._hazard_logliks(psi, data)
        prior_logliks = _prior_logliks(b)

        # Sum all likelihood components
        logliks = long_logliks + hazard_logliks
        logpdfs = logliks + prior_logliks

        return logpdfs, logliks

    def _setup_mcmc(
        self,
        data: CompleteModelData,
        init_step_size: float,
        adapt_rate: float,
        target_accept_rate: float,
    ) -> MetropolisHastingsSampler:
        """Setup the MCMC kernel and hyperparameters.

        Args:
            data (CompleteModelData): The dataset on which the likelihood is to be computed.
            init_step_size (float, optional): Kernel standard error in Metropolis Hastings.
            adapt_rate (float, optional): Adaptation rate for the step_size.
            target_accept_rate (float, optional): Mean acceptance target.

        Returns:
            MetropolisHastingsSampler: The intialized Markov kernel.
        """
        # Initialize random effects
        init_b = torch.zeros((data.size, self.params_.Q_dim_), dtype=torch.float32)

        return MetropolisHastingsSampler(
            lambda b: self._logliks(b, data),
            init_b,
            init_step_size,
            adapt_rate,
            target_accept_rate,
        )

    def do(
        self,
        new_data: ModelData | None = None,
        *,
        jobs: Job | list[Job],
        n_iterations: int = 2000,
        batch_size: int = 1,
        init_step_size: float = 0.1,
        adapt_rate: float = 0.01,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
        verbose: bool = True,
    ) -> Any | Metrics | None:
        """Runs the MultiStateJointModel loop and some jobs.

        Args:
            new_data (ModelData): The dataset to learn from.
            jobs (Job | list[Job]): A list of jobs to execute in order.
            n_iterations (int, optional): Number of iterations for optimization. Defaults to 2000.
            batch_size (int, optional): Batch size used. Defaults to 1.
            init_step_size (float, optional): Kernel standard error in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the step_size. Defaults to 0.01.
            accept_target (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
            verbose (bool, optional): Wheter or not to show progress. Defaults to True.

        Raises:
            ValueError: If n_iterations is not greater than 0.
            ValueError: If batch_size is not greater than 0.
            TypeError: If some callback is not callable.

        Returns:
            Any | dict[str, Any] | None: The metrics dict, or its single element, possibly None if none were recorded.
        """
        # Verify batch_size
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be greater than 0, got {n_iterations}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be greater than 0, got {batch_size}")

        # Load and complete data
        if new_data is None and self.data is None:
            raise ValueError("data must not be None if self.data is None")

        # Repeat data to do minibatch in a vectorized fashion
        data = cast(ModelData, new_data if new_data is not None else self.data)
        x_rep = data.x.repeat(batch_size, 1) if data.x is not None else None
        t_rep = data.t if data.t.ndim == 1 else data.t.repeat(batch_size, 1)
        y_rep = data.y.repeat(batch_size, 1, 1)
        trajectories_rep = data.trajectories * batch_size
        c_rep = data.c.repeat(batch_size)

        data_rep = CompleteModelData(
            x_rep, t_rep, y_rep, trajectories_rep, c_rep, skip_validation=True
        )
        data_rep.prepare(self.model_design)

        # Set up MCMC
        sampler = self._setup_mcmc(data_rep, init_step_size, adapt_rate, accept_target)
        sampler.warmup(init_warmup)

        # Initialize metrics
        metrics = Metrics()

        info = Info(
            data=data,
            iteration=-1,
            n_iterations=n_iterations,
            batch_size=batch_size,
            model=self,
            sampler=sampler,
        )
        do_jobs("init", jobs, info, metrics)

        def _do(iteration: int):
            nonlocal info

            (info.b, info.logpdfs), info.logliks = sampler.step()
            info.iteration = iteration

            do_jobs("run", jobs, info, metrics)

        # Main loop
        sampler.loop(
            n_iterations, cont_warmup, _do, desc="Running joint model", verbose=verbose
        )

        info.iteration += 1
        do_jobs("end", jobs, info, metrics)

        self.clear_cache()

        if new_data is None:
            self.metrics_ = metrics

        match len(vars(metrics)):
            case 0:
                return None
            case 1:
                return next(iter(vars(metrics).values()))
            case _:
                return metrics

    @property
    def fim(self) -> torch.Tensor:
        """Returns the Fisher Information Matrix.

        Raises:
            ValueError: If Fisher Information Matrix has not yet been computed.

        Returns:
            torch.Tensor: The Fisher Information Matrix.
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
