from numbers import Integral, Real
from typing import cast

import torch
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
from torch.nn.utils import parameters_to_vector

from ..typedefs._data import CompleteModelData, ModelDesign
from ..typedefs._parameters import ModelParameters
from ._fit import FitMixin
from ._predict import PredictMixin


class MultiStateJointModel(BaseEstimator, FitMixin, PredictMixin):
    r"""A class of the nonlinear multistate joint model.

    Please note this class encompasses both the linear joint model and the standard
    joint model, but also allows for the modeling of multiple states assuming a semi
    Markov property. The model is defined by a set of longitudinal and hazard
    functions in `ModelDesign`, which are parameterized by a set of parameters in
    `ModelParams`. Parameters may also be shared (see the documentation).

    The model is fit using a stochastic gradient ascent algorithm, and the parameters
    are sampled using a Metropolis-Hastings algorithm.

    Dynamic prediciton is possible using the different prediction methods, that support
    dynamic prediction with both single and double Monte Carlo integration.

    For numerical integration, use `n_quad` to specify the number of nodes for the
    Gauss Legendre quadrature of hazard, and `n_bisect` to specify the number of
    bisection steps for the bisection algorithm for transition sampling.

    For caching, use `cache_limit` to specify the limit of the cache used in hazard
    computation, greatly reducing memory and CPU usage. None means infinite, 0 means
    no caching.

    For MCMC, use `n_chains` to specify the number of chains, `init_step_size` to
    specify the initial step size for the MCMC sampler, `adapt_rate` to specify the
    adaptation rate for the step size, `target_accept_rate` to specify the target
    acceptance rate, `n_warmup` to specify the number of warmup iterations, and
    `n_subsample` to specify the number of subsamples.

    For fitting, use `max_iter_fit` to specify the maximum number of iterations for
    stochastic gradient ascent, `n_samples_summary` to specify the number of samples to
    compute Fisher Information Matrix and model selection criteria, `tol` to specify the
    tolerance for the R2 convergence, and `window_size` to specify the window size for
    the R2 convergence. The longer, the more stable the R2 convergence. The default
    value seems a sweet spot, and the stopping criterion is scale-agnostic. Any
    optimizer can be through the `optimizer` parameter. If set to `None`, then fitting
    is not possible. Recommended to use `torch.optim.Adam` with a learning rate of
    0.1 to 1.0. A value of 0.5 is a good starting point.

    For printing, use `verbose` to specify whether to print the progress of the model
    fitting and predicting.

    Attributes:
        model_design (ModelDesign): The model design.
        params (ModelParams): The (variable) model parameters.
        optimizer (torch.optim.Optimizer | None): The optimizer.
        n_quad (int): The number of nodes for the Gauss Legendre quadrature of hazard.
        n_bisect (int): The number of bisection steps for the bisection algorithm.
        cache_limit (int | None): The limit of the cache used in hazard computation,
            greatly reducing memory and CPU usage. None means infinite, 0 means no
            caching.
        n_chains (int): The number of parallel MCMC chains.
        init_step_size (float): Kernel standard error in Metropolis.
        adapt_rate (float): Adaptation rate for the step_size.
        target_accept_rate (float): Mean acceptance target.
        n_warmup (int): The number of warmup iterations for the MCMC sampler.
        n_subsample (int): The number of subsamples for the MCMC sampler.
        max_iter_fit (int): The maximum number of iterations for stochastic gradient
            ascent.
        tol (float): The tolerance for the R2 convergence.
        window_size (int): The window size for the R2 convergence.
        n_samples_summary (int): The number of samples used to compute Fisher
            Information Matrix and model selection criteria.
        verbose (bool): Whether to print the progress of the model fitting.
        params_vector_history_ (list[torch.Tensor]): The history of model parameters.
        fim_ (torch.Tensor | None): The Fisher Information Matrix.
        loglik_ (float | None): The log likelihood.
        aic_ (float | None): The Akaike Information Criterion.
        bic_ (float | None): The Bayesian Information Criterion.

    Examples:
        >>> # Declares initial model
        >>> optimizer = torch.optim.Adam(init_params.parameters(), lr=0.5)
        >>> model = MultiStateJointModel(model_design, init_params, optimizer)
        >>> # Runs optimization process
        >>> model.fit(data)
        >>> # Prints a summary of the model
        >>> model.summary()
    """

    model_design: ModelDesign
    params: ModelParameters
    optimizer: torch.optim.Optimizer | None
    n_quad: int
    n_bisect: int
    cache_limit: int | None
    n_chains: int
    init_step_size: float
    adapt_rate: float
    target_accept_rate: float
    n_warmup: int
    n_subsample: int
    max_iter_fit: int
    tol: float
    window_size: int
    n_samples_summary: int
    verbose: bool
    vector_model_parameters_history_: list[torch.Tensor]
    fim_: torch.Tensor | None
    loglik_: float | None
    aic_: float | None
    bic_: float | None

    @validate_params(
        {
            "model_design": [ModelDesign],
            "model_parameters": [ModelParameters],
            "optimizer": [torch.optim.Optimizer, None],
            "n_quad": [Interval(Integral, 1, None, closed="left")],
            "n_bisect": [Interval(Integral, 1, None, closed="left")],
            "cache_limit": [Interval(Integral, 0, None, closed="left"), None],
            "n_chains": [Interval(Integral, 1, None, closed="left")],
            "init_step_size": [Interval(Real, 0, None, closed="neither")],
            "adapt_rate": [Interval(Real, 0, None, closed="left")],
            "target_accept_rate": [Interval(Real, 0, 1, closed="neither")],
            "n_warmup": [Interval(Integral, 0, None, closed="left")],
            "n_subsample": [Interval(Integral, 0, None, closed="left")],
            "max_iter_fit": [Interval(Integral, 1, None, closed="left")],
            "tol": [Interval(Real, 0, 1, closed="both")],
            "window_size": [Interval(Integral, 1, None, closed="left")],
            "n_samples_summary": [Interval(Integral, 1, None, closed="left")],
            "verbose": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        model_design: ModelDesign,
        model_parameters: ModelParameters,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        n_quad: int = 32,
        n_bisect: int = 32,
        cache_limit: int | None = 256,
        n_chains: int = 5,
        init_step_size: float = 0.1,
        adapt_rate: float = 0.01,
        target_accept_rate: float = 0.234,
        n_warmup: int = 100,
        n_subsample: int = 10,
        max_iter_fit: int = 500,
        tol: float = 0.1,
        window_size: int = 100,
        n_samples_summary: int = 500,
        verbose: bool = True,
    ):
        """Initializes the joint model based on the user defined design.

        Args:
            model_design (ModelDesign): Model design containing modeling information.
            model_parameters (ModelParameters): (Initial) values for the parameters.
            optimizer (torch.optim.Optimizer | None, optional): The optimizer   used for
                fitting. Defaults to None.
            n_quad (int, optional): The used number of points for Gauss-Legendre
                quadrature. Defaults to 32.
            n_bisect (int, optional): The number of bisection steps used in transition
                sampling. Defaults to 32.
            cache_limit (int | None, optional): The max length of cache. Defaults to
                256.
            n_chains (int, optional): The number of chains for MCMC. Defaults to 5.
            init_step_size (float, optional): The initial step size for the MCMC
                sampler. Defaults to 0.1.
            adapt_rate (float, optional): The adaptation rate for the MCMC sampler.
                Defaults to 0.01.
            target_accept_rate (float, optional): The target acceptance rate for the
                MCMC sampler. Defaults to 0.234.
            n_warmup (int, optional): The number of warmup iterations for the MCMC
                sampler. Defaults to 100.
            n_subsample (int, optional): The number of subsamples for the MCMC sampler.
                Defaults to 10.
            max_iter_fit (int, optional): The maximum number of iterations for
                stochastic gradient ascent. Defaults to 500.
            tol (float, optional): The tolerance for the convergence. Defaults to 0.1.
            window_size (int, optional): The window size for the convergence. Defaults
                to 100.
            n_samples_summary (int, optional): The number of samples used to compute
                Fisher Information Matrix and model selection criteria. Defaults to 500.
            verbose (bool, optional): Whether to print the progress of the model
                fitting. Defaults to True.
        """
        # Info of the Mixin Classes
        super().__init__(
            optimizer=optimizer,
            n_quad=n_quad,
            n_bisect=n_bisect,
            cache_limit=cache_limit,
            n_chains=n_chains,
            init_step_size=init_step_size,
            adapt_rate=adapt_rate,
            target_accept_rate=target_accept_rate,
            max_iter_fit=max_iter_fit,
            tol=tol,
            window_size=window_size,
            n_samples_summary=n_samples_summary,
        )

        # Store model components
        self.model_design = model_design
        self.model_parameters = model_parameters
        self.n_warmup = n_warmup
        self.n_subsample = n_subsample
        self.verbose = verbose
        self.vector_model_parameters_history_ = [
            parameters_to_vector(self.model_parameters.parameters()).detach()
        ]
        self.fim_ = None
        self.loglik_ = None
        self.aic_ = None
        self.bic_ = None

    def _logpdfs_aux_fn(
        self, data: CompleteModelData, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the log pdfs with individual effects and log likelihoods.

        Args:
            data (CompleteModelData): Dataset on which likelihood is computed.
            b (torch.Tensor): The random effects.

        Returns:
           tuple[torch.Tensor, torch.Tensor]: The log pdfs and aux.
        """
        psi = self.model_design.individual_effects_fn(self.params.gamma, data.x, b)
        logpdfs = (
            super()._long_logliks(data, psi)
            + super()._hazard_logliks(data, psi)
            + super()._prior_logliks(b)
        )

        return logpdfs, psi

    @property
    def stderr(self) -> torch.Tensor:
        r"""Returns the estimated standard error of the parameters as a vector.

        They can be used to draw confidence intervals. The standard errors are computed
        using the diagonal of the inverse of the inverse Fisher Information Matrix at
        the MLE

        .. math::
            \mathrm{stderr} = \sqrt{\operatorname{diag}\left( \hat{\mathcal{I}}_n
            (\hat{\theta})^{-1} \right)}

        Raises:
            ValueError: If the model is not fitted.

        Returns:
            torch.Tensor: The estimated standard error as a vector.
        """
        check_is_fitted(self, "fim_")

        return cast(torch.Tensor, self.fim_).inverse().diag().sqrt()
