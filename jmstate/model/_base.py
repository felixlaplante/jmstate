from bisect import bisect_left
from numbers import Integral, Real

import torch
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from torch.distributions import Normal

from ..typedefs._data import CompleteModelData, ModelDesign
from ..typedefs._defs import SIGNIFICANCE_CODES, SIGNIFICANCE_LEVELS
from ..typedefs._params import ModelParams
from ..visualization._print import rich_str
from ._fit import FitMixin
from ._hazard import HazardMixin
from ._longitudinal import LongitudinalMixin
from ._prediction import PredictionMixin
from ._prior import PriorMixin


class MultiStateJointModel(
    PriorMixin, LongitudinalMixin, HazardMixin, FitMixin, PredictionMixin
):
    r"""A class of the nonlinear multistate joint model.

    Please note this class encompasses both the linear joint model and the standard
    joint model, but also allows for the modeling of multiple states assuming a semi
    Markov property. The model is defined by a set of longitudinal and hazard
    functions, which are parameterized by a set of parameters.

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

    For fitting, use `n_iter_fit` to specify the number of iterations for stochastic
    gradient ascent, `n_iter_summary` to specify the number of iterations to compute
    Fisher Information Matrix and criteria, `lr` to specify the learning rate for the
    optimizer, `tol` to specify the tolerance for the R2 convergence, and `window_size`
    to specify the window size for the R2 convergence. The longer, the more stable
    the R2 convergence. The default value seems a sweet spot, and the stopping criterion
    is scale-agnostic.

    For printing, use `verbose` to specify whether to print the progress of the model
    fitting and predicting.

    Attributes:
        model_design (ModelDesign): The model design.
        params_ (ModelParams): The (variable) model parameters.
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
        n_iter_fit (int): The number of iterations for stochastic gradient ascent.
        n_iter_summary (int): The number of iterations to compute Fisher Information
            Matrix and criteria.
        lr (float): The learning rate for the optimizer.
        tol (float): The tolerance for the R2 convergence.
        window_size (int): The window size for the R2 convergence.
        verbose (bool): Whether to print the progress of the model fitting.
        params_history_ (list[ModelParams]): The history of model parameters.
        fim_ (torch.Tensor | None): The Fisher Information Matrix.
        loglik_ (float | None): The log likelihood.
        aic_ (float | None): The Akaike Information Criterion.
        bic_ (float | None): The Bayesian Information Criterion.
    """

    model_design: ModelDesign
    params_: ModelParams
    n_quad: int
    n_bisect: int
    cache_limit: int | None
    n_chains: int
    init_step_size: float
    adapt_rate: float
    target_accept_rate: float
    n_warmup: int
    n_subsample: int
    n_iter_fit: int
    n_iter_summary: int
    lr: float
    tol: float
    window_size: int
    verbose: bool
    params_history_: list[ModelParams]
    fim_: torch.Tensor | None
    loglik_: float | None
    aic_: float | None
    bic_: float | None

    @validate_params(
        {
            "model_design": [ModelDesign],
            "init_params": [ModelParams],
            "n_quad": [Interval(Integral, 1, None, closed="left")],
            "n_bisect": [Interval(Integral, 1, None, closed="left")],
            "cache_limit": [Interval(Integral, 0, None, closed="left"), None],
            "n_chains": [Interval(Integral, 1, None, closed="left")],
            "init_step_size": [Interval(Real, 0, None, closed="neither")],
            "adapt_rate": [Interval(Real, 0, None, closed="left")],
            "target_accept_rate": [Interval(Real, 0, 1, closed="neither")],
            "n_warmup": [Interval(Integral, 0, None, closed="left")],
            "n_subsample": [Interval(Integral, 0, None, closed="left")],
            "n_iter_fit": [Interval(Integral, 1, None, closed="left")],
            "n_iter_summary": [Interval(Integral, 1, None, closed="left")],
            "lr": [Interval(Real, 0, None, closed="neither")],
            "tol": [Interval(Real, 0, 1, closed="both")],
            "window_size": [Interval(Integral, 1, None, closed="left")],
            "verbose": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        model_design: ModelDesign,
        init_params: ModelParams,
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
        n_iter_fit: int = 500,
        n_iter_summary: int = 100,
        lr: float = 0.5,
        tol: float = 0.1,
        window_size: int = 100,
        verbose: bool = True,
    ):
        """Initializes the joint model based on the user defined design.

        Args:
            model_design (ModelDesign): Model design containing modeling information.
            init_params (ModelParams): Initial values for the parameters.
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
            n_iter_fit (int, optional): The number of iterations for stochastic
                gradient ascent. Defaults to 500.
            n_iter_summary (int, optional): The number of iterations to compute Fisher
                Information Matrix and criteria. Defaults to 100.
            lr (float, optional): The learning rate for the optimizer. Defaults to 0.5.
            tol (float, optional): The tolerance for the convergence. Defaults to 0.1.
            window_size (int, optional): The window size for the convergence. Defaults
                to 100.
            verbose (bool, optional): Whether to print the progress of the model
                fitting. Defaults to True.
        """
        # Info of the Mixin Classes
        super().__init__(
            n_quad=n_quad,
            n_bisect=n_bisect,
            cache_limit=cache_limit,
            n_chains=n_chains,
            init_step_size=init_step_size,
            adapt_rate=adapt_rate,
            target_accept_rate=target_accept_rate,
            lr=lr,
            tol=tol,
            window_size=window_size,
        )

        # Store model components
        self.model_design = model_design
        self.params_ = init_params.clone().detach()
        self.n_warmup = n_warmup
        self.n_subsample = n_subsample
        self.n_iter_fit = n_iter_fit
        self.n_iter_summary = n_iter_summary
        self.verbose = verbose
        self.params_history_ = [self.params_.clone()]
        self.fim_ = None
        self.loglik_ = None
        self.aic_ = None
        self.bic_ = None

    def _logpdfs_aux_fn(  # type: ignore
        self, params: ModelParams, data: CompleteModelData, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the log pdfs with individual effects and log likelihoods.

        Args:
            params (ModelParams): The model parameters.
            data (CompleteModelData): Dataset on which likelihood is computed.
            b (torch.Tensor): The random effects.

        Returns:
           tuple[torch.Tensor, torch.Tensor]: The log pdfs and aux.
        """
        psi = self.model_design.individual_effects_fn(params.gamma, data.x, b)
        logpdfs = (
            super()._long_logliks(params, data, psi)
            + super()._hazard_logliks(params, data, psi)
            + super()._prior_logliks(params, b)
        )

        return logpdfs, psi

    def __str__(self) -> str:
        """Returns a string representation of the model.

        Returns:
            str: The string representation.
        """
        tree = Tree("MultiStateJointModel")
        tree.add(f"model_design: {self.model_design}")
        tree.add(f"params_: {self.params_}")
        tree.add(f"n_quad: {self.n_quad}")
        tree.add(f"n_bisect: {self.n_bisect}")
        tree.add(f"cache_limit: {self.cache_limit}")
        tree.add(f"n_warmup: {self.n_warmup}")
        tree.add(f"n_subsample: {self.n_subsample}")
        tree.add(f"n_iter_fit: {self.n_iter_fit}")
        tree.add(f"n_iter_summary: {self.n_iter_summary}")
        tree.add(f"tol: {self.tol}")
        tree.add(f"window_size: {self.window_size}")
        tree.add(f"params_history_: {len(self.params_history_)} element(s)")
        tree.add(f"fim_: {self.fim_}")
        tree.add(f"loglik_: {self.loglik_}")
        tree.add(f"aic_: {self.aic_}")
        tree.add(f"bic_: {self.bic_}")

        return rich_str(tree)

    def print_summary(self):
        """Prints a summary of the fitted model.

        This function prints the p-values of the parameters as well as values and
        standard error. Also prints the log likelihood, AIC, BIC with lovely colors.
        """
        values = self.params_.as_flat_tensor
        stderrs = self.stderr.as_flat_tensor
        zvalues = torch.abs(values / stderrs)
        pvalues = 2 * (1 - Normal(0, 1).cdf(zvalues))

        table = Table()
        table.add_column("Parameter name", justify="left")
        table.add_column("Value", justify="center")
        table.add_column("Standard Error", justify="center")
        table.add_column("z-value", justify="center")
        table.add_column("p-value", justify="center")
        table.add_column("Significance level", justify="center")

        i = 0
        for name, value in self.params_.as_dict.items():
            for j in range(value.numel()):
                code = SIGNIFICANCE_CODES[
                    bisect_left(SIGNIFICANCE_LEVELS, pvalues[i].item())
                ]

                table.add_row(
                    f"{name}[{j}]",
                    f"{values[i].item():.3f}",
                    f"{stderrs[i].item():.3f}",
                    f"{zvalues[i].item():.3f}",
                    f"{pvalues[i].item():.3f}",
                    code,
                )
                i += 1

        criteria = Text(
            f"Log-likelihood: {self.loglik_:.3f}\n"
            f"AIC: {self.aic_:.3f}\n"
            f"BIC: {self.bic_:.3f}",
            style="bold cyan",
        )

        content = Group(table, Rule(style="dim"), criteria, Rule(style="dim"))

        panel = Panel(
            content, title="Model Summary", border_style="green", expand=False
        )

        Console().print(panel)

    @validate_params(
        {
            "sample_size": [Interval(Integral, 1, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def sample_params(self, sample_size: int) -> list[ModelParams]:
        """Sample parameters based on asymptotic behavior of the MLE.

        Args:
            sample_size (int): The desired sample size.

        Raises:
            ValueError: If Fisher Information Matrix has not been computed.

        Returns:
            list[ModelParams]: A list of sampled model parameters.
        """
        if self.fim_ is None:
            raise ValueError("Fisher Information Matrix must be previously computed.")

        dist = torch.distributions.MultivariateNormal(
            self.params_.as_flat_tensor, self.fim_.inverse()
        )
        flat_samples = dist.sample((sample_size,))

        return [self.params_.from_flat_tensor(sample) for sample in flat_samples]

    @property
    def stderr(self) -> ModelParams:
        r"""Returns the standard error of the parameters.

        They can be used to draw confidence intervals. The standard errors are computed
        using the diagonal of the inverse of the inverse Fisher Information Matrix at
        the MLE:

        .. math::
            \text{sd} = \sqrt{\operatorname{diag}\left( \mathcal{I}(\hat{\theta})^{-1}
            \right)}

        Raises:
            ValueError: If Fisher Information Matrix has not been computed.

        Returns:
            ModelParams: The standard error in the same format as the parameters.
        """
        if self.fim_ is None:
            raise ValueError("Fisher Information Matrix must be previously computed.")

        return self.params_.from_flat_tensor(self.fim_.inverse().diag().sqrt())
