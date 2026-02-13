from bisect import bisect_left
from collections.abc import Callable
from typing import Self

import torch
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from torch.distributions import Normal
from tqdm import trange

from ..typedefs._data import CompleteModelData, ModelData, ModelDesign, SampleData
from ..typedefs._defs import SIGNIFICANCE_CODES, SIGNIFICANCE_LEVELS, Trajectory
from ..typedefs._params import ModelParams
from ..utils._checks import check_consistent_size, check_inf, check_nan
from ..visualization._print import rich_str
from ._fit import FitMixin
from ._hazard import HazardMixin
from ._longitudinal import LongitudinalMixin
from ._prior import PriorMixin
from ._sampler import MCMCMixin


class MultiStateJointModel(
    PriorMixin, LongitudinalMixin, HazardMixin, MCMCMixin, FitMixin
):
    r"""A class of the nonlinear multistate joint model.

    It features methods to simulate data, fit based on stochastic gradient with any
    `torch.optim.Optimizer` of choice.

    It leverages the Fisher identity and stochastic gradient algorithm coupled
    with a MCMC (Metropolis-Hastings) sampler:

    .. math::
        \nabla_\theta \log \mathcal{L}(\theta ; x) = \mathbb{E}_{b \sim p(\cdot \mid x,
        \theta)} \left( \nabla_\theta \log \mathcal{L}(\theta ; x, b) \right).

    The use of penalization is possible through the attribute `pen`, which is multiplied
    by the number of samples.

    Please note this class encompasses both the linear joint model and the standard
    joint model, but also allows for the modeling of multiple states assuming a semi
    Markov property.

    Attributes:
        model_design (ModelDesign): The model design.
        params_ (ModelParams): The (variable) model parameters.
        pen (Callable[[ModelParams], torch.Tensor] | None): The log likelihood penalty.
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
        n_steps (int): The number of steps for the MCMC sampler.
        n_iters (tuple[int, int]): The number of iterations for stochastic gradient and
            MCMC.
        lr (float): The learning rate for the optimizer.
        atol (float): The absolute tolerance for convergence.
        rtol (float): The relative tolerance for convergence.
        verbose (bool): Whether to print the progress of the model fitting.
        params_history_ (list[ModelParams]): The history of model parameters.
        fim_ (torch.Tensor | None): The Fisher Information Matrix.
        loglik_ (float | None): The log likelihood.
        nloglik_pen (float | None): The penalized negative log likelihood.
        aic_ (float | None): The Akaike Information Criterion.
        bic_ (float | None): The Bayesian Information Criterion.
    """

    model_design: ModelDesign
    params_: ModelParams
    pen: Callable[[ModelParams], torch.Tensor] | None
    n_quad: int
    n_bisect: int
    cache_limit: int | None
    n_chains: int
    init_step_size: float
    adapt_rate: float
    target_accept_rate: float
    n_warmup: int
    n_steps: int
    n_iters: tuple[int, int]
    lr: float
    tols: tuple[float, float]
    verbose: bool
    params_history_: list[ModelParams]
    fim_: torch.Tensor | None
    loglik_: float | None
    nloglik_pen: float | None
    aic_: float | None
    bic_: float | None

    def __init__(
        self,
        model_design: ModelDesign,
        init_params: ModelParams,
        *,
        pen: Callable[[ModelParams], torch.Tensor] | None = None,
        n_quad: int = 32,
        n_bisect: int = 32,
        cache_limit: int | None = 256,
        n_chains: int = 5,
        init_step_size: float = 0.1,
        adapt_rate: float = 0.01,
        target_accept_rate: float = 0.234,
        n_warmup: int = 100,
        n_steps: int = 10,
        n_iters: tuple[int, int] = (500, 100),
        lr: float = 0.5,
        tols: tuple[float, float] = (1e-6, 1e-1),
        verbose: bool = True,
    ):
        """Initializes the joint model based on the user defined design.

        Args:
            model_design (ModelDesign): Model design containing modeling information.
            init_params (ModelParams): Initial values for the parameters.
            pen (Callable[[ModelParams], torch.Tensor] | None, optional):
                The penalization function. Defaults to None.
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
            n_steps (int, optional): The number of steps for the MCMC sampler. Defaults
                to 10.
            n_iters (tuple[int, int], optional): The number of iterations for stochastic
                gradient and MCMC. Defaults to (500, 100).
            lr (float, optional): The learning rate for the optimizer. Defaults to 0.5.
            tols (tuple[float, float], optional): The absolute and relative tolerances
                for convergence. Defaults to (1e-6, 1e-1).
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
            tols=tols,
        )

        # Store model components
        self.model_design = model_design
        self.params_ = init_params.clone()
        self.pen = pen
        self.n_warmup = n_warmup
        self.n_steps = n_steps
        self.n_iters = n_iters
        self.verbose = verbose
        self.params_history_ = [self.params_.clone().detach()]
        self.fim_ = None
        self.loglik_ = None
        self.nloglik_pen = None
        self.aic_ = None
        self.bic_ = None

    def __str__(self) -> str:
        """Returns a string representation of the model.

        Returns:
            str: The string representation.
        """
        tree = Tree("MultiStateJointModel")
        tree.add(f"model_design: {self.model_design}")
        tree.add(f"params_: {self.params_}")
        tree.add(f"pen: {self.pen}")
        tree.add(f"n_quad: {self.n_quad}")
        tree.add(f"n_bisect: {self.n_bisect}")
        tree.add(f"cache_limit: {self.cache_limit}")
        tree.add(f"params_history_: {len(self.params_history_)} element(s)")
        tree.add(f"fim_: {self.fim_}")
        tree.add(f"loglik_: {self.loglik_}")
        tree.add(f"nloglik_pen: {self.nloglik_pen}")
        tree.add(f"aic_: {self.aic_}")
        tree.add(f"bic_: {self.bic_}")
        tree.add(f"n_warmup: {self.n_warmup}")
        tree.add(f"n_iters: {self.n_iters}")
        tree.add(f"tols: {self.tols}")

        return rich_str(tree)

    def summary(self):
        """Prints a summary of the model.

        This function prints the p-values of the parameters as well as values and
        standard error. Also prints the log likelihood, AIC, BIC with lovely colors!
        """
        values = self.params_.as_flat_tensor
        stderrors = self.stderror.as_flat_tensor
        zvalues = torch.abs(values / stderrors)
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
                    f"{stderrors[i].item():.3f}",
                    f"{zvalues[i].item():.3f}",
                    f"{pvalues[i].item():.3f}",
                    code,
                )
                i += 1

        criteria = Text(
            f"Log-likelihood: {self.loglik_:.3f}\n"
            f"Penalized negative log-likelihood: {self.nloglik_pen:.3f}\n"
            f"AIC: {self.aic_:.3f}\n"
            f"BIC: {self.bic_:.3f}",
            style="bold cyan",
        )

        content = Group(table, Rule(style="dim"), criteria, Rule(style="dim"))

        panel = Panel(
            content, title="Model Summary", border_style="green", expand=False
        )

        Console().print(panel)

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

    def sample_trajectories(
        self,
        sample_data: SampleData,
        c_max: torch.Tensor,
        *,
        max_length: int = 10,
    ) -> list[Trajectory]:
        """Sample trajectories from the joint model.

        The sampling is done usign a bisection algorithm by inversing the log cdf of the
        transitions inside a Gillespie-like algorithm.

        Checks are run only if the `skip_validation` attribute of `sample_date` is not
        set to `True`.

        Args:
            sample_data (SampleData): Prediction data.
            c_max (TensorCol): The maximum trajectory censoring times.
            max_length (IntStrictlyPositive, optional): Maximum iterations or sampling.
                Defaults to 10.

        Raises:
            ValueError: If c_max contains inf values.
            ValueError: If c_max contains NaN values.
            ValueError: If c_max has incorrect shape.

        Returns:
            list[Trajectory]: The sampled trajectories.
        """
        if not sample_data.skip_validation:
            check_inf(((c_max, "c_max"),))
            check_nan(((c_max, "c_max"),))
            check_consistent_size(
                ((c_max, 0, "c_max"), (sample_data.size, None, "sample_data.size"))
            )

        return super()._sample_trajectories(sample_data, c_max, max_length=max_length)

    def compute_surv_logps(
        self, sample_data: SampleData, u: torch.Tensor
    ) -> torch.Tensor:
        r"""Computes log probabilites of remaining event free up to time u.

        A censoring time may also be given. With known individual effects, this computes
        at the times :math:`u` the values of the log survival probabilities given input
        data conditionally to survival up to time :math:`c`:

        .. math::
            \log \mathbb{P}(T^* \geq u \mid T^* > c) = -\int_c^u \lambda(t) \, dt.

        When multiple transitions are allowed, :math:`\lambda(t)` is a sum over all
        possible transitions, that is to say if an individual is in the state :math:`k`
        from time :math:`t_0`, this gives:

        .. math::
            -\int_c^u \sum_{k'} \lambda^{k' \mid k}(t \mid t_0) \, dt.

        Please note this makes use of the Chasles property in order to avoid the
        computation of two integrals and make computations more precise.

        The variable `u` is expected to be a matrix with the same number of rows as
        individuals, and the same number of columns as prediction times.

        Checks are run only if the `skip_validation` attribute of `sample_date` is not
        set to `True`.

        Args:
            sample_data (SampleData): The data on which to compute the probabilities.
            u (torch.Tensor): The time at which to evaluate the probabilities.

        Raises:
            ValueError: If u contains inf values.
            ValueError: If u contains NaN values.
            ValueError: If u has incorrect shape.

        Returns:
            torch.Tensor: The computed survival log probabilities.
        """
        if not sample_data.skip_validation:
            check_inf(((u, "u"),))
            check_nan(((u, "u"),))
            check_consistent_size(
                ((u, 0, "u"), (sample_data.size, None, "sample_data.size"))
            )

        return super()._compute_surv_logps(sample_data, u)

    def fit(self, data: ModelData) -> Self:
        # Load and complete data
        data = CompleteModelData(
            data.x, data.t, data.y, data.trajectories, data.c, skip_validation=True
        )
        data.prepare(self.model_design, self.params_)

        # Initialize optimizer and MCMC
        optimizer = self._init_optimizer()
        sampler = self._init_mcmc(data)
        sampler.run(self.n_warmup)

        # Main fitting loop
        for _ in trange(
            self.n_iters[0], desc="Fitting joint model", disable=not self.verbose
        ):
            self._step(optimizer, sampler, data)
            self.params_history_.append(self.params_.clone().detach())
            if self._is_converged(optimizer):
                break
            sampler.run(self.n_steps)

        # Initialize Jacobian matrix and criteria
        mjac, jac_fn = self._init_jac(data)
        logpdf, mb, mb2 = self._init_criteria(data)

        # FIM and Criteria loop
        for _ in trange(
            self.n_iters[1],
            desc="Computing FIM and Criteria",
            disable=not self.verbose,
        ):
            self._update_jac(mjac, jac_fn, sampler)
            self._update_criteria(logpdf, mb, mb2, sampler)
            for _ in range(self.n_steps):
                sampler.step()

        self.fim_ = self._compute_fim(mjac)
        self.loglik, self.nloglik_pen, self.aic, self.bic = self._compute_criteria(
            logpdf, mb, mb2, self.fim_, data
        )

        self._cache.clear_cache()
        return self

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
    def stderror(self) -> ModelParams:
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
