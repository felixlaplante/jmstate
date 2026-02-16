from collections.abc import Callable
from math import ceil
from typing import Any, Self, cast
from warnings import warn

import torch
from sklearn.utils._param_validation import validate_params  # type: ignore
from torch.func import functional_call, jacfwd  # type: ignore
from torch.nn.utils import parameters_to_vector
from tqdm import trange

from ..types._data import ModelData, ModelDataUnchecked, ModelDesign
from ..types._parameters import ModelParameters, UniqueParametersNNModule
from ..utils._cache import Cache
from ._hazard import HazardMixin
from ._longitudinal import LongitudinalMixin
from ._prior import PriorMixin
from ._sampler import MCMCMixin, MetropolisHastingsSampler


class FitMixin(
    PriorMixin, LongitudinalMixin, HazardMixin, MCMCMixin, UniqueParametersNNModule
):
    """Mixin for fitting the model."""

    design: ModelDesign
    params: ModelParameters
    optimizer: torch.optim.Optimizer | None
    n_warmup: int
    n_subsample: int
    max_iter_fit: int
    tol: float
    window_size: int
    n_samples_summary: int
    verbose: bool | int
    params_history_: list[torch.Tensor]
    fim_: torch.Tensor | None
    loglik_: float | None
    aic_: float | None
    bic_: float | None
    _cache: Cache

    def __init__(
        self,
        optimizer: torch.optim.Optimizer | None,
        max_iter_fit: int,
        tol: float,
        window_size: int,
        n_samples_summary: int,
        *args: Any,
        **kwargs: Any,
    ):
        """Initializes the fit parameters.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.
            max_iter_fit (int): The maximum number of iterations for fitting.
            tol (float): The tolerance for the convergence.
            window_size (int): The window size for the convergence.
            n_samples_summary (int): The number of samples used to compute Fisher
                Information Matrix and model selection criteria.
        """
        super().__init__(*args, **kwargs)

        self.optimizer = optimizer
        self.max_iter_fit = max_iter_fit
        self.tol = tol
        self.window_size = window_size
        self.n_samples_summary = n_samples_summary

    def forward(self, data: ModelData, b: torch.Tensor) -> torch.Tensor:
        """Computes the mean log pdfs for the random effects `b`.

        This is used to compute the Fisher Information Matrix, as
        `torch.func.functional_call` requires the function to implement a `forward`
        method that takes the parameters as the first argument. This can also be used
        to compute the log likelihood of the model manually by integrating it over
        the random effects with Gauss-Hermite quadrature.

        Args:
            data (ModelData): Dataset on which likelihood is computed.
            b (torch.Tensor): The random effects.

        Returns:
           torch.Tensor: The log pdfs.
        """
        logpdfs, _ = self._logpdfs_psi_fn(data, b)
        return logpdfs if logpdfs.ndim == 1 else logpdfs.mean(dim=0)

    def _step(
        self,
        sampler: MetropolisHastingsSampler,
        data: ModelData,
    ):
        """Performs a step of the optimizer.

        Args:
            sampler (MetropolisHastingsSampler): The sampler.
            data (ModelData): The data.

        Raises:
            ValueError: If the optimizer is not initialized.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized.")

        def closure():
            self.optimizer.zero_grad()  # type: ignore
            logpdfs, _ = self._logpdfs_psi_fn(data, sampler.b)
            loss = -logpdfs.mean()
            loss.backward()  # type: ignore
            return loss.item()

        self.optimizer.step(closure)

        # Restore logpdfs and psi
        sampler.logpdfs, sampler.psi = sampler.logpdfs_psi_fn(sampler.b)

    def _is_converged(self) -> bool:
        """Checks if the optimizer has converged.

        This is based on a linear regression of the parameters with the current
        number of iterations. It R2 is below a threshold, the optimizer is
        considered to have converged.

        Args:
            optimizer (torch.optim.Adam): The optimizer.

        Returns:
            bool: True if the optimizer has converged, False otherwise.
        """

        def r2(Y: torch.Tensor) -> torch.Tensor:
            n = Y.size(0)
            i = torch.arange(n, dtype=torch.get_default_dtype())
            i_centered = i - (n - 1) / 2
            y_centered = Y - Y.mean(dim=0)
            num = (i_centered @ y_centered) ** 2
            den = i_centered.pow(2).sum() * y_centered.pow(2).sum(dim=0)
            return (num / den).nan_to_num()

        if len(self.params_history_) < self.window_size:
            return False

        Y = torch.stack(self.params_history_[-self.window_size :])
        return r2(Y).max().item() < self.tol

    def _init_jac(
        self, data: ModelData
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        """Initializes the Jacobian matrix and function.

        Args:
            data (ModelData): The complete model data.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]: The Jacobian
                matrix and the Jacobian function.
        """

        @jacfwd  # type: ignore
        def _dict_jac_fn(
            named_parameters_dict: dict[str, torch.Tensor], b: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            return functional_call(self, named_parameters_dict, args=(data, b))

        def _jac_fn(b: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                list(_dict_jac_fn(dict(self.named_parameters()), b).values()),  # type: ignore
                dim=-1,
            )

        return torch.zeros(len(data), self.params.numel()), cast(
            Callable[[torch.Tensor], torch.Tensor], _jac_fn
        )

    def _update_jac(
        self,
        mjac: torch.Tensor,
        jac_fn: Callable[[torch.Tensor], torch.Tensor],
        sampler: MetropolisHastingsSampler,
    ):
        """Updates the Jacobian matrix.

        Args:
            mjac (torch.Tensor): The mean Jacobian matrix.
            jac_fn (Callable[[torch.Tensor], torch.Tensor]): The Jacobian function.
            sampler (MetropolisHasstingsSampler): The sampler.
        """
        mjac += jac_fn(sampler.b).detach()  # type: ignore

    def _compute_fim(self, mjac: torch.Tensor) -> torch.Tensor:
        """Computes the Fisher Information Matrix.

        Args:
            mjac (torch.Tensor): The mean Jacobian matrix.

        Returns:
            torch.Tensor: The Fisher Information Matrix.
        """
        mjac /= ceil(self.n_samples_summary / self.n_chains)
        return mjac.T @ mjac

    def _init_criteria(self, data: ModelData):
        """Initializes the model selection criteria.

        Args:
            data (ModelData): The complete model data.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The criteria.
        """
        logpdf = torch.tensor(0.0)
        mb = torch.zeros(len(data), self.params.q.dim)
        mb2 = torch.zeros(len(data), self.params.q.dim, self.params.q.dim)
        return logpdf, mb, mb2

    def _update_criteria(
        self,
        logpdf: torch.Tensor,
        mb: torch.Tensor,
        mb2: torch.Tensor,
        sampler: MetropolisHastingsSampler,
    ):
        """Updates the model selection criteria.

        Args:
            logpdf (torch.Tensor): The logpdf.
            mb (torch.Tensor): The mean of b.
            mb2 (torch.Tensor): The mean of squared b.
            sampler (MetropolisHastingsSampler): The sampler.
        """
        logpdf += sampler.logpdfs.sum()
        mb += sampler.b.sum(dim=0)
        mb2 += torch.einsum("ijk,ijl->jkl", sampler.b, sampler.b)

    def _compute_criteria(
        self,
        logpdf: torch.Tensor,
        mb: torch.Tensor,
        mb2: torch.Tensor,
        fim: torch.Tensor,
    ) -> tuple[float, float, float]:
        """Computes the model selection criteria.

        Args:
            logpdf (torch.Tensor): The logpdf.
            mb (torch.Tensor): The mean of b.
            mb2 (torch.Tensor): The mean of squared b.
            fim (torch.Tensor): The Fisher Information Matrix.

        Returns:
            tuple[float, float, float]: The criteria.
        """
        den = self.n_chains * ceil(self.n_samples_summary / self.n_chains)
        logpdf /= den
        mb /= den
        mb2 /= den

        covs = mb2 - torch.einsum("ij,ik->ijk", mb, mb)
        entropy = 0.5 * (torch.logdet(covs) + self.params.q.dim).sum().item()

        loglik = logpdf.item() + entropy
        aic = -2 * loglik + 2 * self.params.numel()
        bic = -2 * loglik + torch.logdet(fim).item()
        return loglik, aic, bic

    @validate_params(
        {
            "data": [ModelData],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, data: ModelData) -> Self:
        r"""Fits the model to the data.

        This method computes an estimate of the Maximum Likelihood Estimate (MLE) noted
        :math:`\hat{\theta}`. It is optimized using the `optimizer` attribute for a
        maximum of `max_iter_fit` iterations.

        It leverages the Fisher identity and stochastic gradient algorithm coupled
        with a MCMC (Metropolis-Hastings) sampler. The Fisher identity states that

        .. math::
            \nabla_\theta \log \mathcal{L}(\theta ; x) = \mathbb{E}_{b \sim p(\cdot
            \mid x, \theta)} \left( \nabla_\theta \log \mathcal{L}(\theta ; x, b)
            \right).

        Many methods exist for estimating the Fisher Information Matrix in latent
        variable models. In particular, this class leverages the expected Fisher
        Information Matrix using the identity

        .. math::
            \mathcal{I}_n(\theta) = \sum_{i=1}^n \mathbb{E}_{b \sim p(\cdot \mid x_i,
            \hat{\theta})} \left(\nabla \log \mathcal{L}(\hat{\theta} ; x_i, b) \nabla
            \log \mathcal{L}(\hat{\theta} ; x_i, b)^T \right).

        Model selection criteria are based on a Laplace approximation of the posterior
        distribution to ensure a closed form entropy.

        `n_samples_summary` is the number of samples used to compute the Fisher
        Information Matrix and model selection criteria. The greater this number,
        the more accurate the estimates are, but the longer it takes to compute them.

        For more information, see ISSN 2824-7795.

        Args:
            data (ModelData): The data to fit the model to.

        Returns:
            Self: The fitted model.
        """
        data = ModelDataUnchecked(
            data.x, data.t, data.y, data.trajectories, data.c
        ).prepare(self)

        # Initialize MCMC
        sampler = self._init_mcmc(data)
        sampler.run(self.n_warmup)

        # Main fitting loop
        for _ in trange(
            self.max_iter_fit, desc="Fitting joint model", disable=not self.verbose
        ):
            self._step(sampler, data)
            self.params_history_.append(
                parameters_to_vector(self.params.parameters()).detach()
            )
            if self._is_converged():
                break
            sampler.run(self.n_subsample)

        if not self._is_converged():
            warn(
                "Model may not have converged in the specified number of iterations.",
                stacklevel=2,
            )

        # Initialize Jacobian matrix and criteria variables
        mjac, jac_fn = self._init_jac(data)
        logpdf, mb, mb2 = self._init_criteria(data)

        # FIM and Criteria loop
        n_iter = ceil(self.n_samples_summary / self.n_chains)
        for _ in trange(
            n_iter,
            desc="Computing FIM and Model Selection Criteria",
            disable=not self.verbose,
        ):
            self._update_jac(mjac, jac_fn, sampler)
            self._update_criteria(logpdf, mb, mb2, sampler)
            for _ in range(self.n_subsample):
                sampler.step()

        self.fim_ = self._compute_fim(mjac)
        self.loglik_, self.aic_, self.bic_ = self._compute_criteria(
            logpdf, mb, mb2, self.fim_
        )

        self._cache.clear()
        return self
