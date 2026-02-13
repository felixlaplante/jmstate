from collections.abc import Callable
from typing import Any, Self, cast
from warnings import warn

import torch
from tqdm import trange

from ..typedefs._data import CompleteModelData, ModelData, ModelDesign
from ..typedefs._params import ModelParams
from ..utils._cache import Cache
from ._sampler import MCMCMixin, MetropolisHastingsSampler


class FitMixin(MCMCMixin):
    """Mixin for fitting the model."""

    model_design: ModelDesign
    params_: ModelParams
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
    _cache: Cache

    def __init__(
        self, lr: float, tol: float, window_size: int, *args: Any, **kwargs: Any
    ):
        """Initializes the fit parameters.

        Args:
            lr (float): The learning rate.
            tol (float): The tolerance for the convergence.
            window_size (int): The window size for the convergence.
        """
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.tol = tol
        self.window_size = window_size

    def _init_optimizer(self) -> torch.optim.Adam:
        """Initializes the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        self.params_.requires_grad_(True)

        return torch.optim.Adam(
            self.params_.as_list + (self.params_.extra or []), lr=self.lr
        )

    def _step(
        self,
        optimizer: torch.optim.Adam,
        sampler: MetropolisHastingsSampler,
        data: CompleteModelData,
    ):
        """Performs a step of the optimizer.

        Args:
            optimizer (torch.optim.Adam): The optimizer.
        """
        optimizer.zero_grad()  # type: ignore
        logpdfs, _ = self._logpdfs_aux_fn(self.params_, data, sampler.b)
        loss = -logpdfs.mean()
        loss.backward()  # type: ignore
        optimizer.step()  # type: ignore

        # Restore logpdfs and aux
        sampler.logpdfs, sampler.psi = sampler.logpdfs_aux_fn(sampler.b)

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
            return num / den

        if len(self.params_history_) < self.window_size:
            return False

        params = torch.stack(
            [p.as_flat_tensor for p in self.params_history_[-self.window_size :]]
        )
        return r2(params).max().item() < self.tol

    def _init_jac(
        self, data: CompleteModelData
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """Initializes the Jacobian matrix.

        Args:
            data (CompleteModelData): The complete model data.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
                The Jacobian matrix and the Jacobian function.
        """

        @torch.func.jacfwd  # type: ignore
        def _jac_fn(params_flat_tensor: torch.Tensor, b: torch.Tensor):
            params = self.params_.from_flat_tensor(params_flat_tensor)
            return self._logpdfs_aux_fn(params, data, b)[0].mean(dim=0)

        return torch.zeros(data.size, self.params_.numel), cast(
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor], _jac_fn
        )

    def _update_jac(
        self,
        mjac: torch.Tensor,
        jac_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        sampler: MetropolisHastingsSampler,
    ):
        """Updates the Jacobian matrix.

        Args:
            mjac (torch.Tensor): The mean Jacobian matrix.
            jac_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The Jacobian
                function.
            sampler (MetropolisHastingsSampler): The sampler.
        """
        mjac += (
            jac_fn(self.params_.as_flat_tensor, sampler.b).detach()
            / self.n_iter_summary
        )

    @staticmethod
    def _compute_fim(mjac: torch.Tensor) -> torch.Tensor:
        """Computes the Fisher Information Matrix.

        Args:
            mjac (torch.Tensor): The mean Jacobian matrix.

        Returns:
            torch.Tensor: The Fisher Information Matrix.
        """
        return mjac.T @ mjac

    def _init_criteria(self, data: CompleteModelData):
        """Initializes the criteria.

        Args:
            data (CompleteModelData): The complete model data.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The criteria.
        """
        logpdf = torch.tensor(0.0)
        mb = torch.zeros(data.size, self.params_.Q.dim)
        mb2 = torch.zeros(data.size, self.params_.Q.dim, self.params_.Q.dim)
        return logpdf, mb, mb2

    def _update_criteria(
        self,
        logpdf: torch.Tensor,
        mb: torch.Tensor,
        mb2: torch.Tensor,
        sampler: MetropolisHastingsSampler,
    ):
        """Updates the criteria.

        Args:
            logpdf (torch.Tensor): The logpdf.
            mb (torch.Tensor): The mean of b.
            mb2 (torch.Tensor): The mean of squared b.
            sampler (MetropolisHastingsSampler): The sampler.
        """
        logpdf += sampler.logpdfs.mean() / self.n_iter_summary
        mb += sampler.b.mean(dim=0) / self.n_iter_summary
        mb2 += torch.einsum("ijk,ijl->jkl", sampler.b, sampler.b) / (
            self.n_iter_summary * sampler.n_chains
        )

    def _compute_criteria(
        self,
        logpdf: torch.Tensor,
        mb: torch.Tensor,
        mb2: torch.Tensor,
        fim: torch.Tensor,
    ) -> tuple[float, float, float]:
        """Computes the criteria.

        Args:
            logpdf (torch.Tensor): The logpdf.
            mb (torch.Tensor): The mean of b.
            mb2 (torch.Tensor): The mean of squared b.
            fim (torch.Tensor): The Fisher Information Matrix.

        Returns:
            tuple[float, float, float]: The criteria.
        """
        covs = mb2 - torch.einsum("ij,ik->ijk", mb, mb)
        entropy = 0.5 * (torch.logdet(covs) + self.params_.Q.dim).sum().item()
        loglik = logpdf.item() + entropy
        aic = -2 * loglik + 2 * self.params_.numel
        bic = -2 * loglik + torch.logdet(fim).item()
        return loglik, aic, bic

    def fit(self, data: ModelData) -> Self:
        r"""Fits the model to the data.

        It leverages the Fisher identity and stochastic gradient algorithm coupled
        with a MCMC (Metropolis-Hastings) sampler:

        .. math::
            \nabla_\theta \log \mathcal{L}(\theta ; x) = \mathbb{E}_{b \sim p(\cdot
            \mid x, \theta)} \left( \nabla_\theta \log \mathcal{L}(\theta ; x, b)
            \right).

        Args:
            data (ModelData): The data to fit the model to.

        Returns:
            Self: The fitted model.
        """
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
            self.n_iter_fit, desc="Fitting joint model", disable=not self.verbose
        ):
            self._step(optimizer, sampler, data)
            self.params_history_.append(self.params_.clone().detach())
            if self._is_converged():
                break
            sampler.run(self.n_subsample)

        # Warn if not converged
        if not self._is_converged():
            warn(
                "Model may not have converged in the specified number of iterations.",
                stacklevel=2,
            )

        # Initialize Jacobian matrix and criteria
        mjac, jac_fn = self._init_jac(data)
        logpdf, mb, mb2 = self._init_criteria(data)

        # FIM and Criteria loop
        for _ in trange(
            self.n_iter_summary,
            desc="Computing FIM and Criteria",
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

        # Restore to default state
        self.params_.requires_grad_(False)
        self._cache.clear()
        return self
