from collections.abc import Callable
from typing import cast

import torch

from ..typedefs._data import CompleteModelData
from ..typedefs._params import ModelParams
from ._sampler import MetropolisHastingsSampler


class FitMixin:
    """Mixin for fitting the model.

    Attributes:
        params_ (ModelParams): The parameters of the model.
        pen (Callable[[ModelParams], torch.Tensor] | None): The penalization function.
        n_iters (tuple[int, int]): The number of iterations for stochastic gradient and
            MCMC.
        lr (float): The learning rate.
        tols (tuple[float, float]): The tolerances for the convergence.
        _logpdfs_aux_fn (Callable[[ModelParams, CompleteModelData, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor]]): The auxiliary function for logpdfs.
    """

    pen: Callable[[ModelParams], torch.Tensor] | None
    params_: ModelParams
    n_iters: tuple[int, int]
    lr: float
    tols: tuple[float, float]
    _logpdfs_aux_fn: Callable[
        [ModelParams, CompleteModelData, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]

    def __init__(self, lr: float, tols: tuple[float, float]):
        """Initializes the fit parameters.

        Args:
            lr (float): The learning rate.
            tols (tuple[float, float]): The tolerances for the convergence.
        """
        self.lr = lr
        self.tols = tols

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
        loss = (
            -logpdfs.mean()
            if self.pen is None
            else -logpdfs.mean() + self.pen(self.params_)
        )
        loss.backward()  # type: ignore
        optimizer.step()  # type: ignore

        # Restore logpdfs and aux
        sampler.logpdfs, sampler.aux = sampler.logpdfs_aux_fn(sampler.b)

    def _is_converged(self, optimizer: torch.optim.Adam) -> bool:
        """Checks if the optimizer has converged.

        Args:
            optimizer (torch.optim.Adam): The optimizer.

        Returns:
            bool: True if the optimizer has converged, False otherwise.
        """
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if not state:
                    continue
                m_t, v_t = state["exp_avg"], state["exp_avg_sq"]
                if not torch.all(m_t.abs() <= self.tols[0] + self.tols[1] * v_t.sqrt()):
                    return False
        return True

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
            jac_fn(self.params_.as_flat_tensor, sampler.b).detach() / self.n_iters[1]
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
        logpdf += sampler.logpdfs.mean() / self.n_iters[1]
        mb += sampler.b.mean(dim=0) / self.n_iters[1]
        mb2 += torch.einsum("ijk,ijl->jkl", sampler.b, sampler.b) / (
            self.n_iters[1] * sampler.n_chains
        )

    def _compute_criteria(
        self,
        logpdf: torch.Tensor,
        mb: torch.Tensor,
        mb2: torch.Tensor,
        fim: torch.Tensor,
        data: CompleteModelData,
    ) -> tuple[float, float, float, float]:
        """Computes the criteria.

        Args:
            logpdf (torch.Tensor): The logpdf.
            mb (torch.Tensor): The mean of b.
            mb2 (torch.Tensor): The mean of squared b.
            fim (torch.Tensor): The Fisher Information Matrix.
            data (CompleteModelData): The complete model data.

        Returns:
            tuple[float, float, float, float]: The criteria.
        """
        covs = mb2 - torch.einsum("ij,ik->ijk", mb, mb)
        entropy = 0.5 * (torch.logdet(covs) + self.params_.Q.dim).sum().item()
        loglik = logpdf.item() + entropy
        nloglik_pen = (
            data.size * self.pen(self.params_).item() - loglik
            if self.pen is not None
            else -loglik
        )
        aic = 2 * nloglik_pen + 2 * self.params_.numel
        bic = 2 * nloglik_pen + torch.logdet(fim).item()
        return loglik, nloglik_pen, aic, bic
