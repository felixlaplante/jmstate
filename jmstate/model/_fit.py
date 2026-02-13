from collections.abc import Callable

import torch

from ..typedefs._data import CompleteModelData
from ..typedefs._params import ModelParams
from ._sampler import MetropolisHastingsSampler


class FitMixin:
    """Mixin for fitting the model.

    Attributes:
        params_ (ModelParams): The parameters of the model.
        pen (Callable[[ModelParams], torch.Tensor] | None): The penalization function.
        lr (float): The learning rate.
        tols (tuple[float, float]): The tolerances for the convergence.
    """

    pen: Callable[[ModelParams], torch.Tensor] | None
    params_: ModelParams
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
