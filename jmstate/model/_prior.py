from typing import Any

import torch

from ..types._parameters import ModelParameters


class PriorMixin:
    """Mixin class for prior model computations."""

    params: ModelParameters

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the prior mixin."""
        super().__init__(*args, **kwargs)

    def _prior_logliks(self, b: torch.Tensor) -> torch.Tensor:
        """Computes the prior log likelihoods.

        Args:
            b (torch.Tensor): The 3D tensor of random effects.

        Returns:
            torch.Tensor: The computed log likelihoods.
        """
        Q_inv_cholesky, Q_nlog_eigvals = self.params.q._inv_cholesky_and_log_eigvals  # type: ignore
        Q_quad_form = (b @ Q_inv_cholesky).pow(2).sum(dim=-1)
        Q_norm_factor = Q_nlog_eigvals.sum()

        return 0.5 * (Q_norm_factor - Q_quad_form)
