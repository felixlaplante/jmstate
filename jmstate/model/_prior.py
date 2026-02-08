from typing import Any

import torch

from ..typedefs._defs import LOG_TWO_PI
from ..typedefs._params import ModelParams
from ..utils._linalg import get_cholesky_and_log_eigvals


class PriorMixin:
    """Mixin class for prior model computations."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the prior mixin."""
        super().__init__(*args, **kwargs)

    @staticmethod
    def _prior_logliks(params: ModelParams, b: torch.Tensor) -> torch.Tensor:
        """Computes the prior log likelihoods.

        Args:
            params (ModelParams): The model parameters.
            b (torch.Tensor): The 3D tensor of random effects.

        Returns:
            torch.Tensor: The computed log likelihoods.
        """
        Q_inv_cholesky, Q_nlog_eigvals = get_cholesky_and_log_eigvals(params, "Q")
        Q_quad_form = (b @ Q_inv_cholesky).pow(2).sum(dim=-1)
        Q_norm_factor = (Q_nlog_eigvals - LOG_TWO_PI).sum()

        return 0.5 * (Q_norm_factor - Q_quad_form)
