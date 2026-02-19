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
        random_prec_cholesky, random_nlog_eigvals = (
            self.params.random_cov._prec_cholesky_and_log_eigvals  # type: ignore
        )
        random_quad_form = (b @ random_prec_cholesky).pow(2).sum(dim=-1)
        random_norm_factor = random_nlog_eigvals.sum()

        return 0.5 * (random_norm_factor - random_quad_form)
