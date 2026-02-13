from typing import Any

import torch

from ..typedefs._data import CompleteModelData, ModelDesign
from ..typedefs._defs import LOG_TWO_PI
from ..typedefs._params import ModelParams


class LongitudinalMixin:
    """Mixin class for longitudinal model computations.

    Attributes:
        model_design (ModelDesign): The model design.
    """

    model_design: ModelDesign

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the longitudinal mixin."""
        super().__init__(*args, **kwargs)

    def _long_logliks(
        self, params: ModelParams, data: CompleteModelData, psi: torch.Tensor
    ) -> torch.Tensor:
        """Computes the longitudinal log likelihoods.

        Args:
            params (ModelParams): The model parameters.
            data (CompleteModelData): Dataset on which likelihood is computed.
            psi (torch.Tensor): A 3D tensor of individual parameters.

        Returns:
            torch.Tensor: The computed log likelihoods.
        """
        # Careful with NaNs
        predicted = self.model_design.regression_fn(data.valid_t, psi)
        diffs = data.valid_y.addcmul(predicted, data.valid_mask, value=-1.0)

        R_inv_cholesky, R_nlog_eigvals = params.R._inv_cholesky_and_log_eigvals  # type: ignore
        R_quad_forms = (diffs @ R_inv_cholesky).pow(2).sum(dim=(-2, -1))
        R_norm_factor = data.n_valid @ (R_nlog_eigvals - LOG_TWO_PI)

        return 0.5 * (R_norm_factor - R_quad_forms)
