import torch

from ..typedefs._defs import LOGTWOPI
from ..typedefs._structures import ModelData, ModelDesign, ModelParams
from ..utils._linalg import get_cholesky_and_log_eigvals


class LongitudinalMixin:
    """Mixin class for longitudinal model computations."""

    params_: ModelParams
    model_design: ModelDesign

    def _long_logliks(self, psi: torch.Tensor, data: ModelData) -> torch.Tensor:
        """Computes the longitudinal log likelihood.

        Args:
            psi (torch.Tensor): A matrix of individual parameters.
            data (ModelData): Dataset on which likelihood is computed.

        Returns:
            torch.Tensor: The computed log likelihood.
        """
        predicted = self.model_design.regression_fn(t=data.extra_["valid_t"], psi=psi)
        diffs = data.extra_["valid_y"] - predicted * data.extra_["valid_mask"]

        R_inv_cholesky, R_log_eigvals = get_cholesky_and_log_eigvals(self.params_, "R")
        R_quad_forms = (diffs @ R_inv_cholesky).pow(2).sum(dim=(1, 2))
        R_log_dets = data.extra_["n_valid"] @ (R_log_eigvals - LOGTWOPI)

        return 0.5 * (R_log_dets - R_quad_forms)
