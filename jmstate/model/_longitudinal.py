import torch

from ..utils.structures import ModelData, ModelDesign, ModelParams


class LongitudinalMixin:
    """Mixin class for longitudinal model computations."""

    params_: ModelParams
    model_design: ModelDesign

    def _long_loglik(self, psi: torch.Tensor, data: ModelData) -> torch.Tensor:
        """Computes the longitudinal log likelihood.

        Args:
            psi (torch.Tensor): A matrix of individual parameters.
            data (ModelData): Dataset on which likelihood is computed.

        Returns:
            torch.Tensor: The computed log likelihood.
        """
        # Compute residuals
        predicted = self.model_design.h(data.extra_["valid_t"], data.x, psi)
        diffs = data.extra_["valid_y"] - predicted * data.extra_["valid_mask"]

        R_inv_cholesky, R_log_eigvals = self.params_.get_cholesky_and_log_eigvals("R")
        R_quad_forms = (diffs @ R_inv_cholesky).pow(2).sum(dim=(1, 2))
        R_log_dets = torch.einsum("ij,j->i", data.extra_["n_valid"], R_log_eigvals)

        # Log likelihood
        ll = 0.5 * (R_log_dets - R_quad_forms)

        return ll
