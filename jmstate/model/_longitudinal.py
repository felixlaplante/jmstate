from typing import Any

from ..typedefs._data import CompleteModelData, ModelDesign
from ..typedefs._defs import LOGTWOPI, Tensor2D, Tensor3D
from ..typedefs._params import ModelParams
from ..utils._linalg import get_cholesky_and_log_eigvals


class LongitudinalMixin:
    """Mixin class for longitudinal model computations."""

    model_design: ModelDesign

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the class.

        Args:
            args (Any): Additional args.
            kwargs (Any): Additional kwargs.
        """
        super().__init__(*args, **kwargs)

    def _long_logliks(
        self, params: ModelParams, psi: Tensor3D, data: CompleteModelData
    ) -> Tensor2D:
        """Computes the longitudinal log likelihood.

        Args:
            params (ModelParams): The model parameters.
            psi (Tensor3D): A 3D tensor of individual parameters.
            data (ModelData): Dataset on which likelihood is computed.

        Returns:
            Tensor2D: The computed log likelihood.
        """
        predicted = self.model_design.regression_fn(data.valid_t, psi)
        diffs = data.valid_y - predicted * data.valid_mask

        R_inv_cholesky, R_nlog_eigvals = get_cholesky_and_log_eigvals(params, "R")
        R_quad_forms = (diffs @ R_inv_cholesky).pow(2).sum(dim=(-2, -1))
        R_norm_factor = data.n_valid @ (R_nlog_eigvals - LOGTWOPI)

        return 0.5 * (R_norm_factor - R_quad_forms)
