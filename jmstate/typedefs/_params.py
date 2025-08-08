import itertools
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import torch
from beartype import beartype

from ..utils._checks import check_inf, check_matrix_dim
from ..utils._linalg import cov_from_repr
from ._defs import IntPositive, MatRepr, Tensor1D, Tensor2D


@beartype
@dataclass
class ModelParams:
    """Dataclass containing model parameters.

    Raises:
        ValueError: If the number of elements is not triangular with method "full".
        ValueError: If the number of elements is not one and the method is "ball".
        ValueError: If the method is unknown.
    """

    gamma: torch.Tensor | None
    Q_repr: MatRepr
    R_repr: MatRepr
    alphas: dict[tuple[Any, Any], Tensor1D]
    betas: dict[tuple[Any, Any], Tensor1D] | None
    extra: list[torch.Tensor] | None = field(default=None, repr=False)
    skip_validation: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Validate and put to float32 all tensors."""
        if self.skip_validation:
            return

        for t in self.as_list:
            t.data = t.to(torch.get_default_dtype())
        if self.extra is not None:
            for t in self.extra:
                t.data = t.to(torch.get_default_dtype())

        check_matrix_dim(self.Q_repr)
        check_matrix_dim(self.R_repr)
        for val in self.as_groups.values():
            check_inf(tuple(val))

    @property
    def as_groups(self) -> dict[str, list[torch.Tensor]]:
        """Get a grouped dict of all the parameters.

        Returns:
            dict[str, list[torch.Tensor]]: The dict of the parameters.
        """
        groups = {
            "gamma": None if self.gamma is None else [self.gamma],
            "Q": [self.Q_repr.flat],
            "R": [self.R_repr.flat],
            "alphas": list(self.alphas.values()),
            "betas": None if self.betas is None else list(self.betas.values()),
        }
        return {key: val for key, val in groups.items() if val is not None}

    @property
    def as_list(self) -> list[torch.Tensor]:
        """Get a list of all the parameters.

        Returns:
            list[torch.Tensor]: The list of the parameters.
        """
        return list(itertools.chain.from_iterable(self.as_groups.values()))

    @property
    def as_flat_tensor(self) -> Tensor1D:
        """Get the flattened parameters.

        Returns:
            torch.Tensor: The flattened parameters.
        """
        return torch.cat([p.detach().view(-1) for p in self.as_list])

    @cached_property
    def numel(self) -> IntPositive:
        """Return the number of parameters.

        Returns:
            IntPositive: The number of the parameters.
        """
        return sum(p.numel() for p in self.as_list)

    def requires_grad_(self, req: bool):
        """Enable or disable gradient computation on non extra parameters.

        Args:
            req (bool): Wether to require or not.
        """
        for t in self.as_list:
            t.requires_grad_(req)

    def extra_requires_grad_(self, req: bool):
        """Enable or disable gradient computation on extra parameters.

        Args:
            req (bool): Wether to require or not.
        """
        if self.extra is None:
            return
        for t in self.extra:
            t.requires_grad_(req)

    def get_cov(self, matrix: str) -> Tensor2D:
        """Get covariance from parameter.

        Args:
            matrix (str): Either "Q" or "R".

        Raises:
            ValueError: If the matrix is not in ("Q", "R")

        Returns:
            Tensor2D: The covariance matrix.
        """
        if matrix not in ("Q", "R"):
            raise ValueError(f"matrix must be either Q or R, got {matrix}")

        # Get flat then covariance
        return cov_from_repr(getattr(self, matrix + "_repr"))


@beartype
def params_like_from_flat(ref_params: ModelParams, flat: Tensor1D) -> ModelParams:
    """Gets a ModelParams instance based on the flat representation.

    Args:
        ref_params (ModelParams): The reference params.
        flat (torch.Tensor): The flat representation.

    Raises:
        ValueError: If the shape makes the conversion impossible.

    Returns:
        ModelParams: The constructed ModelParams.
    """
    i = 0

    def _next(ref: torch.Tensor):
        nonlocal i
        n = ref.numel()
        result = flat[i : i + n]
        i += n
        return result.view(ref.shape)

    gamma = None if ref_params.gamma is None else _next(ref_params.gamma)

    Q_flat = _next(ref_params.Q_repr.flat)
    R_flat = _next(ref_params.R_repr.flat)

    alphas = {key: _next(val) for key, val in ref_params.alphas.items()}

    betas = (
        None
        if ref_params.betas is None
        else {key: _next(val) for key, val in ref_params.betas.items()}
    )

    return ModelParams(
        gamma,
        ref_params.Q_repr._replace(flat=Q_flat),
        ref_params.R_repr._replace(flat=R_flat),
        alphas,
        betas,
        extra=ref_params.extra,
        skip_validation=True,
    )
