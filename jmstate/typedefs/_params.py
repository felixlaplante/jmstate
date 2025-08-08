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
    skip_validation: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Validate and put to float32 all tensors."""
        if self.skip_validation:
            return

        for tensor in self.as_list:
            tensor.data = tensor.to(torch.get_default_dtype())

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
        """Enable or disable gradient computation on all parameters.

        Args:
            req (bool): Wether to require or not.
        """
        for tensor in self.as_list:
            tensor.requires_grad_(req)

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
