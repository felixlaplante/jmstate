import itertools
from dataclasses import dataclass, field
from math import isqrt

import torch
from beartype import beartype

from ..utils._checks import check_inf
from ..utils._linalg import cov_from_flat
from ._defs import Tensor1D, Tensor2D


@beartype
@dataclass
class ModelParams:
    """Dataclass containing model parameters.

    Raises:
        ValueError: If the name matrix is not "Q" nor "R".
        ValueError: If the number of elements is not triangular and method is "full".
        ValueError: If the number of elements is not one and the method is "ball".
        ValueError: If the name matrix is not "Q" nor "R".
        ValueError: If the name matrix is not "Q" nor "R".
        ValueError: If any of the tensors contains inf.
    """

    gamma: torch.Tensor | None
    Q_repr: tuple[Tensor1D, str]
    R_repr: tuple[Tensor1D, str]
    alphas: dict[tuple[int, int], Tensor1D]
    betas: dict[tuple[int, int], Tensor1D] | None
    Q_dim_: int = field(init=False, repr=False)
    R_dim_: int = field(init=False, repr=False)
    skip_validation: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Validate and put to float32 all tensors."""
        for val in self.as_groups.values():
            for i, t in enumerate(val):
                val[i] = t.to(torch.float32)

        self._set_dims("Q")
        self._set_dims("R")

        if self.skip_validation:
            return

        for val in self.as_groups.values():
            check_inf(tuple(val))

    def _set_dims(self, matrix: str):
        """Sets dimensions for matrix.

        Args:
            matrix (str): Either "Q" or "R".

        Raises:
            ValueError: If the name matrix is not "Q" nor "R".
            ValueError: If the number of elements is not triangular with method "full".
            ValueError: If the number of elements is not one and the method is "ball".
        """
        if matrix not in ("Q", "R"):
            raise ValueError(f"matrix must be either Q or R, got {matrix}")

        flat, method = getattr(self, matrix + "_repr")

        match method:
            case "full":
                n = (isqrt(1 + 8 * flat.numel()) - 1) // 2
                if (n * (n + 1)) // 2 != flat.numel():
                    raise ValueError(
                        f"{flat.numel()} is not a triangular number for matrix {matrix}"
                    )
                setattr(self, matrix + "_dim_", n)
            case "diag":
                n = flat.numel()
                setattr(self, matrix + "_dim_", n)
            case "ball":
                if flat.numel() != 1:
                    f"Excepected 1 element for flat, got {flat.numel()}"
                setattr(self, matrix + "_dim_", 1)
            case _:
                raise ValueError(f"Got method {method} unknown for matrix {matrix}")

    @property
    def as_groups(self) -> dict[str, list[torch.Tensor]]:
        """Get a grouped dict of all the parameters.

        Returns:
            dict[str, list[torch.Tensor]]: The dict of the parameters.
        """
        groups = {
            "gamma": None if self.gamma is None else [self.gamma],
            "Q": [self.Q_repr[0]],
            "R": [self.R_repr[0]],
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
        return torch.cat([p.view(-1) for p in self.as_list])

    @property
    def numel(self) -> int:
        """Return the number of parameters.

        Returns:
            int: The number of the parameters.
        """
        return sum(p.numel() for p in self.as_list)

    def requires_grad_(self, req: bool):
        """Enable gradient computation on all parameters.

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
            torch.Tensor: The covariance matrix.
        """
        if matrix not in ("Q", "R"):
            raise ValueError(f"matrix must be either Q or R, got {matrix}")

        # Get flat then covariance
        flat, method = getattr(self, matrix + "_repr")
        n = getattr(self, matrix + "_dim_")

        return cov_from_flat(flat, n, method)
