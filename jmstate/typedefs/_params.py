import itertools
from dataclasses import dataclass, field
from math import isqrt

import torch

from ..utils._linalg import cov_from_flat


@dataclass
class ModelParams:
    """Dataclass containing model parameters.

    Raises:
        ValueError: If the name matrix is not "Q" nor "R".
        ValueError: If the number of elements is not triangular and method is "full".
        ValueError: If the number of elements is not one and the method is "ball".
        ValueError: If the name matrix is not "Q" nor "R".
        ValueError: If the name matrix is not "Q" nor "R".
        ValueError: If any of the main tensors contains inf.
        ValueError: If any of the alpha tensors contains inf.
        ValueError: If any of the beta tensors contains inf.
    """

    gamma: torch.Tensor | None
    Q_repr: tuple[torch.Tensor, str]
    R_repr: tuple[torch.Tensor, str]
    alphas: dict[tuple[int, int], torch.Tensor]
    betas: dict[tuple[int, int], torch.Tensor] | None
    Q_dim_: int = field(init=False, repr=False)
    R_dim_: int = field(init=False, repr=False)
    skip_validation: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Validate all tensors are 1D and don't contain inf.

        Raises:
            ValueError: If any of the main tensors contains inf.
            ValueError: If any of the alpha tensors contains inf.
            ValueError: If any of the beta tensors contains inf.
        """
        Q_flat, Q_method = self.Q_repr
        R_flat, R_method = self.R_repr

        Q_flat = torch.as_tensor(Q_flat, dtype=torch.float32).view(-1)
        R_flat = torch.as_tensor(R_flat, dtype=torch.float32).view(-1)

        # Update representation tuples
        self.Q_repr = (Q_flat, Q_method)
        self.R_repr = (R_flat, R_method)

        self.gamma = (
            torch.as_tensor(self.gamma, dtype=torch.float32)
            if self.gamma is not None
            else None
        )

        for key, alpha in self.alphas.items():
            self.alphas[key] = torch.as_tensor(alpha, dtype=torch.float32).view(-1)

        if self.betas is not None:
            for key, beta in self.betas.items():
                self.betas[key] = torch.as_tensor(beta, dtype=torch.float32).view(-1)

        self._set_dims("Q")
        self._set_dims("R")

        if self.skip_validation:
            return

        for name, tensor in [
            ("gamma", self.gamma),
            ("Q_flat_", self.Q_repr[0]),
            ("R_flat_", self.R_repr[0]),
        ]:
            if tensor is None:
                continue
            if tensor.isinf().any():
                raise ValueError(f"{name} contains inf")

        # Check dictionary tensors
        for key, alpha in self.alphas.items():
            if alpha.isinf().any():
                raise ValueError(f"alpha {key} contains inf")

        if self.betas is not None:
            for key, beta in self.betas.items():
                if beta.isinf().any():
                    raise ValueError(f"beta {key} contains inf")

    def _set_dims(self, matrix: str) -> None:
        """Sets dimensions for matrix.

        Args:
            matrix (str): Either "Q" or "R".

        Raises:
            ValueError: If the name matrix is not "Q" nor "R".
            ValueError: If the number of elements is not triangular with method "full".
            ValueError: If the number of elements is not one and the method is "ball".
        """
        if matrix not in ("Q", "R"):
            raise ValueError(f"matrix should be either Q or R, got {matrix}")

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
        """Get a dict of all the parameters.

        Returns:
            dict[str, list[torch.Tensor]]: The dict of the parameters.
        """
        groups = {
            "gamma": [self.gamma] if self.gamma is not None else [],
            "Q": [self.Q_repr[0]],
            "R": [self.R_repr[0]],
            "alphas": list(self.alphas.values()),
            "betas": list(self.betas.values()) if self.betas is not None else [],
        }
        return {key: val for key, val in groups.items() if val != []}

    @property
    def as_list(self) -> list[torch.Tensor]:
        """Get a list of all the parameters.

        Returns:
            list[torch.Tensor]: The list of the parameters.
        """
        return list(itertools.chain.from_iterable(self.as_groups.values()))

    @property
    def as_flat_tensor(self) -> torch.Tensor:
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

    def require_grad(self, req: bool):
        """Enable gradient computation on all parameters.

        Args:
            req (bool): Wether to require or not.
        """
        # Enable or diasable gradients
        for tensor in self.as_list:
            tensor.requires_grad_(req)

    def get_cov(self, matrix: str) -> torch.Tensor:
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
