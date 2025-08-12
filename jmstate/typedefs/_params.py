import itertools
from dataclasses import field
from functools import cached_property
from typing import Any, Self

import torch
from pydantic import ConfigDict, dataclasses, validate_call

from ..utils._checks import check_inf, check_matrix_dim
from ..utils._linalg import cov_from_repr
from ._defs import MatRepr, Tensor1D, Tensor2D


@dataclasses.dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelParams:
    r"""Dataclass containing model parameters.

    Note three types of covariance matrices parametrization are provided: scalar
    matrix; diagonal matrix; full matrix. Defaults to the full matrix parametrization.
    This is achieved through a log Cholesky parametrization of the inverse covariance
    matrix. Formally, consider :math:`P = \Sigma^{-1}` the precision matrix and let
    :math:`L` be the Cholesky factor with positive diagonal elements, the log Cholseky
    is given by:

    .. math::
        \tilde{L}_{ij} = L_{ij}, i > j,

    and:

    .. math::
        \tilde{L}_{ii} = \log L_{ii}.

    This is very numerically stable and fast, as it doesn't require inverting the
    matrix when computing quadratic forms. The log determinant is then equal to:

    .. math::

        \log \det P = 2 \operatorname{Tr}(\tilde{L}).

    You can use these methods by creating the appropriate `MatRepr` with methods of
    `ball`, `diag` or `full`.

    Additionnally, if your data has mixed missing values, do not use `full` matrix
    parametrization for the residuals, as is this case the components must be
    independent.

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

        check_matrix_dim(self.Q_repr, "Q")
        check_matrix_dim(self.R_repr, "R")
        for key, val in self.as_groups.items():
            check_inf(tuple((t, key) for t in val))

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
    def numel(self) -> int:
        """Return the number of parameters.

        Returns:
            int: The number of the parameters.
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

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def from_flat_tensor(self, flat: Tensor1D) -> Self:
        """Gets a ModelParams object based on the flat representation.

        This uses the current object as the reference.

        Args:
            flat (torch.Tensor): The flat representation.

        Raises:
            ValueError: If the shape makes the conversion impossible.

        Returns:
            Self: The constructed ModelParams.
        """
        i = 0

        def _next(ref: torch.Tensor):
            nonlocal i
            n = ref.numel()
            result = flat[i : i + n]
            i += n
            return result.view(ref.shape)

        gamma = None if self.gamma is None else _next(self.gamma)

        Q_flat = _next(self.Q_repr.flat)
        R_flat = _next(self.R_repr.flat)

        alphas = {key: _next(val) for key, val in self.alphas.items()}

        betas = (
            None
            if self.betas is None
            else {key: _next(val) for key, val in self.betas.items()}
        )

        return type(self)(
            gamma,
            self.Q_repr._replace(flat=Q_flat),
            self.R_repr._replace(flat=R_flat),
            alphas,
            betas,
            extra=self.extra,
            skip_validation=True,
        )

    def clone(self) -> Self:
        """Returns a detached clone of the parameters.

        Returns:
            ModelParams: The clone.
        """
        return self.from_flat_tensor(self.as_flat_tensor)
