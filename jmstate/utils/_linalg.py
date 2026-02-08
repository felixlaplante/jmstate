from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import MatRepr, Tensor2D

if TYPE_CHECKING:
    from ..typedefs._params import ModelParams


def _tril_from_flat(flat: torch.Tensor, dim: int) -> torch.Tensor:
    """Generate the lower triangular matrix associated with flat tensor.

    Args:
        flat (torch.Tensor): Flat tensor
        dim (int): Dimension of the matrix.

    Returns:
        torch.Tensor: The lower triangular matrix.
    """
    return torch.zeros(dim, dim, dtype=flat.dtype).index_put_(
        tuple(torch.tril_indices(dim, dim)), flat
    )


def _flat_from_tril(L: torch.Tensor) -> torch.Tensor:
    """Flatten the lower triangular part (including the diagonal) of a square matrix.

    Into a 1D tensor, in row-wise order.

    Args:
        L (torch.Tensor): Square lower-triangular matrix of shape (dim, dim).

    Raises:
        RuntimeError: If the flattening fails.

    Returns:
        torch.Tensor: Flattened 1D tensor containing the lower triangular entries.
    """
    dim = L.size(0)
    return L[tuple(torch.tril_indices(dim, dim))]


def _log_cholesky_from_flat(
    flat: torch.Tensor, dim: int, method: str = "full"
) -> torch.Tensor:
    """Computes log cholesky from flat tensor according to choice of method.

    Args:
        flat (torch.Tensor): The flat tensor parameter.
        dim (int): The dimension of the matrix.
        method (str, optional): The method, full, diag or ball. Defaults to "full".

    Raises:
        ValueError: If the method is not in ("full", "diag", "ball").

    Returns:
        torch.Tensor: The log cholesky representation.
    """
    match method:
        case "full":
            return _tril_from_flat(flat, dim)
        case "diag":
            return torch.diag(flat)
        case "ball":
            return flat * torch.eye(dim)
        case _:
            raise ValueError(
                f"Method must be be either 'full', 'diag' or 'ball', got {method}"
            )


def _flat_from_log_cholesky(L: torch.Tensor, method: str = "full") -> torch.Tensor:
    """Computes flat tensor from log cholesky matrix according to choice of method.

    Args:
        L (torch.Tensor): The square lower triangular matrix parameter.
        method (str, optional): The method, full, diag or ball. Defaults to "full".

    Raises:
        ValueError: If the method is not in ("full", "diag", "ball").

    Returns:
        torch.Tensor: The flat representation.
    """
    match method:
        case "full":
            return _flat_from_tril(L)
        case "diag":
            return L.diag()
        case "ball":
            return L[0, 0].reshape(1)
        case _:
            raise ValueError(
                f"Method must be be either 'full', 'diag' or 'ball', got {method}"
            )


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def cov_from_repr(mat_repr: MatRepr) -> Tensor2D:
    r"""Computes covariance matrix from representation.

    Note three types of covariance matrices parametrization are provided: scalar
    matrix; diagonal matrix; full matrix. Defaults to the full matrix parametrization.
    This is achieved through a log Cholesky parametrization of the inverse covariance
    matrix. Formally, consider :math:`P = \Sigma^{-1}` the precision matrix and let
    :math:`L` be the Cholesky factor with positive diagonal elements, the log Cholseky
    is given by:

    .. math::
        \tilde{L}_{ij} = L_{ij}, \, i > j,

    and:

    .. math::
        \tilde{L}_{ii} = \log L_{ii}.

    This is very numerically stable and fast, as it doesn't require inverting the
    matrix when computing quadratic forms. The log determinant is then equal to:

    .. math::

        \log \det P = 2 \operatorname{Tr}(\tilde{L}).

    You can use these methods by creating the appropriate `MatRepr` with methods of
    `ball`, `diag` or `full`.

    Args:
        mat_repr (MatRepr): The matrix representation.

    Raises:
        ValueError: If method 'full' and number of elements are inconsistent with dim.
        ValueError: If method 'diag' and number of elements are inconsistent with dim.
        ValueError: If method 'ball' and number of elements is not one.

    Returns:
        torch.Tensor: The covariance matrix.
    """
    flat, dim, method = mat_repr

    if method == "full" and flat.numel() != (dim * (dim + 1)) // 2:
        raise ValueError(
            f"Inconsistent dim:{dim} with method 'full', flat with {flat.numel()} "
            "elements"
        )
    if method == "diag" and flat.numel() != dim:
        raise ValueError(
            f"Inconsistent dim:{dim} with method 'diag', flat with {flat.numel()} "
            "elements"
        )
    if method == "ball" and flat.numel() != 1:
        raise ValueError("Inconsistent with method 'ball', flat must have one element")

    L = _log_cholesky_from_flat(flat, dim, method)
    L.diagonal().exp_()

    return torch.cholesky_inverse(L)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def repr_from_cov(V: Tensor2D, method: str = "full") -> MatRepr:
    r"""Computes representation from covariance matrix according to choice of method.

    Note three types of covariance matrices parametrization are provided: scalar
    matrix; diagonal matrix; full matrix. Defaults to the full matrix parametrization.
    This is achieved through a log Cholesky parametrization of the inverse covariance
    matrix. Formally, consider :math:`P = \Sigma^{-1}` the precision matrix and let
    :math:`L` be the Cholesky factor with positive diagonal elements, the log Cholseky
    is given by:

    .. math::
        \tilde{L}_{ij} = L_{ij}, \, i > j,

    and:

    .. math::
        \tilde{L}_{ii} = \log L_{ii}.

    This is very numerically stable and fast, as it doesn't require inverting the
    matrix when computing quadratic forms. The log determinant is then equal to:

    .. math::

        \log \det P = 2 \operatorname{Tr}(\tilde{L}).

    You can use these methods by creating the appropriate `MatRepr` with methods of
    `ball`, `diag` or `full`.

    Args:
        V (Tensor2D): The square covariance matrix parameter.
        method (str, optional): The method, full, diag or ball. Defaults to "full".

    Returns:
        MatRepr: The flat representation.
    """
    L = cast(torch.Tensor, torch.linalg.cholesky(V.inverse()))  # type: ignore
    L.diagonal().log_()
    return MatRepr(_flat_from_log_cholesky(L, method), L.size(0), method)


def get_cholesky_and_log_eigvals(
    params: ModelParams, matrix: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gets Cholesky factor as well as log eigvals.

    Args:
        params (ModelParams): The model parameters.
        matrix (str): Either "Q" or "R".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Precision matrix and log eigvals.
    """
    # Get flat then log cholesky
    flat, dim, method = getattr(params, matrix + "_repr")

    L = _log_cholesky_from_flat(flat, dim, method)
    log_eigvals = 2 * L.diag()
    L.diagonal().exp_()

    return L, log_eigvals
