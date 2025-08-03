from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from beartype import beartype

from ..typedefs._defs import Tensor1D, Tensor2D

if TYPE_CHECKING:
    from ..typedefs._params import ModelParams


def _tril_from_flat(flat: Tensor1D, n: int) -> Tensor2D:
    """Generate the lower triangular matrix associated with flat tensor.

    Args:
        flat (Tensor1D): Flat tehsnro
        n (int): Dimension of the matrix.

    Returns:
        Tensor2D: The lower triangular matrix.
    """
    return torch.zeros(n, n, dtype=flat.dtype).index_put_(
        tuple(torch.tril_indices(n, n)), flat
    )


def _flat_from_tril(L: Tensor2D) -> Tensor1D:
    """Flatten the lower triangular part (including the diagonal) of a square matrix.

    Into a 1D tensor, in row-wise order.

    Args:
        L (Tensor2D): Square lower-triangular matrix of shape (n, n).

    Raises:
        RuntimeError: If the flattening fails.

    Returns:
        Tensor1D: Flattened 1D tensor containing the lower triangular entries.
    """
    n = L.shape[0]
    i, j = torch.tril_indices(n, n)

    return L[i, j]


def _log_cholesky_from_flat(flat: Tensor1D, n: int, method: str = "full") -> Tensor2D:
    """Computes log cholesky from flat tensor according to choice of method.

    Args:
        flat (Tensor1D): The flat tensor parameter.
        n (int): The dimension of the matrix.
        method (str, optional): The method, full, diagonal or ball. Defaults to "full".

    Raises:
        ValueError: If the method is not in ("full", "diag", "ball").

    Returns:
        Tensor2D: The log cholesky representation.
    """
    match method:
        case "full":
            return _tril_from_flat(flat, n)
        case "diag":
            return torch.diag(flat)
        case "ball":
            if flat.numel() != 1:
                f"flat has {flat.numel()} elements, expected 1"
            return flat * torch.eye(n)
        case _:
            raise ValueError(f"Got method {method} unknown")


def _flat_from_log_cholesky(L: Tensor2D, method: str = "full") -> Tensor1D:
    """Computes flat tensor from log cholesky matrix according to choice of method.

    Args:
        L (Tensor2D): The square lower triangular matrix parameter.
        method (str, optional): The method, full, diagonal or ball. Defaults to "full".

    Raises:
        ValueError: If the method is not in ("full", "diag", "ball").

    Returns:
        Tensor1D: The flat representation.
    """
    match method:
        case "full":
            return _flat_from_tril(L)
        case "diag":
            return L.diagonal()
        case "ball":
            return L[0, 0].view(1)
        case _:
            raise ValueError(f"Got method {method} unknown")


@beartype
def cov_from_flat(flat: Tensor1D, n: int, method: str = "full") -> Tensor2D:
    """Computes covariance matrix from flat representation according to choice of method.

    Args:
        flat (Tensor1D): The flat tensor parameter.
        n (int): The dimension of the matrix.
        method (str, optional): The method, full, diagonal or ball. Defaults to "full".

    Raises:
        ValueError: If method 'full' and number of elements are inconsistent with n.
        ValueError: If method 'diag' and number of elements are inconsistent with n.
        ValueError: If method 'ball' and number of elements is not one.

    Returns:
        Tensor2D: The flat representation.
    """
    if method == "full" and flat.numel() != (n * (n + 1)) // 2:
        raise ValueError(
            f"Inconsistent n:{n} with method 'full', flat with {flat.numel()} elements"
        )
    if method == "diag" and flat.numel() != n:
        raise ValueError(
            f"Inconsistent n:{n} with method 'diag', flat with {flat.numel()} elements"
        )
    if method == "ball" and flat.numel() != 1:
        raise ValueError("Inconsistent with method 'ball', flat must have one element")

    L = _log_cholesky_from_flat(flat, n, method)
    L.diagonal().exp_()

    L_inv = cast(
        Tensor2D,
        torch.linalg.solve_triangular(  # type: ignore
            L,
            torch.eye(n, dtype=L.dtype),
            upper=False,
        ),
    )

    return L_inv.T @ L_inv


@beartype
def flat_from_cov(V: Tensor2D, method: str = "full") -> Tensor1D:
    """Computes flat tensor from covariance matrix according to choice of method.

    Args:
        V (Tensor2D): The square covariance matrix parameter.
        method (str, optional): The method, full, diagonal or ball. Defaults to "full".

    Returns:
        Tensor1D: The flat representation.
    """
    L = cast(Tensor2D, torch.linalg.cholesky(V.inverse()))  # type: ignore
    L.diagonal().log_()
    return _flat_from_log_cholesky(L, method)


def get_cholesky_and_log_eigvals(
    params: ModelParams, matrix: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get Cholesky factor as well as log eigenvalues.

    Args:
        params (ModelParams): The model parameters.
        matrix (str): Either "Q" or "R".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Precision matrix and log eigenvalues.
    """
    # Get flat then log cholesky
    flat, method = getattr(params, matrix + "_repr")
    n = getattr(params, matrix + "_dim_")

    L = _log_cholesky_from_flat(flat, n, method)
    eigvals = 2 * L.diagonal()
    L.diagonal().exp_()

    return L, eigvals
