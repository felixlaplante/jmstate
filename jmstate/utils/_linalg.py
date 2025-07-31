from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from ..typedefs._params import ModelParams


def _tril_from_flat(flat: torch.Tensor, n: int) -> torch.Tensor:
    """Generate the lower triangular matrix associated with flat tensor.

    Args:
        flat (torch.Tensor): Flat tehsnro
        n (int): Dimension of the matrix.

    Returns:
        torch.Tensor: The lower triangular matrix.
    """
    return torch.zeros(n, n, dtype=flat.dtype).index_put_(
        tuple(torch.tril_indices(n, n)), flat
    )


def _flat_from_tril(L: torch.Tensor) -> torch.Tensor:
    """Flatten the lower triangular part (including the diagonal) of a square matrix.

    Into a 1D tensor, in row-wise order.

    Args:
        L (torch.Tensor): Square lower-triangular matrix of shape (n, n).

    Raises:
        RuntimeError: If the flattening fails.

    Returns:
        torch.Tensor: Flattened 1D tensor containing the lower triangular entries.
    """
    n = L.shape[0]
    i, j = torch.tril_indices(n, n)

    return L[i, j]


def _log_cholesky_from_flat(
    flat: torch.Tensor, n: int, method: str = "full"
) -> torch.Tensor:
    """Computes log cholesky from flat tensor according to choice of method.

    Args:
        flat (torch.Tensor): The flat tensor parameter.
        n (int): The dimension of the matrix.
        method (str, optional): The method, full, diagonal or ball. Defaults to "full".

    Raises:
        ValueError: If the method is not in ("full", "diag", "ball").

    Returns:
        torch.Tensor: The log cholesky representation.
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


def _flat_from_log_cholesky(L: torch.Tensor, method: str = "full") -> torch.Tensor:
    """Computes flat tensor from log cholesky matrix according to choice of method.

    Args:
        L (torch.Tensor): The square lower triangular matrix parameter.
        method (str, optional): The method, full, diagonal or ball. Defaults to "full".

    Raises:
        ValueError: If the method is not in ("full", "diag", "ball").

    Returns:
        torch.Tensor: The flat representation.
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


def cov_from_flat(flat: torch.Tensor, n: int, method: str = "full") -> torch.Tensor:
    """Computes covariance matrix from flat representation according to choice of method.

    Args:
        flat (torch.Tensor): The flat tensor parameter.
        n (int): The dimension of the matrix.
        method (str, optional): The method, full, diagonal or ball. Defaults to "full".

    Raises:
        ValueError: If method 'full' and number of elements are inconsistent with n.
        ValueError: If method 'diag' and number of elements are inconsistent with n.
        ValueError: If method 'ball' and number of elements is not one.

    Returns:
        torch.Tensor: The flat representation.
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

    return (L @ L.T).inverse()


def flat_from_cov(V: torch.Tensor, method: str = "full") -> torch.Tensor:
    """Computes flat tensor from covariance matrix according to choice of method.

    Args:
        V (torch.Tensor): The square covariance matrix parameter.
        method (str, optional): The method, full, diagonal or ball. Defaults to "full".

    Returns:
        torch.Tensor: The flat representation.
    """
    L = cast(torch.Tensor, torch.linalg.cholesky(V.inverse()))  # type: ignore
    L.diagonal().log_()
    return _flat_from_log_cholesky(L, method)


def get_cholesky_and_log_eigvals(
    params: "ModelParams", matrix: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get Cholesky factor as well as log eigenvalues.

    Args:
        params (ModelParams): The model parameters.
        matrix (str): Either "Q" or "R".

    Raises:
        ValueError: If the matrix is not in ("Q", "R")

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Precision matrix and log eigenvalues.
    """
    if matrix not in ("Q", "R"):
        raise ValueError(f"matrix should be either Q or R, got {matrix}")

    # Get flat then log cholesky
    flat, method = getattr(params, matrix + "_repr")
    n = getattr(params, matrix + "_dim_")

    L = _log_cholesky_from_flat(flat, n, method)
    eigvals = 2 * L.diagonal()
    L.diagonal().exp_()

    return L, eigvals
