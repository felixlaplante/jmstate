from typing import cast

import torch


def tril_from_flat(flat: torch.Tensor, n: int) -> torch.Tensor:
    """Generate the lower triangular matrix associated with flat tensor.

    Args:
        flat (torch.Tensor): Flat tehsnro
        n (int): Dimension of the matrix.

    Raises:
        ValueError: Error if the the dimensions do not allow matrix computation.
        RuntimeError: Error if the computation fails.

    Returns:
        torch.Tensor: The lower triangular matrix.
    """
    if flat.numel() != (n * (n + 1)) // 2:
        raise ValueError("Incompatible dimensions for lower triangular matrix")

    return torch.zeros(n, n, dtype=flat.dtype).index_put_(
        tuple(torch.tril_indices(n, n)), flat
    )


def flat_from_tril(L: torch.Tensor) -> torch.Tensor:
    """Flatten the lower triangular part (including the diagonal) of a square matrix.

    Into a 1D tensor, in row-wise order.

    Args:
        L (torch.Tensor): Square lower-triangular matrix of shape (n, n).

    Raises:
        ValueError: If the input is not square.
        RuntimeError: If the flattening fails.

    Returns:
        torch.Tensor: Flattened 1D tensor containing the lower triangular entries.
    """
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("Input must be a square matrix")

    n = L.shape[0]
    i, j = torch.tril_indices(n, n)

    return L[i, j]


def log_cholesky_from_flat(
    flat: torch.Tensor, n: int, method: str = "full"
) -> torch.Tensor:
    """Computes log cholesky from flat tensor according to choice of method.

    Args:
        flat (torch.Tensor): The flat tensor parameter.
        n (int): The dimension of the matrix.
        method (str, optional): The method, full, diagonal or ball. Defaults to "full".

    Raises:
        ValueError: If the array is not flat.
        ValueError: If the number of parameters is inconsistent with n.
        ValueError: If the number of parameters does not equal one.
        ValueError: If the method is not in ("full", "diag", "ball").

    Returns:
        torch.Tensor: The log cholesky representation.
    """
    if flat.ndim != 1:
        raise ValueError(f"flat should be flat, got shape {flat.shape}")

    match method:
        case "full":
            return tril_from_flat(flat, n)
        case "diag":
            if flat.numel() != n:
                raise ValueError(f"flat has {flat.numel()} elements, expected {n}")
            return torch.diag(flat)
        case "ball":
            if flat.numel() != 1:
                f"flat has {flat.numel()} elements, expected 1"
            return flat * torch.eye(n)
        case _:
            raise ValueError(f"Got method {method} unknown")


def flat_from_log_cholesky(L: torch.Tensor, method: str = "full") -> torch.Tensor:
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
            return flat_from_tril(L)
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
        method (str, optional): The method, full, diagonal or ball. Defaults to "full".

    Returns:
        torch.Tensor: The flat representation.
    """
    L = log_cholesky_from_flat(flat, n, method)
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
    return flat_from_log_cholesky(L, method)
