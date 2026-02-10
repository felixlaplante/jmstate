import torch


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


def log_cholesky_from_flat(
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


def flat_from_log_cholesky(L: torch.Tensor, method: str = "full") -> torch.Tensor:
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
