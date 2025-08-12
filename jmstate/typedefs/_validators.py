from collections.abc import Callable

import torch


def is_ndim(ndim: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Checks the number of dimension of a tensor.

    Args:
        ndim (int): The expected ndim.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: The checker function.
    """

    def _is_ndim(t: torch.Tensor) -> torch.Tensor:
        if t.ndim != ndim:
            raise ValueError(f"expected {ndim}D, got {t.ndim}D")
        return t

    return _is_ndim


def is_col(t: torch.Tensor) -> torch.Tensor:
    """Checks if a tensor is a column vector.

    Args:
        t (torch.Tensor): The tensor to test.

    Raises:
        ValueError: If the tensor is not a column vector.

    Returns:
        torch.Tensor: The output tensor.
    """
    if t.size(1) != 1:
        raise ValueError(f"expected column tensor, got  shape {t.shape}")
    return t


def is_non_neg(x: int | float | torch.Tensor) -> int | float | torch.Tensor:
    """Checks if the argument is non-negative.

    Args:
        x (int | float | torch.Tensor): The input number or tensor.

    Raises:
        ValueError: If the tensor is not all non-negative.
        ValueError: If the number is not non-negative.

    Returns:
        int | float | torch.Tensor: The output number or tensor.
    """
    if isinstance(x, torch.Tensor) and (x < 0).any():
        raise ValueError(f"Expected non-negative tensor, got {x}")
    if x < 0:
        raise ValueError(f"Expected non-negative number, got {x}")
    return x


def is_strict_pos(x: int | float | torch.Tensor) -> int | float | torch.Tensor:
    """Checks if the argument is strictly positive.

    Args:
        x (int | float | torch.Tensor): The input number or tensor.

    Raises:
        ValueError: If the tensor is not all strictly positive.
        ValueError: If the number is not strictly positive.

    Returns:
        int | float | torch.Tensor: The output number or tensor.
    """
    if isinstance(x, torch.Tensor) and (x <= 0).any():
        raise ValueError(f"Expected strictly positive tensor, got {x}")
    if x <= 0:
        raise ValueError(f"Expected strictly positive number, got {x}")
    return x


def is_prob(x: int | float) -> int | float:
    """Checks if the number is a probability.

    Args:
        x (int | float): The number to check.

    Raises:
        ValueError: If the number is not 0 < x < 1.

    Returns:
        int | float: The output number.
    """
    if not 0 < x < 1:
        raise ValueError(f"Expected probability, got {x}")
    return x
