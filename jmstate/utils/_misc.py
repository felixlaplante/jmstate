from typing import Any, Callable, cast

import numpy as np
import torch


def legendre_quad(n_quad: int) -> tuple[torch.Tensor, ...]:
    """Get the Legendre quadrature nodes and weights.

    Args:
        n_quad (int, optional): The number of quadrature points.

    Returns:
        tuple[torch.Tensor, ...]: The nodes and weights.
    """
    nodes, weights = cast(
        tuple[
            np.ndarray[Any, np.dtype[np.float32]],
            np.ndarray[Any, np.dtype[np.float32]],
        ],
        np.polynomial.legendre.leggauss(n_quad),  # type: ignore
    )

    std_nodes = torch.tensor(nodes, dtype=torch.float32).unsqueeze(0)
    std_weights = torch.tensor(weights, dtype=torch.float32)

    return std_nodes, std_weights


def call_callbacks(
    callbacks: Callable[..., None] | list[Callable[..., None]] | None,
    *args: Any,
    **kwargs: Any,
):
    """Call one or multiple functions.

    Args:
        callbacks (Callable[..., None] | list[Callable[..., None]] | None): The function(s) to call, or None.
    """
    if callbacks is None:
        return
    if callable(callbacks):
        callbacks = [callbacks]
    for callback in callbacks:
        callback(*args, **kwargs)
