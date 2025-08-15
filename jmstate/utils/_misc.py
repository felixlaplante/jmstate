from typing import Any, cast

import numpy as np
import torch

from ..typedefs._defs import Info, Job


def legendre_quad(n_quad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the Legendre quadrature nodes and weights.

    Args:
        n_quad (int): The number of quadrature points.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The nodes and weights.
    """
    nodes, weights = cast(
        tuple[
            np.ndarray[Any, np.dtype[np.float64]],
            np.ndarray[Any, np.dtype[np.float64]],
        ],
        np.polynomial.legendre.leggauss(n_quad),  # type: ignore
    )

    std_nodes = torch.tensor(nodes, dtype=torch.get_default_dtype()).unsqueeze(0)
    std_weights = torch.tensor(weights, dtype=torch.get_default_dtype())

    return std_nodes, std_weights


def run_jobs(jobs: list[Job], info: Info) -> bool:
    """Call jobs.

    Args:
        jobs (list[Job]): The jobs to execute.
        info (Info): The information container.

    Returns:
        bool: Set to true to stop the iterations.
    """
    stop = None
    for job in jobs:
        result = job.run(info)
        stop = (
            stop if result is None else (result if stop is None else (stop and result))
        )

    return False if stop is None else stop
