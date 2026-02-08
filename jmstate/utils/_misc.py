from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from ..typedefs._defs import Info, Job
from ..utils._dtype import get_dtype

if TYPE_CHECKING:
    from ..typedefs._params import ModelParams


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

    dtype = get_dtype()
    std_nodes = torch.tensor(nodes, dtype=dtype).unsqueeze(0)
    std_weights = torch.tensor(weights, dtype=dtype)

    return std_nodes, std_weights


def map_fn_params(
    params: ModelParams, fn: Callable[[torch.Tensor], torch.Tensor]
) -> ModelParams:
    """Map operation and get new parameters.

    Args:
        params (ModelParams): The model parameters to use.
        fn (Callable[[torch.Tensor], torch.Tensor]): The operation.

    Returns:
        ModelParams: The new parameters (it might be a reshape).
    """
    from ..typedefs._params import ModelParams  # noqa: PLC0415

    def _map_fn(dict: dict[tuple[Any, Any], torch.Tensor]):
        return {key: fn(val) for key, val in dict.items()}

    return ModelParams(
        None if params.gamma is None else fn(params.gamma),
        params.Q_repr._replace(flat=fn(params.Q_repr.flat)),
        params.R_repr._replace(flat=fn(params.R_repr.flat)),
        _map_fn(params.alphas),
        None if params.betas is None else _map_fn(params.betas),
        extra=params.extra,
        skip_validation=True,
    )


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
        result = job.run(info=info)
        stop = (
            stop if result is None else (result if stop is None else (stop and result))
        )

    return False if stop is None else stop
