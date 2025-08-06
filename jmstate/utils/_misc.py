from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from beartype import beartype

from ..typedefs._defs import Info, Job, Tensor1D, TensorRow
from ..typedefs._params import ModelParams


def legendre_quad(n_quad: int) -> tuple[TensorRow, Tensor1D]:
    """Get the Legendre quadrature nodes and weights.

    Args:
        n_quad (int, optional): The number of quadrature points.

    Returns:
        tuple[TensorRow, Tensor1D]: The nodes and weights.
    """
    nodes, weights = cast(
        tuple[
            np.ndarray[Any, np.dtype[np.float64]],
            np.ndarray[Any, np.dtype[np.float64]],
        ],
        np.polynomial.legendre.leggauss(n_quad),  # type: ignore
    )

    std_nodes = torch.tensor(nodes, dtype=torch.float32).unsqueeze(0)
    std_weights = torch.tensor(weights, dtype=torch.float32)

    return std_nodes, std_weights


@beartype
def params_like_from_flat(ref_params: ModelParams, flat: Tensor1D) -> ModelParams:
    """Gets a ModelParams instance based on the flat representation.

    Args:
        ref_params (ModelParams): The reference params.
        flat (torch.Tensor): The flat representation.

    Raises:
        ValueError: If the shape makes the conversion impossible.

    Returns:
        ModelParams: The constructed ModelParams.
    """
    from ..typedefs._params import ModelParams  # noqa: PLC0415

    i = 0

    def _next(ref: torch.Tensor):
        nonlocal i
        n = ref.numel()
        result = flat[i : i + n]
        i += n
        return result.view(ref.shape)

    gamma = None if ref_params.gamma is None else _next(ref_params.gamma)

    Q_flat = _next(ref_params.Q_repr.flat)
    R_flat = _next(ref_params.R_repr.flat)

    alphas = {key: _next(val) for key, val in ref_params.alphas.items()}

    betas = (
        None
        if ref_params.betas is None
        else {key: _next(val) for key, val in ref_params.betas.items()}
    )

    return ModelParams(
        gamma,
        ref_params.Q_repr._replace(flat=Q_flat),
        ref_params.R_repr._replace(flat=R_flat),
        alphas,
        betas,
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
        result = job.run(info)
        stop = (
            stop if result is None else (result if stop is None else (stop and result))
        )

    return False if stop is None else stop
