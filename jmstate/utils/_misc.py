from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from beartype import beartype

from ..typedefs._defs import Info, Job, Metrics, Tensor1D, Tensor2D
from ..typedefs._params import ModelParams


def legendre_quad(n_quad: int) -> tuple[Tensor1D | Tensor2D, ...]:
    """Get the Legendre quadrature nodes and weights.

    Args:
        n_quad (int, optional): The number of quadrature points.

    Returns:
        tuple[Tensor1D | Tensor2D, ...]: The nodes and weights.
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

    gamma = _next(ref_params.gamma) if ref_params.gamma is not None else None

    Q_flat = _next(ref_params.Q_repr[0])
    Q_method = ref_params.Q_repr[1]

    R_flat = _next(ref_params.R_repr[0])
    R_method = ref_params.R_repr[1]

    alphas = {key: _next(val) for key, val in ref_params.alphas.items()}

    betas = (
        {key: _next(val) for key, val in ref_params.betas.items()}
        if ref_params.betas is not None
        else None
    )

    return ModelParams(
        gamma,
        (Q_flat, Q_method),
        (R_flat, R_method),
        alphas,
        betas,
        skip_validation=True,
    )


def do_jobs(
    method: str,
    jobs: Job | list[Job],
    info: Info,
    metrics: Metrics,
):
    """Call jobs.

    Args:
        method (str): Either 'init', 'run' or 'end'.
        jobs (Job | list[Job] | None): The jobs to execute.
        info (Info): The information container.
        metrics (Metrics): The computed metrics dict output.
    """
    if isinstance(jobs, Job):
        jobs = [jobs]

    match method:
        case "init":
            for job in jobs:
                job.init(info, metrics)
        case "run":
            for job in jobs:
                job.run(info, metrics)
        case "end":
            for job in jobs:
                job.end(info, metrics)
        case _:
            pass
