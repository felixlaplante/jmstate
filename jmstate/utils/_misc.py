from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from ..model._base import MultiStateJointModel
    from ..typedefs._structures import AllInfo, BaseInfo, Job, ModelParams


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


def params_like_from_flat(
    ref_params: "ModelParams", flat: torch.Tensor
) -> "ModelParams":
    """Gets a ModelParams instance based on the flat representation.

    Args:
        ref_params (ModelParams): The reference params.
        flat (torch.Tensor): The flat representation.

    Raises:
        ValueError: If the shape makes the conversion impossible.

    Returns:
        _type_: The constructed ModelParams.
    """
    from ..typedefs._structures import ModelParams  # noqa: PLC0415

    if flat.shape != (ref_params.numel,):
        raise ValueError(f"flat.shape is {flat.shape} is not {(ref_params.numel,)}")

    i = 0

    def _next(ref: torch.Tensor):
        nonlocal i
        n = ref.numel()
        result = flat[i : i + n]
        i += n
        return result

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

    return ModelParams(gamma, (Q_flat, Q_method), (R_flat, R_method), alphas, betas)


def sample_params_from_model(
    model: "MultiStateJointModel", sample_size: int
) -> list["ModelParams"]:
    """Sample parameters based on asymptotic behavior of the MLE.

    Args:
        model (MultiStateJointModel): The fitted model.
        sample_size (int): The desired sample size.

    Raises:
        ValueError: If the model has not been fitted, or Fisher Information Matrix not computed.

    Returns:
        list[ModelParams]: A list of model parameters.
    """
    if not model.fit_:
        raise ValueError("Model must be fit")

    dist = torch.distributions.MultivariateNormal(
        model.params_.as_flat_tensor, model.fim.inverse()
    )
    flat_samples = dist.sample((sample_size,))

    return [params_like_from_flat(model.params_, sample) for sample in flat_samples]


def do_jobs(
    method: str,
    jobs: "Job | list[Job]",
    info: "BaseInfo | AllInfo",
    metrics: dict[str, Any],
) -> None:
    """Call jobs.

    Args:
        method (str): Either 'init', 'run' or 'end'.
        jobs (Job | list[Job] | None): The jobs to execute.
        info (BaseInfo | AllInfo): The information container.
        metrics (dict[str, Any]): The computed metrics dict output.
    """
    from ..typedefs._structures import AllInfo, Job

    if isinstance(jobs, Job):
        jobs = [jobs]

    match method:
        case "init":
            for job in jobs:
                job.init(info=info, metrics=metrics)
        case "run":
            for job in jobs:
                job.run(info=cast(AllInfo, info), metrics=metrics)
        case "end":
            for job in jobs:
                job.end(info=info, metrics=metrics)
        case _:
            pass
