import warnings
from typing import Any

import torch


def compute_fim(info: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Callback to compute the Fisher Information Matrix.

    Args:
        info (dict[str, Any]): The information dict.
        metrics (dict[str, Any]): The metrics dict.
    """
    model = info["model"]
    params = info["params"]
    n_iter = info["n_iter"]
    start = info["start"]
    end = info["end"]

    if not model.fit_ and start:
        warnings.warn(
            "Model should be fit before computing Fisher Information Matrix",
            stacklevel=2,
        )

    if start:
        d = params.numel
        metrics["grad_m1"] = torch.zeros(d, dtype=torch.float32)
        metrics["grad_m2"] = torch.zeros((d, d), dtype=torch.float32)

    # Collect gradient vector
    grad_chunks: list[torch.Tensor] = []
    for p in params.as_list:
        if p.grad is not None:
            grad_chunks.append(p.grad.view(-1))
        else:
            grad_chunks.append(torch.zeros(p.numel()))

    grad = torch.cat(grad_chunks)

    # Update
    metrics["grad_m1"] += grad / n_iter
    metrics["grad_m2"] += torch.outer(grad, grad) / n_iter

    # Set Fisher Information Matrix if it is done
    if end:
        metrics["fim"] = metrics["grad_m2"] - torch.outer(
            metrics["grad_m1"], metrics["grad_m1"]
        )
        metrics.pop("grad_m1")
        metrics.pop("grad_m2")


def compute_criteria(info: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Callback to compute the Fisher Information Matrix.

    Args:
        info (dict[str, Any]): The information dict.
        metrics (dict[str, Any]): The metrics dict.
    """
    data = info["data"]
    model = info["model"]
    params = info["params"]
    n_iter = info["n_iter"]
    loglik = info["loglik"]
    nloglik_pen = info["nloglik_pen"]
    start = info["start"]
    end = info["end"]

    if not model.fit_ and start:
        warnings.warn(
            "Model should be fit before computing logLik, AIC, or BIC", stacklevel=2
        )

    if start:
        metrics["loglik"] = torch.zeros(1, dtype=torch.float32)
        metrics["nloglik_pen"] = torch.zeros(1, dtype=torch.float32)

    # Update
    metrics["loglik"] += loglik.detach() / n_iter
    metrics["nloglik_pen"] += nloglik_pen.detach() / n_iter

    # Set other metrics when it is done
    if end:
        aic_pen = 2 * params.numel
        bic_pen = params.numel * torch.log(torch.tensor(data.size, dtype=torch.float32))

        metrics["aic"] = 2 * metrics["nloglik_pen"] + aic_pen
        metrics["bic"] = 2 * metrics["nloglik_pen"] + bic_pen


def compute_ebes(info: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Callback to compute the EBEs of the random effects.

    Args:
        info (dict[str, Any]): The information dict.
        metrics (dict[str, Any]): The metrics dict.
    """
    b = info["b"]
    n_iter = info["n_iter"]
    start = info["start"]

    if start:
        metrics["b_ebes"] = torch.zeros_like(b, dtype=torch.float32)

    # Update
    metrics["b_ebes"] += b.detach() / n_iter
