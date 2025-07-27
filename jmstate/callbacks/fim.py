from typing import Any
import warnings

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
        warnings.warn("Model should be fit before computing Fisher Information Matrix")

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


