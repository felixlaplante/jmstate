from typing import Any

import torch


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