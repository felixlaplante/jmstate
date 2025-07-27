from typing import Any
import warnings

import torch


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
    tot_loglik = info["tot_loglik"]
    tot_nloglik_pen = info["tot_nloglik_pen"]
    start = info["start"]
    end = info["end"]

    if not model.fit_ and start:
        warnings.warn("Model should be fit before computing logLik, AIC, or BIC")

    if start:
        metrics["loglik"] = torch.zeros(1, dtype=torch.float32)
        metrics["nloglik_pen"] = torch.zeros(1, dtype=torch.float32)

    # Update
    metrics["loglik"] += tot_loglik.detach() / n_iter
    metrics["nloglik_pen"] += tot_nloglik_pen.detach() / n_iter

    # Set other metrics when it is done
    if end:
        aic_pen = 2 * params.numel
        bic_pen = params.numel * torch.log(torch.tensor(data.size, dtype=torch.float32))

        metrics["aic"] = 2 * metrics["nloglik_pen"] + aic_pen
        metrics["bic"] = 2 * metrics["nloglik_pen"] + bic_pen
