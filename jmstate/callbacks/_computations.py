import warnings
from typing import Any

import torch

from ..types._structures import CallbackFn


class ComputeFIM(CallbackFn):
    """Callback to compute the Fisher Information Matrix."""

    def init(
        self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
    ) -> None:
        if not info["model"].fit_:
            warnings.warn(
                "Model should be fit before computing Fisher Information Matrix",
                stacklevel=2,
            )
        d = info["params"].numel
        tmp["grad_m1"] = torch.zeros(d, dtype=torch.float32)
        tmp["grad_m2"] = torch.zeros((d, d), dtype=torch.float32)

    def run(
        self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
    ) -> None:
        grad_chunks: list[torch.Tensor] = []
        for p in info["params"].as_list:
            if p.grad is not None:
                grad_chunks.append(p.grad.view(-1))
            else:
                grad_chunks.append(torch.zeros(p.numel()))

        grad = torch.cat(grad_chunks)

        # Update
        tmp["grad_m1"] += grad / info["n_iter"]
        tmp["grad_m2"] += torch.outer(grad, grad) / info["n_iter"]

    def end(
        self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
    ) -> None:
        metrics["fim"] = tmp["grad_m2"] - torch.outer(tmp["grad_m1"], tmp["grad_m1"])
        tmp.pop("grad_m1")
        tmp.pop("grad_m2")


class ComputeCriteria(CallbackFn):
    """Callback to compute the Fisher Information Matrix."""

    def init(
        self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
    ) -> None:
        if not info["model"].fit_:
            warnings.warn(
                "Model should be fit before computing logLik, AIC, or BIC", stacklevel=2
            )
        metrics["loglik"] = torch.zeros(1, dtype=torch.float32)
        metrics["nloglik_pen"] = torch.zeros(1, dtype=torch.float32)

    def run(
        self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
    ) -> None:
        metrics["loglik"] += info["loglik"].detach() / info["n_iter"]
        metrics["nloglik_pen"] += info["nloglik_pen"].detach() / info["n_iter"]

    def end(
        self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
    ) -> None:
        aic_pen = 2 * info["params"].numel
        bic_pen = info["params"].numel * torch.log(
            torch.tensor(info["data"].size, dtype=torch.float32)
        )

        metrics["aic"] = 2 * metrics["nloglik_pen"] + aic_pen
        metrics["bic"] = 2 * metrics["nloglik_pen"] + bic_pen


class ComputeEBEs(CallbackFn):
    """Callback to compute the EBEs of b."""

    def init(
        self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
    ) -> None:
        metrics["b_ebes"] = torch.zeros_like(
            info["sampler"].current_state, dtype=torch.float32
        )

    def run(
        self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
    ) -> None:
        metrics["b_ebes"] += info["b"].detach() / info["n_iter"]

    def end(
        self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
    ) -> None:
        pass
