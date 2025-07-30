import warnings
from typing import Any

import torch

from ..typedefs._structures import AllInfo, BaseInfo, Job


class ComputeFIM(Job):
    """Job to compute the Fisher Information Matrix."""

    retain_graph: bool
    grad_m1: torch.Tensor
    grad_m2: torch.Tensor

    def __init__(self, retain_graph: bool = False):
        self.retain_graph = retain_graph

    def init(self, info: BaseInfo, metrics: dict[str, Any]) -> None:
        if not info.model.fit_:
            warnings.warn(
                "Model should be fit before computing Fisher Information Matrix",
                stacklevel=2,
            )

        info.model.params_.require_grad(True)
        d = info.model.params_.numel
        self.grad_m1 = torch.zeros(d, dtype=torch.float32)
        self.grad_m2 = torch.zeros((d, d), dtype=torch.float32)

    def run(self, info: AllInfo, metrics: dict[str, Any]) -> None:
        for p in info.model.params_.as_list:
            if p.grad is not None:
                p.grad.zero_()

        info.logpdfs.sum().backward(retain_graph=self.retain_graph)  # type: ignore

        grad_chunks: list[torch.Tensor] = []
        for p in info.model.params_.as_list:
            if p.grad is not None:
                grad_chunks.append(p.grad.view(-1))
            else:
                grad_chunks.append(torch.zeros(p.numel()))

        grad = torch.cat(grad_chunks)

        # Update moments
        self.grad_m1 += grad / info.n_iterations
        self.grad_m2 += torch.outer(grad, grad) / info.n_iterations

    def end(self, info: BaseInfo, metrics: dict[str, Any]) -> None:
        metrics["fim"] = self.grad_m2 - torch.outer(self.grad_m1, self.grad_m1)
        info.model.params_.require_grad(False)


class ComputeCriteria(Job):
    """Job to compute the Fisher Information Matrix."""

    def init(self, info: BaseInfo, metrics: dict[str, Any]) -> None:
        if not info.model.fit_:
            warnings.warn(
                "Model should be fit before computing logLik, AIC, or BIC", stacklevel=2
            )

        metrics["loglik"] = torch.zeros(1, dtype=torch.float32)

    def run(self, info: AllInfo, metrics: dict[str, Any]) -> None:
        metrics["loglik"] += info.logliks.detach().sum() / info.n_iterations

    def end(self, info: BaseInfo, metrics: dict[str, Any]) -> None:
        metrics["nloglik_pen"] = (
            info.model.pen(info.model.params_) - metrics["loglik"]
            if info.model.pen is not None
            else -metrics["loglik"]
        )

        d = info.model.params_.numel
        aic_pen = 2 * d
        bic_pen = d * torch.log(torch.tensor(info.data.size, dtype=torch.float32))

        metrics["aic"] = 2 * metrics["nloglik_pen"] + aic_pen
        metrics["bic"] = 2 * metrics["nloglik_pen"] + bic_pen


class ComputeEBEs(Job):
    """Job to compute the EBEs of b."""

    def init(self, info: BaseInfo, metrics: dict[str, Any]) -> None:
        metrics["b_ebes"] = torch.zeros_like(
            info.sampler.current_state, dtype=torch.float32
        )

    def run(self, info: AllInfo, metrics: dict[str, Any]) -> None:
        metrics["b_ebes"] += info.b.detach() / info.n_iterations

    def end(self, info: BaseInfo, metrics: dict[str, Any]) -> None:
        pass
