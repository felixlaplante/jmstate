import warnings

import torch

from ..typedefs._defs import Info, Job, Metrics


class ComputeFIM(Job):
    """Job to compute the Fisher Information Matrix."""

    retain_graph: bool
    grad_m1: torch.Tensor
    grad_m2: torch.Tensor

    def __init__(self, retain_graph: bool = False):
        self.retain_graph = retain_graph

    def init(self, info: Info, metrics: Metrics) -> None:
        if not info.model.fit_:
            warnings.warn(
                "Model should be fit before computing Fisher Information Matrix",
                stacklevel=2,
            )
        if info.n_chains != 1:
            warnings.warn(
                "n_chains should be set to 1",
                stacklevel=2,
            )

        info.model.params_.require_grad(True)
        d = info.model.params_.numel
        self.grad_m1 = torch.zeros(d, dtype=torch.float32)
        self.grad_m2 = torch.zeros((d, d), dtype=torch.float32)

    def run(self, info: Info, metrics: Metrics) -> None:
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

    def end(self, info: Info, metrics: Metrics) -> None:
        metrics.fim = self.grad_m2 - torch.outer(self.grad_m1, self.grad_m1)
        info.model.params_.require_grad(False)


class ComputeCriteria(Job):
    """Job to compute AIC, BIC and Log Likelihood."""

    n: int
    m: int

    def init(self, info: Info, metrics: Metrics) -> None:
        self.n = info.data.size
        self.m = info.n_iterations * info.n_chains
        metrics.loglik = 0.0

    def run(self, info: Info, metrics: Metrics) -> None:
        metrics.loglik += info.logliks.detach().sum().item() / self.m

    def end(self, info: Info, metrics: Metrics) -> None:
        metrics.nloglik_pen = (
            self.n * info.model.pen(info.model.params_).item() - metrics.loglik
            if info.model.pen is not None
            else -metrics.loglik
        )

        d = info.model.params_.numel
        aic_pen = 2 * d
        bic_pen = (
            d * torch.log(torch.tensor(self.n, dtype=torch.float32)).item()
        )

        metrics.aic = 2 * metrics.nloglik_pen + aic_pen
        metrics.bic = 2 * metrics.nloglik_pen + bic_pen


class ComputeEBEs(Job):
    """Job to compute the EBEs of b."""

    n: int
    p: int

    def init(self, info: Info, metrics: Metrics) -> None:
        self.n, self.p = info.data.size, info.model.params_.Q_dim_

        metrics.ebes = torch.zeros((self.n, self.p), dtype=torch.float32)

    def run(self, info: Info, metrics: Metrics) -> None:
        metrics.ebes += (
            info.b.detach().view(-1, self.n, self.p).mean(dim=0) / info.n_iterations
        )

    def end(self, info: Info, metrics: Metrics) -> None:
        pass
