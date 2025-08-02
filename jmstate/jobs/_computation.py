import warnings

import torch
from beartype import beartype

from ..typedefs._defs import Info, Job, Metrics, Tensor1D, Tensor2D


class ComputeFIM(Job):
    """Job to compute the Fisher Information Matrix."""

    retain_graph: bool
    grad_m1: Tensor1D
    grad_m2: Tensor2D
    scale: float

    @beartype
    def __init__(self, retain_graph: bool = False):
        self.retain_graph = retain_graph

    def init(self, info: Info, metrics: Metrics):  # noqa: ARG002
        if not info.model.fit_:
            warnings.warn(
                "Model should be fit before computing Fisher Information Matrix",
                stacklevel=2,
            )

        info.model.params_.require_grad(True)

        d = info.model.params_.numel
        self.grad_m1 = torch.zeros(d, dtype=torch.float32)
        self.grad_m2 = torch.zeros((d, d), dtype=torch.float32)

        self.scale = 1.0 / (info.n_iterations * info.n_chains)

    def run(self, info: Info, metrics: Metrics):  # noqa: ARG002
        for i in range(info.n_chains):
            grads = torch.autograd.grad(
                outputs=info.logpdfs[i].sum(),
                inputs=info.model.params_.as_list,
                retain_graph=(i - self.retain_graph) < (info.n_chains - 1),
            )
            grads = torch.cat([p.view(-1) for p in grads])

            self.grad_m1.add_(grads, alpha=self.scale)
            self.grad_m2.add_(torch.outer(grads, grads), alpha=self.scale)

    def end(self, info: Info, metrics: Metrics):
        metrics.fim = self.grad_m2 - torch.outer(self.grad_m1, self.grad_m1)
        info.model.params_.require_grad(False)


class ComputeCriteria(Job):
    """Job to compute AIC, BIC and Log Likelihood."""

    n: int
    scale: float

    def init(self, info: Info, metrics: Metrics):
        self.n = info.data.size
        self.scale = 1.0 / (info.n_iterations * info.n_chains)
        metrics.loglik = 0.0

    def run(self, info: Info, metrics: Metrics):
        metrics.loglik += info.logliks.detach().sum().item() * self.scale

    def end(self, info: Info, metrics: Metrics):
        metrics.nloglik_pen = (
            self.n * info.model.pen(info.model.params_).item() - metrics.loglik
            if info.model.pen is not None
            else -metrics.loglik
        )

        d = info.model.params_.numel
        aic_pen = 2 * d
        bic_pen = d * torch.log(torch.tensor(self.n, dtype=torch.float32)).item()

        metrics.aic = 2 * metrics.nloglik_pen + aic_pen
        metrics.bic = 2 * metrics.nloglik_pen + bic_pen


class ComputeEBEs(Job):
    """Job to compute the EBEs of b."""

    ebes: Tensor2D
    scale: float

    def init(self, info: Info, metrics: Metrics):  # noqa: ARG002
        self.ebes = torch.zeros(
            info.sampler.current_state.shape[1:], dtype=torch.float32
        )
        self.scale = 1.0 / (info.n_iterations * info.n_chains)

    def run(self, info: Info, metrics: Metrics):  # noqa: ARG002
        self.ebes.add_(info.b.detach().mean(dim=0), alpha=self.scale)

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        metrics.ebes = self.ebes
