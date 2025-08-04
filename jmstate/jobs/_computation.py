import warnings
from typing import Callable

import torch

from ..typedefs._defs import Info, Job, Metrics, Tensor1D, Tensor2D, Tensor3D
from ..utils._misc import params_like_from_flat


class ComputeFIM(Job):
    """Job to compute the Fisher Information Matrix."""

    grad_m1: Tensor1D
    grad_m2: Tensor2D
    scale: float
    jac_fn: Callable[[Tensor1D, Tensor3D], Tensor2D]

    def init(self, info: Info, metrics: Metrics):  # noqa: ARG002
        if not info.model.fit_:
            warnings.warn(
                "Model should be fit before computing Fisher Information Matrix",
                stacklevel=2,
            )

        d = info.model.params_.numel
        self.grad_m1 = torch.zeros(d, dtype=torch.float32)
        self.grad_m2 = torch.zeros((d, d), dtype=torch.float32)

        self.scale = 1.0 / (info.n_iterations * info.sampler.n_chains)

        def _jac_fn(params_flat_tensor: Tensor1D, b: Tensor3D):
            params = params_like_from_flat(info.model.params_, params_flat_tensor)
            return info.logpdfs_fn(params, b).sum(dim=1)

        self.jac_fn = torch.func.jacrev(_jac_fn)  # type: ignore

    def run(self, info: Info, metrics: Metrics):  # noqa: ARG002
        jac = self.jac_fn(info.model.params_.as_flat_tensor, info.b)

        self.grad_m1.add_(jac.mean(dim=0), alpha=self.scale)
        self.grad_m2.add_(jac.T @ jac, alpha=self.scale)

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        metrics.fim = self.grad_m2 - torch.outer(self.grad_m1, self.grad_m1)


class ComputeCriteria(Job):
    """Job to compute AIC, BIC and Log Likelihood."""

    n: int
    scale: float
    loglik: float

    def init(self, info: Info, metrics: Metrics):
        self.n = info.data.size
        self.scale = 1.0 / (info.n_iterations * info.sampler.n_chains)
        self.loglik = 0.0

    def run(self, info: Info, metrics: Metrics):
        self.loglik += info.logliks.detach().sum().item() * self.scale

    def end(self, info: Info, metrics: Metrics):
        metrics.loglik = self.loglik
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
        self.scale = 1.0 / (info.n_iterations)

    def run(self, info: Info, metrics: Metrics):  # noqa: ARG002
        self.ebes.add_(info.b.detach().mean(dim=0), alpha=self.scale)

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        metrics.ebes = self.ebes
