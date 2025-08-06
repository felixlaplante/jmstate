import warnings
from typing import Callable

import torch

from ..typedefs._defs import Info, Job, Metrics, Tensor1D, Tensor2D, Tensor3D
from ..utils._misc import params_like_from_flat


class ComputeFIM(Job):
    """Job to compute the Fisher Information Matrix."""

    grad_m1: Tensor1D
    grad_m2: Tensor2D
    jac_fn: Callable[[Tensor1D, Tensor3D], Tensor2D]

    def init(self, info: Info):
        if not info.model.fit_:
            warnings.warn(
                "Model should be (random) fitted before computing Fisher Information Matrix",
                stacklevel=2,
            )

        d = info.model.params_.numel
        self.grad_m1 = torch.zeros(d, dtype=torch.float32)
        self.grad_m2 = torch.zeros((d, d), dtype=torch.float32)

        def _jac_fn(params_flat_tensor: Tensor1D, b: Tensor3D):
            params = params_like_from_flat(info.model.params_, params_flat_tensor)
            return info.logpdfs_fn(params, b).sum(dim=1)

        self.jac_fn = torch.func.jacrev(_jac_fn)  # type: ignore

    def run(self, info: Info):
        jac = self.jac_fn(info.model.params_.as_flat_tensor, info.b)

        self.grad_m1 += jac.mean(dim=0)
        self.grad_m2 += (jac.T @ jac) / info.sampler.n_chains

    def end(self, info: Info, metrics: Metrics):
        self.grad_m2 /= info.iteration
        self.grad_m1 /= info.iteration
        metrics.fim = self.grad_m2 - torch.outer(self.grad_m1, self.grad_m1)


class ComputeCriteria(Job):
    """Job to compute AIC, BIC and Log Likelihood."""

    n: int
    loglik: float

    def __init__(self):
        self.loglik = 0.0

    def init(self, info: Info):
        self.n = info.data.size

    def run(self, info: Info):
        self.loglik += info.logliks.detach().sum().item() / info.sampler.n_chains

    def end(self, info: Info, metrics: Metrics):
        metrics.loglik = self.loglik / info.iteration
        metrics.nloglik_pen = (
            self.n * info.model.pen(info.model.params_).item() - metrics.loglik
            if info.model.pen is not None
            else -metrics.loglik
        )

        d = info.model.params_.numel
        aic_pen = 2 * d
        bic_pen = (
            d
            * torch.log(
                torch.tensor(info.data.effective_size, dtype=torch.float32)
            ).item()
        )

        metrics.aic = 2 * metrics.nloglik_pen + aic_pen
        metrics.bic = 2 * metrics.nloglik_pen + bic_pen


class ComputeEBEs(Job):
    """Job to compute the EBEs of b."""

    ebes: Tensor2D

    def init(self, info: Info):
        self.ebes = torch.zeros(info.sampler.state.shape[1:], dtype=torch.float32)

    def run(self, info: Info):
        self.ebes += info.b.detach().mean(dim=0)

    def end(self, info: Info, metrics: Metrics):
        metrics.ebes = self.ebes / info.iteration
