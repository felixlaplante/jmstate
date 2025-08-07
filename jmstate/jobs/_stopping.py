import warnings

import torch
from beartype import beartype

from ..typedefs._defs import (
    Info,
    IntPositive,
    IntStrictlyPositive,
    Job,
    Metrics,
    NumPositive,
    NumProbability,
    Tensor1D,
)

# Constants
DEFAULT_NOT_CONVERGED_WARNING = (
    "The parameters are not converged, try increasing the maximum number "
    "of iterations, decreasing the learning rate, or increasing n_chains"
)


class GradStop(Job):
    """Job to test the convergence."""

    atol: NumPositive | Tensor1D
    rtol: NumPositive | Tensor1D
    min_consecutive: IntStrictlyPositive
    betas: tuple[NumProbability, NumProbability]
    m: Tensor1D
    v: Tensor1D
    n_consecutive: IntPositive
    stopped: bool

    @beartype
    def __init__(
        self,
        atol: NumPositive | Tensor1D = 0.01,
        rtol: NumPositive | Tensor1D = 0.01,
        min_consecutive: IntStrictlyPositive = 20,
        betas: tuple[NumProbability, NumProbability] = (0.9, 0.999),
    ):
        self.atol = atol
        self.rtol = rtol
        self.min_consecutive = min_consecutive
        self.betas = betas

        if isinstance(atol, torch.Tensor) and (self.atol < 0).any():  # type: ignore
            raise ValueError(f"atol must be all positive, got {self.atol}")
        if isinstance(rtol, torch.Tensor) and (self.rtol < 0).any():  # type: ignore
            raise ValueError(f"rtol must be all positive, got {self.rtol}")

        self.n_consecutive = 0
        self.stopped = False

    def init(self, info: Info):
        if not hasattr(info, "optimizer"):
            raise ValueError("Optimizer must be set for GradStop")

        d = info.model.params_.numel
        self.m = torch.zeros(d)
        self.v = torch.zeros(d)

    def run(self, info: Info) -> bool | None:
        grads_list: list[torch.Tensor] = []
        for p in info.model.params_.as_list:
            if p.grad is None:
                grads_list.append(torch.zeros_like(p))
            else:
                grads_list.append(p.grad)

        grads = torch.cat([g.view(-1) for g in grads_list])
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * grads
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * grads**2

        m_hat = self.m / (1 - self.betas[0] ** (info.iteration + 1))
        v_hat = self.v / (1 - self.betas[1] ** (info.iteration + 1))

        # Check convergence
        if (m_hat.abs() <= self.atol + self.rtol * v_hat.sqrt()).all():
            self.n_consecutive += 1
            if self.n_consecutive >= self.min_consecutive:
                self.stopped = True
        else:
            self.n_consecutive = 0

        return self.stopped

    def end(self, info: Info, metrics: Metrics):
        if not self.stopped:
            warnings.warn(
                DEFAULT_NOT_CONVERGED_WARNING,
                stacklevel=2,
            )


class ValueStop(Job):
    """Job to test the convergence."""

    atol: NumPositive | Tensor1D
    rtol: NumPositive | Tensor1D
    min_consecutive: IntStrictlyPositive
    beta: NumProbability
    p: Tensor1D
    n_consecutive: IntPositive
    stopped: bool

    @beartype
    def __init__(
        self,
        atol: NumPositive | Tensor1D = 0.01,
        rtol: NumPositive | Tensor1D = 0.01,
        min_consecutive: IntStrictlyPositive = 20,
        beta: NumProbability = 0.9,
    ):
        self.atol = atol
        self.rtol = rtol
        self.min_consecutive = min_consecutive
        self.beta = beta

        if isinstance(atol, torch.Tensor) and (self.atol < 0).any():  # type: ignore
            raise ValueError(f"atol must be all positive, got {self.atol}")
        if isinstance(rtol, torch.Tensor) and (self.rtol < 0).any():  # type: ignore
            raise ValueError(f"rtol must be all positive, got {self.rtol}")

        self.n_consecutive = 0
        self.stopped = False

    def init(self, info: Info):
        self.p = torch.zeros(info.model.params_.numel)

    def run(self, info: Info) -> bool | None:
        params = info.model.params_.as_flat_tensor
        self.p = self.beta * self.p + (1 - self.beta) * params
        p_hat = self.p / (1 - self.beta ** (info.iteration + 1))

        # Check convergence
        if ((params - p_hat).abs() <= self.atol + self.rtol * p_hat.abs()).all():
            self.n_consecutive += 1
            if self.n_consecutive >= self.min_consecutive:
                self.stopped = True
        else:
            self.n_consecutive = 0

        return self.stopped

    def end(self, info: Info, metrics: Metrics):
        if not self.stopped:
            warnings.warn(
                DEFAULT_NOT_CONVERGED_WARNING,
                stacklevel=2,
            )
