import warnings

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import (
    Info,
    IntStrictlyPositive,
    Job,
    Metrics,
    NumPositive,
    NumProbability,
    Tensor1DPositive,
)
from ..utils._checks import check_consistent_size

# Constants
DEFAULT_NOT_CONVERGED_WARNING = (
    "The parameters are not converged, try increasing the maximum number "
    "of iterations, decreasing the learning rate, or increasing n_chains"
)


class GradStop(Job):
    """Job to test the convergence."""

    atol: int | float | torch.Tensor
    rtol: int | float | torch.Tensor
    min_consecutive: int
    betas: tuple[int | float, int | float]
    m: torch.Tensor
    v: torch.Tensor
    n_consecutive: int
    stopped: bool

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        atol: NumPositive | Tensor1DPositive = 0.01,
        rtol: NumPositive | Tensor1DPositive = 0.01,
        min_consecutive: IntStrictlyPositive = 20,
        betas: tuple[NumProbability, NumProbability] = (0.9, 0.999),
    ):
        """Initializes the ValueStop class.

        Args:
            atol (NumPositive | Tensor1DPositive, optional): Absolute tolerance, either
                scalar or elementwise. Defaults to 0.01.
            rtol (NumPositive | Tensor1DPositive, optional): Relative tolerance, either
                scalar or elementwise. Defaults to 0.01.
            min_consecutive (IntStrictlyPositive, optional): The minimum consecutive
                iterations with grad difference less than tolerance. Defaults to 20.
            beta (tuple[NumProbability, NumProbability], optional): Exponential moving
                averages forget parameters. Defaults to (0.9, 0.999).
        """
        self.atol = atol
        self.rtol = rtol
        self.min_consecutive = min_consecutive
        self.betas = betas

        self.n_consecutive = 0
        self.stopped = False

    def init(self, info: Info):
        if not hasattr(info, "optimizer"):
            raise ValueError("Optimizer must be set for GradStop")

        d = info.model.params_.numel
        if isinstance(self.atol, torch.Tensor):
            check_consistent_size(((self.atol, 0, "atol"), (d, None, "params.numel")))
        if isinstance(self.rtol, torch.Tensor):
            check_consistent_size(((self.rtol, 0, "rtol"), (d, None, "params.numel")))

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

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        if not self.stopped:
            warnings.warn(
                DEFAULT_NOT_CONVERGED_WARNING,
                stacklevel=2,
            )


class ValueStop(Job):
    """Job to test the convergence."""

    atol: int | float | torch.Tensor
    rtol: int | float | torch.Tensor
    min_consecutive: int
    beta: int | float
    p: torch.Tensor
    n_consecutive: int
    stopped: bool

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        atol: NumPositive | Tensor1DPositive = 0.01,
        rtol: NumPositive | Tensor1DPositive = 0.01,
        min_consecutive: IntStrictlyPositive = 20,
        beta: NumProbability = 0.9,
    ):
        """Initializes the ValueStop class.

        Args:
            atol (NumPositive | Tensor1DPositive, optional): Absolute tolerance, either
                scalar or elementwise. Defaults to 0.01.
            rtol (NumPositive | Tensor1DPositive, optional): Relative tolerance, either
                scalar or elementwise. Defaults to 0.01.
            min_consecutive (IntStrictlyPositive, optional): The minimum consecutive
                iterations with value difference less than tolerance. Defaults to 20.
            beta (NumProbability, optional): Exponential moving average forget
                parameter. Defaults to 0.9.
        """
        self.atol = atol
        self.rtol = rtol
        self.min_consecutive = min_consecutive
        self.beta = beta

        self.n_consecutive = 0
        self.stopped = False

    def init(self, info: Info):
        d = info.model.params_.numel
        if isinstance(self.atol, torch.Tensor):
            check_consistent_size(((self.atol, 0, "atol"), (d, None, "params.numel")))
        if isinstance(self.rtol, torch.Tensor):
            check_consistent_size(((self.rtol, 0, "rtol"), (d, None, "params.numel")))

        self.p = torch.zeros(d)

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

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        if not self.stopped:
            warnings.warn(
                DEFAULT_NOT_CONVERGED_WARNING,
                stacklevel=2,
            )
