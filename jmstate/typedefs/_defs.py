from __future__ import annotations

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Final,
    NamedTuple,
    TypeAlias,
)

import torch
from beartype.vale import Is

if TYPE_CHECKING:
    from ..model._base import MultiStateJointModel
    from ..model._sampler import MetropolisHastingsSampler
    from ._data import ModelData

# Beartype checks
Tensor0D = Annotated[torch.Tensor, Is[lambda t: t.ndim == 0]]  # type: ignore
Tensor1D = Annotated[torch.Tensor, Is[lambda t: t.ndim == 1]]  # type: ignore
Tensor2D = Annotated[torch.Tensor, Is[lambda t: t.ndim == 2]]  # type: ignore
Tensor3D = Annotated[torch.Tensor, Is[lambda t: t.ndim == 3]]  # type: ignore


# Type Aliases
Trajectory: TypeAlias = list[tuple[float, Any]]
RegressionFn: TypeAlias = Callable[[Tensor1D | Tensor2D, Tensor2D], Tensor3D]
LinkFn: TypeAlias = Callable[[Tensor1D | Tensor2D, Tensor2D], Tensor3D]
IndividualEffectsFn: TypeAlias = Callable[
    [torch.Tensor | None, Tensor2D | None, Tensor2D], Tensor2D
]
BaseHazardFn: TypeAlias = Callable[
    [Tensor1D | Tensor2D, Tensor1D | Tensor2D], Tensor1D | Tensor2D
]
ClockMethod: TypeAlias = Callable[
    [Tensor1D | Tensor2D, Tensor1D | Tensor2D], Tensor1D | Tensor2D
]


# Named tuples
class BucketData(NamedTuple):
    idxs: torch.Tensor
    t0: torch.Tensor
    t1: torch.Tensor


# SimpleNamespaces
class Info(SimpleNamespace):
    data: ModelData
    iteration: int
    n_iterations: int
    n_chains: int
    model: MultiStateJointModel
    sampler: MetropolisHastingsSampler
    optimizer: torch.optim.Optimizer
    b: Tensor2D
    logpdfs: torch.Tensor
    logliks: torch.Tensor


class Metrics(SimpleNamespace):
    fim: Tensor2D
    ebes: Tensor2D
    loglik: float
    nloglik_pen: float
    aic: float
    bic: float
    pred_y: list[Tensor3D]
    pred_surv_logps: list[Tensor2D]
    pred_trajectories: list[list[Trajectory]]
    params_history: list[Tensor1D]
    mcmc_diagnostics: list[dict[str, Any]]


# Abstract
class Job(ABC):
    @abstractmethod
    def init(self, info: Info, metrics: Metrics) -> None:
        pass

    @abstractmethod
    def run(self, info: Info, metrics: Metrics) -> None:
        pass

    @abstractmethod
    def end(self, info: Info, metrics: Metrics) -> None:
        pass


# Constants
LOGTWOPI: Final = torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.float32))
