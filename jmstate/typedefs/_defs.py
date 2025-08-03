from __future__ import annotations

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    NamedTuple,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import torch
from beartype.vale import Is

if TYPE_CHECKING:
    from ..model._base import MultiStateJointModel
    from ..model._sampler import MetropolisHastingsSampler
    from ._data import ModelData
    from ._params import ModelParams


# Beartype checks
Tensor0D = Annotated[torch.Tensor, Is[lambda t: t.ndim == 0]]  # type: ignore
Tensor1D = Annotated[torch.Tensor, Is[lambda t: t.ndim == 1]]  # type: ignore
Tensor2D = Annotated[torch.Tensor, Is[lambda t: t.ndim == 2]]  # type: ignore
Tensor3D = Annotated[torch.Tensor, Is[lambda t: t.ndim == 3]]  # type: ignore
Tensor4D = Annotated[torch.Tensor, Is[lambda t: t.ndim == 4]]  # type: ignore
TensorRow = Annotated[torch.Tensor, Is[lambda t: t.ndim == 2 and t.size(0) == 1]]  # type: ignore
TensorCol = Annotated[torch.Tensor, Is[lambda t: t.ndim == 2 and t.size(1) == 1]]  # type: ignore


# Type Aliases
Trajectory: TypeAlias = list[tuple[float, Any]]


# Protocols
@runtime_checkable
class RegressionFn(Protocol):
    def __call__(
        self, t: Tensor1D | Tensor2D, psi: Tensor2D | Tensor3D
    ) -> Tensor3D | Tensor4D: ...


@runtime_checkable
class LinkFn(Protocol):
    def __call__(
        self, t: Tensor1D | Tensor2D, psi: Tensor2D | Tensor3D
    ) -> Tensor3D | Tensor4D: ...


@runtime_checkable
class IndividualEffectsFn(Protocol):
    def __call__(
        self, gamma: torch.Tensor | None, x: Tensor2D | None, b: Tensor2D | Tensor3D
    ) -> Tensor2D | Tensor3D: ...


@runtime_checkable
class BaseHazardFn(Protocol):
    def __call__(self, t0: TensorCol, t1: Tensor2D) -> Tensor2D: ...


@runtime_checkable
class ClockMethod(Protocol):
    def __call__(self, t0: TensorCol, t1: Tensor2D) -> Tensor2D: ...


# Named tuples
class BucketData(NamedTuple):
    idxs: Tensor1D
    t0: TensorCol
    t1: TensorCol


class VecRep(NamedTuple):
    idxs: Tensor1D
    t0: TensorCol
    t1: TensorCol
    obs: Tensor1D


class HazardInfo(NamedTuple):
    t0: TensorCol
    t1: Tensor2D
    x: Tensor2D | None
    psi: Tensor2D | Tensor3D
    alpha: Tensor1D
    beta: Tensor1D | None
    base_hazard_fn: BaseHazardFn
    link_fn: LinkFn


# SimpleNamespaces
class Info(SimpleNamespace):
    data: ModelData
    iteration: int
    n_iterations: int
    model: MultiStateJointModel
    sampler: MetropolisHastingsSampler
    optimizer: torch.optim.Optimizer
    b: Tensor3D
    logpdfs: Tensor2D
    logliks: Tensor2D
    psi: Tensor3D


class Metrics(SimpleNamespace):
    fim: Tensor2D
    ebes: Tensor2D
    loglik: float
    nloglik_pen: float
    aic: float
    bic: float
    pred_y: list[Tensor3D]
    pred_surv_logps: list[Tensor2D]
    pred_trajectories: list[Trajectory]
    params_history: list[ModelParams]
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
DEFAULT_OPT_KWARGS: Final = {"lr": 0.1, "fused": True}
