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
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import torch
from pydantic import AfterValidator

from ._validators import is_col, is_ndim, is_pos, is_prob, is_strict_pos

if TYPE_CHECKING:
    from ..model._base import MultiStateJointModel
    from ..model._sampler import MetropolisHastingsSampler
    from ._data import ModelData
    from ._params import ModelParams


# Type Aliases
Num = int | float
Trajectory: TypeAlias = list[tuple[Num, Any]]


# Pydantic annotations
Tensor0D = Annotated[torch.Tensor, AfterValidator(is_ndim(0))]
Tensor1D = Annotated[torch.Tensor, AfterValidator(is_ndim(1))]
Tensor2D = Annotated[torch.Tensor, AfterValidator(is_ndim(2))]
Tensor3D = Annotated[torch.Tensor, AfterValidator(is_ndim(3))]
Tensor4D = Annotated[torch.Tensor, AfterValidator(is_ndim(4))]
TensorCol = Annotated[Tensor2D, AfterValidator(is_col)]
Tensor1DPositive = Annotated[Tensor1D, AfterValidator(is_pos)]
IntPositive = Annotated[int, AfterValidator(is_pos)]
IntStrictlyPositive = Annotated[int, AfterValidator(is_strict_pos)]
NumPositive = Annotated[Num, AfterValidator(is_pos)]
NumStrictlyPositive = Annotated[Num, AfterValidator(is_strict_pos)]
NumProbability = Annotated[Num, AfterValidator(is_prob)]


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
class HazardFns(NamedTuple):
    base_hazard_fn: BaseHazardFn
    link_fn: LinkFn


class MatRepr(NamedTuple):
    flat: Tensor1D
    dim: IntStrictlyPositive
    method: str


class BucketData(NamedTuple):
    idxs: Tensor1D
    t0: TensorCol
    t1: TensorCol


class TrajRepr(NamedTuple):
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
    logpdfs_fn: Callable[[ModelParams, Tensor3D], Tensor2D]
    logliks_fn: Callable[[ModelParams, Tensor3D], Tensor2D]
    iteration: int
    max_iterations: int
    model: MultiStateJointModel
    sampler: MetropolisHastingsSampler
    opt: torch.optim.Optimizer
    b: Tensor3D
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
    def init(self, info: Info) -> None:
        pass

    @abstractmethod
    def run(self, info: Info) -> bool | None:
        pass

    @abstractmethod
    def end(self, info: Info, metrics: Metrics) -> None:
        pass


# Constants
LOGTWOPI: Final[Tensor0D] = torch.log(torch.tensor(2.0 * torch.pi))
