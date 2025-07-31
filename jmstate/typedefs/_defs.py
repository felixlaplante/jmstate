from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol, TypeAlias

import torch

if TYPE_CHECKING:
    from ..model._base import MultiStateJointModel
    from ..model._sampler import MetropolisHastingsSampler
    from ._data import ModelData

# Type Aliases
Trajectory: TypeAlias = list[tuple[float, Any]]


# Named tuples
class BucketData(NamedTuple):
    idxs: torch.Tensor
    t0: torch.Tensor
    t1: torch.Tensor


# Protocols
class RegressionFn(Protocol):
    def __call__(self, t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor: ...


class LinkFn(Protocol):
    def __call__(self, t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor: ...


class IndividualEffectsFn(Protocol):
    def __call__(
        self, gamma: torch.Tensor | None, x: torch.Tensor | None, b: torch.Tensor
    ) -> torch.Tensor: ...


class BaseHazardFn(Protocol):
    def __call__(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor: ...


class ClockMethod(Protocol):
    def __call__(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor: ...


# SimpleNamespaces
class Info(SimpleNamespace):
    data: ModelData
    iteration: int
    n_iterations: int
    batch_size: int
    model: MultiStateJointModel
    sampler: MetropolisHastingsSampler
    optimizer: torch.optim.Optimizer
    b: torch.Tensor
    logpdfs: torch.Tensor
    logliks: torch.Tensor


class Metrics(SimpleNamespace):
    fim: torch.Tensor
    ebes: torch.Tensor
    loglik: float
    nloglik_pen: float
    aic: float
    bic: float
    pred_y: list[torch.Tensor]
    pred_surv_logps: list[torch.Tensor]
    pred_trajectories: list[list[Trajectory]]
    params_history: list[torch.Tensor]
    mcmc_diagnostics: list[dict[str, Any]]


# Dataclasses
@dataclass
class ModelDesign:
    """Class containing all multistate joint model design."""

    individual_effects_fn: IndividualEffectsFn
    regression_fn: RegressionFn
    surv: dict[
        tuple[int, int],
        tuple[
            BaseHazardFn,
            LinkFn,
        ],
    ]


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
GAMMA_POS: Final = 0
Q_POS: Final = 1
R_POS: Final = 2
ALPHAS_POS: Final = 3
BETAS_POS: Final = 4
