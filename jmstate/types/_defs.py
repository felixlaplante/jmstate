from typing import Any, Final, NamedTuple, Protocol, TypeAlias

import torch


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


# Type Aliases
Trajectory: TypeAlias = list[tuple[float, Any]]


# Constants
LOGTWOPI: Final = torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.float32))
GAMMA_POS: Final = 0
Q_POS: Final = 1
R_POS: Final = 2
ALPHAS_POS: Final = 3
BETAS_POS: Final = 4
