from typing import Any, Callable, Final, NamedTuple, TypeAlias

import torch


# Named tuples
class BucketData(NamedTuple):
    idxs: torch.Tensor
    t0: torch.Tensor
    t1: torch.Tensor

# Type Aliases
RegressionFn: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor | None, torch.Tensor], torch.Tensor
]
LinkFn: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor | None, torch.Tensor], torch.Tensor
]
IndividualEffectsFn: TypeAlias = Callable[
    [torch.Tensor | None, torch.Tensor], torch.Tensor
]
BaseHazardFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Trajectory: TypeAlias = list[tuple[float, Any]]
ClockMethod: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Constants
ONE: Final = 1
TWO: Final = 2
THREE: Final = 3
LOGTWOPI: Final = torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.float32))
