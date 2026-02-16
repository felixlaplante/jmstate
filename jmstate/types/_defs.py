from abc import ABC, abstractmethod
from typing import Final, NamedTuple, Protocol, TypeAlias

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from xxhash import xxh3_64_intdigest

# Type Aliases
Trajectory: TypeAlias = list[tuple[float, str]]


# Constants
LOG_TWO_PI: Final[torch.Tensor] = torch.log(torch.tensor(2.0 * torch.pi))


# Protocols
class IndividualEffectsFn(Protocol):
    """The individual effects function protocol.

    Calls the individual effects function.

    It must be able to yield 2D or 3D tensors given inputs of `gamma` (population
    parameters), `x` (covariates matrix), and `b` (random effects). Note `b` is
    either 2D or 3D.

    Args:
        gamma (torch.Tensor | None): The population parameters.
        x (torch.Tensor | None): The fixed covariates matrix.
        b (torch.Tensor): The random effects.

    Returns:
        torch.Tensor: The individual parameters `psi`.

    Examples:
        >>> individual_effects_fn = lambda gamma, x, b: gamma + b
    """

    def __call__(
        self, gamma: torch.Tensor | None, x: torch.Tensor | None, b: torch.Tensor
    ) -> torch.Tensor: ...


class RegressionFn(Protocol):
    """The regression function protocol.

    It must be able to yield 3D and 4D tensors given 1D or 2D time inputs, as well
        as `psi` input of order 2 or 3. This is not very restrictive, but requires to be
        careful. The last dimension is the dimension of the response variable; second
        last is the repeated measurements; third last is individual based; possible
        fourth last is for parallelization of the MCMC sampler.

        It is identical to LinkFn.

    Args:
        t (torch.Tensor): The evaluation times.
        psi (torch.Tensor): The individual parameters.

    Returns:
        torch.Tensor: The response variable values.

    Examples:
        >>> def sigmoid(t: torch.Tensor, psi: torch.Tensor):
        ...     scale, offset, slope = psi.chunk(3, dim=-1)
        ...     # Fully broadcasted
        ...     return (scale * torch.sigmoid((t - offset) / slope)).unsqueeze(-1)
        >>> regression_fn = sigmoid
    """

    def __call__(self, t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor: ...


class LinkFn(Protocol):
    """The link function protocol.

    It must be able to yield 3D and 4D tensors given 1D or 2D time inputs, as well
        as `psi` input of order 2 or 3. This is not very restrictive, but requires to be
        careful. The last dimension is the dimension of the response variable; second
        last is the repeated measurements; third last is individual based; possible
        fourth last is for parallelization of the MCMC sampler.

        It is identical to RegressionFn.

    Args:
        t (torch.Tensor): The evaluation times.
        psi (torch.Tensor): The individual parameters.

    Returns:
        torch.Tensor: The response variable values.

    Examples:
        >>> def sigmoid(t: torch.Tensor, psi: torch.Tensor):
        ...     scale, offset, slope = psi.chunk(3, dim=-1)
        ...     # Fully broadcasted
        ...     return (scale * torch.sigmoid((t - offset) / slope)).unsqueeze(-1)
        >>> link_fn = sigmoid
    """

    def __call__(self, t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor: ...


class ClockMethod(Protocol):
    r"""The clock method protocol.

    This protocol is useful to differentiate between two natural mappings when dealing
    with base hazards. It expects a former transition time column vector `t0` as well as
    a matrix of next time points `t1`. `t1` is a matrix with the same number of rows as
    `t0`.

    .. math::
        (t_0, t_1) \mapsto \begin{cases} t_1 - t_0 \text{(clock reset)}, \\ t_1
        \text{(clock forward)} \end{cases}.
    """

    def __call__(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor: ...


# Abstract classes
class LogBaseHazardFn(nn.Module, ABC):
    """The log base hazard base class.

    This is not a protocol because caching is done, and therefore a key is required.
    Making this a `nn.Module`, one can check the value of the parameters and store their
    hashed values.

    Note that the log base hazard function is in log scale, and expects a former
    transition time column vector `t0` as well as next times points at which the log
    base hazard is to be computed. `t1` is a matrix with the same number of rows as
    `t0`.

    Implement a `forward` and do not forget the `super().__init__()` when declaring your
    own class.

    Pass the parameters you want to optimize in the `ModelParams.extra` attribute as a
    list. If you do not want them to be optimized, then by default they do not require
    gradients for the given implementations.
    """

    @property
    def key(self) -> tuple[int, ...]:
        """Returns a hash containing the class type and parameters if there are any.

        Returns:
            tuple[int, ...]: A key used in caching operations.
        """
        ident = id(self)
        try:
            vector = parameters_to_vector(self.parameters())
            return (ident, xxh3_64_intdigest(vector.detach().numpy()))
        except ValueError:
            return (ident,)

    @abstractmethod
    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor: ...


# Named tuples
class BucketData(NamedTuple):
    """A simple `NamedTuple` containing transition information.

    Attributes:
        idxs (torch.Tensor): The individual indices.
        t0 (torch.Tensor): A column vector of previous transition times.
        t1 (torch.Tensor): A column vecotr of next transition times.
    """

    idxs: torch.Tensor
    t0: torch.Tensor
    t1: torch.Tensor
