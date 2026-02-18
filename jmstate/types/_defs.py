from abc import ABC, abstractmethod
from typing import Final, NamedTuple, Protocol, TypeAlias

import torch
from torch import nn

# Type Aliases
Trajectory: TypeAlias = list[tuple[float, str]]


# Constants
LOG_TWO_PI: Final[torch.Tensor] = torch.log(torch.tensor(2.0 * torch.pi))


# User definitions
class LogBaseHazardFn(ABC, nn.Module):
    r"""Abstract base class for log base hazard functions.

    This class represents a log-transformed baseline hazard function in a multistate
    model. The log base hazard is parameterized as a `torch.nn.Module`, allowing its
    parameters to be optimized during model fitting. For default base hazards, a
    `frozen` attribute can be set to prevent optimization of the module parameters.

    The function expects:

    - `t0`: a column vector of previous transition times of shape :math:`(n, 1)`.
    - `t1`: a matrix of future time points at which the log base hazard is evaluated,
        of shape :math:`(n, m)` matching the number of rows in `t0`.

    Attributes:
        frozen (bool): If True, the parameters of the log base hazard are not updated
            during optimization.

    Methods:
        forward(t0, t1): Computes the log base hazard between `t0` and `t1`.

    Notes:
        The outputs are in log scale and can be directly used in likelihood
        computations for multistate models.
    """

    @abstractmethod
    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        r"""Compute the log base hazard values.

        Args:
            t0 (torch.Tensor): Previous transition times, shape :math:`(n, 1)`.
            t1 (torch.Tensor): Future evaluation times, shape :math:`(n, m)`.

        Returns:
            torch.Tensor: Log base hazard values evaluated at `t1` relative to `t0`,
            shape :math:`(n, m)`.
        """
        ...


class IndividualParametersFn(Protocol):
    r"""Protocol defining the individual parameters function.

    This function maps population-level parameters, covariates, and random effects
    to individual-specific parameters used in a multistate model.

    Given inputs:

    - `pop_params`: population-level parameters.
    - `x`: covariates matrix of shape :math:`(n, p)`.
    - `b`: random effects of shape either :math:`(n, q)` (2D) or :math:`(n, m, q)` (3D).

    The function must return individual parameters `indiv_params` of shape 2D or 3D
    consistent with the input dimensions.

    Args:
        pop_params (torch.Tensor): Population-level parameters.
        x (torch.Tensor): Fixed covariates matrix.
        b (torch.Tensor): Random effects tensor.

    Returns:
        torch.Tensor: Individual parameters tensor of appropriate shape for each
        individual.

    Examples:
        >>> indiv_params_fn = lambda pop, x, b: pop + b
    """

    def __call__(
        self, pop_params: torch.Tensor, x: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor: ...


class RegressionFn(Protocol):
    r"""Protocol defining a regression function for multistate models.

    This function maps evaluation times and individual-specific parameters to predicted
    response values. It must support both 1D and 2D time inputs and individual
    parameters of order 2 or 3, returning either 3D or 4D tensors depending on the
    model design.

    Tensor output conventions:

    - Last dimension corresponds to the response variable dimension :math:`d`.
    - Second-last dimension corresponds to repeated measurements :math:`m`.
    - Third-last dimension corresponds to individual index :math:`n`.
    - Optional fourth-last dimension may be used for parallelization across MCMC
      chains or batch processing.

    This protocol is conceptually identical to `LinkFn`.

    Args:
        t (torch.Tensor): Evaluation times of shape :math:`(n, m)` or :math:`(m,)`.
        indiv_params (torch.Tensor): Individual parameters of shape 2D :math:`(n, q)`
            or 3D :math:`(n, m, q)`.

    Returns:
        torch.Tensor: Predicted response values of shape consistent with `(n, m, d)` or
            `(batch, n, m, d)` for parallelized computations.

    Examples:
        >>> def sigmoid(t: torch.Tensor, indiv_params: torch.Tensor):
        ...     scale, offset, slope = indiv_params.chunk(3, dim=-1)
        ...     # Fully broadcasted computation
        ...     return (scale * torch.sigmoid((t - offset) / slope)).unsqueeze(-1)
    """

    def __call__(self, t: torch.Tensor, indiv_params: torch.Tensor) -> torch.Tensor: ...


class LinkFn(Protocol):
    r"""Protocol defining a link function for multistate models.

    A link function maps evaluation times and individual-specific parameters to
    transformed outputs, such as transition-specific parameters. Requirements are
    identical to those of `RegressionFn`.

    Tensor output conventions:

    - Last dimension corresponds to the response variable dimension :math:`d`.
    - Second-last dimension corresponds to repeated measurements :math:`m`.
    - Third-last dimension corresponds to individual index :math:`n`.
    - Optional fourth-last dimension may be used for parallelization across MCMC
      chains or batch computations.

    This protocol is conceptually identical to `RegressionFn`.

    Args:
        t (torch.Tensor): Evaluation times of shape :math:`(n, m)` or :math:`(m,)`.
        indiv_params (torch.Tensor): Individual parameters of shape 2D :math:`(n, q)`
            or 3D :math:`(n, m, q)`.

    Returns:
        torch.Tensor: Transformed outputs consistent with shapes `(n, m, d)` or
            `(batch, n, m, d)` for parallelized computations.

    Examples:
        >>> def sigmoid(t: torch.Tensor, indiv_params: torch.Tensor):
        ...     scale, offset, slope = indiv_params.chunk(3, dim=-1)
        ...     # Fully broadcasted computation
        ...     return (scale * torch.sigmoid((t - offset) / slope)).unsqueeze(-1)
    """

    def __call__(self, t: torch.Tensor, indiv_params: torch.Tensor) -> torch.Tensor: ...


# Named tuples
class BucketData(NamedTuple):
    r"""NamedTuple representing a set of transitions for visualization purposes.

    This structure stores the transition times of individuals grouped together,
    typically used to visualize the trajectories per transition type in multistate
    models. Each entry corresponds to a single transition for a specific individual.

    Attributes:
        idxs (torch.Tensor): Indices of individuals corresponding to the transitions in
            this bucket, shape :math:`(k,)`, where :math:`k` is the number of
            transitions in the bucket.
        t0 (torch.Tensor): Column vector of previous transition times, shape
            :math:`(k, 1)`. Represents the start time of each transition.
        t1 (torch.Tensor): Column vector of next transition times, shape
            :math:`(k, 1)`. Represents the end time of each transition.

    Notes:
        Each tensor is aligned such that `t0[i]` and `t1[i]` correspond to the
        transition of individual `idxs[i]`. This alignment facilitates plotting or
        analyzing transitions per type across individuals.
    """

    idxs: torch.Tensor
    t0: torch.Tensor
    t1: torch.Tensor
