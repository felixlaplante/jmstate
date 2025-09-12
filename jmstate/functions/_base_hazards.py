from typing import cast

import torch
from pydantic import ConfigDict, validate_call
from torch import nn

from ..typedefs._defs import (
    LOG_TWO_PI,
    BaseHazardFn,
    ClockMethod,
    Num,
    NumStrictlyPositive,
)
from ..utils._dtype import get_dtype


def clock_forward(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    r"""Time transformation for clock forward method.

    This is the simple mapping:

    .. math::
        (t_0, t_1) \mapsto t_1.

    This type of mapping is particularly useful when the base risk does not depend on
    relative (sojourn) type, but rather on absolute time.

    Args:
        t0 (torch.Tensor): Past transition time.
        t1 (torch.Tensor): Current time

    Returns:
        torch.Tensor: Current time.

    Examples:
        >>> clock_forward(1, 2)
        2
    """
    return t1


def clock_reset(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    r"""Time transformation for clock reset method.

    This is the simple mapping:

    .. math::
        (t_0, t_1) \mapsto t_1 - t_0.

    This type of mapping is particularly useful when the base risk depends on
    relative (sojourn) type, but not on absolute time.

    Args:
        t0 (torch.Tensor): Past transition time.
        t1 (torch.Tensor): Current time

    Returns:
        torch.Tensor: Current time - Past transition time.

    Examples:
        >>> clock_reset(1, 2)
        1
    """
    return t1 - t0


class Exponential(BaseHazardFn):
    r"""Implements the Exponential base hazard.

    Exponential base hazard is time independent.
    It is given by the formula:

    .. math::
        \lambda(t) = \lambda.

    This returns the base hazard in log scale.$

    Attributes:
        log_lmda (nn.Parameter): The log rate factor.
    """

    log_lmda: nn.Parameter

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, lmda: NumStrictlyPositive):
        """Initializes the Exponential hazard.

        Args:
            lmda (NumStrictlyPositive): The rate factor.

        Raises:
            ValueError: If lmda is not strictly positive.
        """
        super().__init__()  # type: ignore

        lmda_ = torch.tensor(lmda, dtype=get_dtype())
        self.log_lmda = nn.Parameter(torch.log(lmda_), requires_grad=False)

    def forward(
        self,
        t0: torch.Tensor,  # noqa: ARG002
        t1: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        """Calls the Exponential base hazard.

        Args:
            t0 (torch.Tensor): Past transition time.
            t1 (torch.Tensor): Current time

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        return self.log_lmda

    @property
    def lmda(self) -> torch.Tensor:
        """Gets the rate factor.

        Returns:
            torch.Tensor: The rate factor.
        """
        return self.log_lmda.exp()


class Weibull(BaseHazardFn):
    r"""Implements the Weibull base hazard.

    Weibull base hazard is time dependent.
    It is given by the formula:

    .. math::
        \lambda(t) = \frac{k}{\lambda} \frac{x}{\lambda}^{k - 1}.

    This returns the base hazard in log scale.

    Attributes:
        clock_method (ClockMethod): The ClockMethod transformation.
        log_k (nn.Parameter): The log of the shape parameter.
        log_lmda (nn.Parameter): The log of the scale parameter.
    """

    clock_method: ClockMethod
    log_k: nn.Parameter
    log_lmda: nn.Parameter

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        k: NumStrictlyPositive,
        lmda: NumStrictlyPositive,
        clock_method: ClockMethod = clock_reset,
    ):
        """Initializes the Weibull base hazard.

        Args:
            k (NumStrictlyPositive): The shape parameter.
            lmda (NumStrictlyPositive): The scale parameter.
            clock_method (ClockMethod, optional): The ClockMethod transformation.
                Defaults to clock_reset.

        Raises:
            ValueError: If k is not strictly positive.
            ValueError: If lmda is not strictly positive.
        """
        super().__init__()  # type: ignore

        k_ = torch.tensor(k, dtype=get_dtype())
        lmda_ = torch.tensor(lmda, dtype=get_dtype())
        self.clock_method = clock_method
        self.log_k = nn.Parameter(torch.log(k_), requires_grad=False)
        self.log_lmda = nn.Parameter(torch.log(lmda_), requires_grad=False)

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the Weibull base hazard.

        Args:
            t0 (torch.Tensor): Past transition time.
            t1 (torch.Tensor): Current time

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        t = self.clock_method(t0, t1)
        log_t = torch.log(t)
        return self.log_k - self.log_lmda + (self.k - 1) * (log_t - self.log_lmda)

    @property
    def k(self) -> torch.Tensor:
        """Gets the shape parameter.

        Returns:
            torch.Tensor: The shape parameter.
        """
        return self.log_k.exp()

    @property
    def lmda(self) -> torch.Tensor:
        """Gets the scale parameter.

        Returns:
            torch.Tensor: The scale parameter.
        """
        return self.log_lmda.exp()


class Gompertz(BaseHazardFn):
    r"""Implements the Gompertz base hazard.

    Gompertz base hazard is time dependent.
    It is given by the formula:

    .. math::
        \lambda(t) = a \exp{bt}.

    This returns the base hazard in log scale.

    Attributes:
        b (nn.Parameter): The shape parameter.
        clock_method (ClockMethod): The ClockMethod transformation.
        log_a (nn.Parameter): The baseline hazard parameter.
    """

    b: nn.Parameter
    clock_method: ClockMethod
    log_a: nn.Parameter

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        a: NumStrictlyPositive,
        b: NumStrictlyPositive,
        clock_method: ClockMethod = clock_reset,
    ):
        """Initializes the Gompertz base hazard.

        Raises:
            ValueError: If a is not strictly positive.

        Args:
            a (NumStrictlyPositive): The baseline hazard.
            b (Num): The shape parameter.
            clock_method (ClockMethod, optional): The ClockMethod transformation.
                Defaults to clock_reset.
        """
        super().__init__()  # type: ignore

        dtype = get_dtype()
        a_ = torch.tensor(a, dtype=dtype)
        self.b = nn.Parameter(torch.tensor(b, dtype=dtype), requires_grad=False)
        self.clock_method = clock_method
        self.log_a = nn.Parameter(torch.log(a_), requires_grad=False)

    def __call__(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the Gompertz base hazard.

        Args:
            t0 (torch.Tensor): Past transition time.
            t1 (torch.Tensor): Current time

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        t = self.clock_method(t0, t1)
        return self.log_a + self.b * t

    @property
    def a(self) -> torch.Tensor:
        """Gets the baseline hazard.

        Returns:
            torch.Tensor: The baseline hazard.
        """
        return self.log_a.exp()


class LogNormal(BaseHazardFn):
    r"""Implements the log normal base hazard.

    Log normal base hazard is time dependent.
    It is given by the formula :

    .. math::
        \lambda(t) = \frac{\phi\left( \frac{\log t - \mu}{\sigma} \right)}{t \sigma
        \, \Phi\left( -\frac{\log t - \mu}{\sigma} \right)},
        \quad t > 0,

    where:

    .. math::
        \phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}, \quad
        \Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^z e^{-t^2/2} \, dt.

    This returns the base hazard in log scale.

    Attributes:
        mu (nn.Parameter): The log time mean.
        clock_method (ClockMethod): The ClockMethod transformation.
        log_scale (nn.Parameter): The log of scale.
    """

    mu: nn.Parameter
    clock_method: ClockMethod
    log_scale: nn.Parameter

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        mu: Num,
        scale: NumStrictlyPositive,
        clock_method: ClockMethod = clock_reset,
    ):
        """Initializes the log normal base hazard.

        Args:
            mu (Num): The log time mean.
            scale (NumStrictlyPositive): The log time scale.
            clock_method (ClockMethod, optional): The ClockMethod transformation.
                Defaults to clock_reset.

        Raises:
            ValueError: If scale is not strictly positive.

        Returns:
            BaseHazardFn: Returns the log normal base hazard function.
        """
        super().__init__()  # type: ignore

        dtype = get_dtype()
        self.mu = nn.Parameter(torch.tensor(mu, dtype=dtype), requires_grad=False)
        scale_ = torch.tensor(scale, dtype=dtype)
        self.clock_method = clock_method
        self.log_scale = nn.Parameter(torch.log(scale_), requires_grad=False)

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the log normal base hazard.

        Args:
            t0 (torch.Tensor): Past transition time.
            t1 (torch.Tensor): Current time

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        t = self.clock_method(t0, t1)
        log_t = torch.log(t)
        z = (log_t - self.mu) / self.scale
        log_pdf = -log_t - self.log_scale - 0.5 * LOG_TWO_PI - 0.5 * z**2
        log_sf = cast(torch.Tensor, torch.special.log_ndtr(-z))  # type: ignore
        return log_pdf - log_sf

    @property
    def scale(self) -> torch.Tensor:
        """Gets the scale.

        Returns:
            torch.Tensor: The scale.
        """
        return self.log_scale.exp()
