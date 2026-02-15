"""Base hazard functions."""

__all__ = [
    "Exponential",
    "Gompertz",
    "LogNormal",
    "Weibull",
    "clock_forward",
    "clock_reset",
]

from numbers import Real
from typing import cast

import torch
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from torch import nn

from ..typedefs._defs import LOG_TWO_PI, ClockMethod, LogBaseHazardFn


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


class Exponential(LogBaseHazardFn):
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

    @validate_params(
        {
            "lmda": [Interval(Real, 0, None, closed="neither")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, lmda: float):
        """Initializes the Exponential hazard.

        Args:
            lmda (float): The rate factor.
        """
        super().__init__()  # type: ignore

        lmda_ = torch.tensor(lmda)
        self.log_lmda = nn.Parameter(torch.log(lmda_))

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


class Weibull(LogBaseHazardFn):
    r"""Implements the Weibull base hazard.

    Weibull base hazard is time dependent.

    It is given by the formula:

    .. math::
        \lambda(t) = \frac{k}{\lambda} \frac{t}{\lambda}^{k - 1}.

    This returns the base hazard in log scale.

    Attributes:
        clock_method (ClockMethod): The ClockMethod transformation.
        log_k (nn.Parameter): The log of the shape parameter.
        log_lmda (nn.Parameter): The log of the scale parameter.
    """

    clock_method: ClockMethod
    log_lmda: nn.Parameter
    log_k: nn.Parameter

    @validate_params(
        {
            "lmda": [Interval(Real, 0, None, closed="neither")],
            "k": [Interval(Real, 0, None, closed="neither")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, lmda: float, k: float, clock_method: ClockMethod = clock_reset):
        """Initializes the Weibull base hazard.

        Args:
            lmda (float): The scale parameter.
            k (float): The shape parameter.
            clock_method (ClockMethod, optional): The ClockMethod transformation.
                Defaults to clock_reset.
        """
        super().__init__()  # type: ignore

        k_ = torch.tensor(k)
        lmda_ = torch.tensor(lmda)
        self.clock_method = clock_method
        self.log_k = nn.Parameter(torch.log(k_))
        self.log_lmda = nn.Parameter(torch.log(lmda_))

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


class Gompertz(LogBaseHazardFn):
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

    @validate_params(
        {
            "a": [Interval(Real, 0, None, closed="neither")],
            "b": [Interval(Real, None, None, closed="neither")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, a: float, b: float, clock_method: ClockMethod = clock_reset):
        """Initializes the Gompertz base hazard.

        Args:
            a (float): The baseline hazard.
            b (float): The shape parameter.
            clock_method (ClockMethod, optional): The ClockMethod transformation.
                Defaults to clock_reset.
        """
        super().__init__()  # type: ignore

        a_ = torch.tensor(a)
        self.b = nn.Parameter(torch.tensor(b))
        self.clock_method = clock_method
        self.log_a = nn.Parameter(torch.log(a_))

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


class LogNormal(LogBaseHazardFn):
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

    @validate_params(
        {
            "mu": [Interval(Real, None, None, closed="neither")],
            "scale": [Interval(Real, 0, None, closed="neither")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self, mu: float, scale: float, clock_method: ClockMethod = clock_reset
    ):
        """Initializes the log normal base hazard.

        Args:
            mu (float): The log time mean.
            scale (float): The log time scale.
            clock_method (ClockMethod, optional): The ClockMethod transformation.
                Defaults to clock_reset.

        Returns:
            LogBaseHazardFn: Returns the log normal base hazard function.
        """
        super().__init__()  # type: ignore

        self.mu = nn.Parameter(torch.tensor(mu))
        scale_ = torch.tensor(scale)
        self.clock_method = clock_method
        self.log_scale = nn.Parameter(torch.log(scale_))

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
