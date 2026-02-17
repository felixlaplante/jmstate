"""Base hazard functions."""

__all__ = ["Exponential", "Gompertz", "LogNormal", "Weibull"]

from numbers import Real
from typing import cast

import torch
from sklearn.utils._param_validation import (  # type: ignore
    Interval,
    StrOptions,
    validate_params,  # type: ignore
)
from torch import nn

from ..types._defs import LOG_TWO_PI, LogBaseHazardFn


class Exponential(LogBaseHazardFn):
    r"""Implements the Exponential base hazard.

    Exponential base hazard is time independent.

    It is given by the formula

    .. math::
        \lambda(t) = \lambda.

    This returns the base hazard in log scale. It expects a former transition time
    column vector `t0` as well as a matrix of next time points `t1`. `t1` is a matrix
    with the same number of rows as `t0`.

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

        self.log_lmda = nn.Parameter(torch.log(torch.tensor(lmda)))

    def forward(
        self,
        t0: torch.Tensor,  # noqa: ARG002
        t1: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        """Calls the Exponential base hazard.

        Args:
            t0 (torch.Tensor): Former transition time.
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

    It is given by the formula

    .. math::
        \lambda(t) = \frac{k}{\lambda} \left( \frac{t}{\lambda} \right)^{k - 1}.

    This returns the base hazard in log scale. It expects a former transition time
    column vector `t0` as well as a matrix of next time points `t1`. `t1` is a matrix
    with the same number of rows as `t0`.

    If `clock_type` is set to `sojourn`, given `t0` and `t1`, the transformation will be
    computed at `t1 - t0` (sojourn time), and simply `t1` if set to `absolute`.

    Attributes:
        log_k (nn.Parameter): The log of the shape parameter.
        log_lmda (nn.Parameter): The log of the scale parameter.
        clock_type (str): The type of clock to use.
    """

    log_lmda: nn.Parameter
    log_k: nn.Parameter
    clock_type: str

    @validate_params(
        {
            "lmda": [Interval(Real, 0, None, closed="neither")],
            "k": [Interval(Real, 0, None, closed="neither")],
            "clock_type": [StrOptions({"sojourn", "absolute"})],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, lmda: float, k: float, *, clock_type: str = "sojourn"):
        """Initializes the Weibull base hazard.

        Args:
            lmda (float): The scale parameter.
            k (float): The shape parameter.
            clock_type (str, optional): The type of clock to use. Defaults to
                "sojourn".
        """
        super().__init__()  # type: ignore

        self.log_lmda = nn.Parameter(torch.log(torch.tensor(lmda)))
        self.log_k = nn.Parameter(torch.log(torch.tensor(k)))
        self.clock_type = clock_type

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the Weibull base hazard.

        Args:
            t0 (torch.Tensor): Former transition time.
            t1 (torch.Tensor): Current time

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        t = t1 - t0 if self.clock_type == "sojourn" else t1
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
    It is given by the formula

    .. math::
        \lambda(t) = a \exp{bt}.

    This returns the base hazard in log scale. It expects a former transition time
    column vector `t0` as well as a matrix of next time points `t1`. `t1` is a matrix
    with the same number of rows as `t0`.

    If `clock_type` is set to `sojourn`, given `t0` and `t1`, the transformation will be
    computed at `t1 - t0` (sojourn time), and simply `t1` if `clock_type` is set to
    `absolute`.

    Attributes:
        log_a (nn.Parameter): The baseline hazard parameter.
        b (nn.Parameter): The shape parameter.
        clock_type (str): The type of clock to use.
    """

    log_a: nn.Parameter
    b: nn.Parameter
    clock_type: str

    @validate_params(
        {
            "a": [Interval(Real, 0, None, closed="neither")],
            "b": [Interval(Real, None, None, closed="neither")],
            "clock_type": [StrOptions({"sojourn", "absolute"})],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, a: float, b: float, *, clock_type: str = "sojourn"):
        """Initializes the Gompertz base hazard.

        Args:
            a (float): The baseline hazard.
            b (float): The shape parameter.
            clock_type (str, optional): The type of clock to use. Defaults to
                "sojourn".
        """
        super().__init__()  # type: ignore

        self.log_a = nn.Parameter(torch.log(torch.tensor(a)))
        self.b = nn.Parameter(torch.tensor(b))
        self.clock_type = clock_type

    def __call__(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the Gompertz base hazard.

        Args:
            t0 (torch.Tensor): Former transition time.
            t1 (torch.Tensor): Current time

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        t = t1 - t0 if self.clock_type == "sojourn" else t1
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
    It is given by the formula

    .. math::
        \lambda(t) = \frac{\phi\left( \frac{\log t - \mu}{\sigma} \right)}{t \sigma
        \, \Phi\left( -\frac{\log t - \mu}{\sigma} \right)},
        \quad t > 0,

    where:

    .. math::
        \phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}, \quad
        \Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^z e^{-t^2/2} \, dt.

    This returns the base hazard in log scale. It expects a former transition time
    column vector `t0` as well as a matrix of next time points `t1`. `t1` is a matrix
    with the same number of rows as `t0`.

    If `clock_type` is set to `sojourn`, given `t0` and `t1`, the transformation will be
    computed at `t1 - t0` (sojourn time), and simply `t1` if `clock_type` is set to
    `absolute`.

    Attributes:
        mu (nn.Parameter): The log time mean.
        log_scale (nn.Parameter): The log of scale.
        clock_type (str): The type of clock to use.
    """

    mu: nn.Parameter
    log_scale: nn.Parameter
    clock_type: str

    @validate_params(
        {
            "mu": [Interval(Real, None, None, closed="neither")],
            "scale": [Interval(Real, 0, None, closed="neither")],
            "clock_type": [StrOptions({"sojourn", "absolute"})],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, mu: float, scale: float, *, clock_type: str = "sojourn"):
        """Initializes the log normal base hazard.

        Args:
            mu (float): The log time mean.
            scale (float): The log time scale.
            clock_type (str, optional): The type of clock to use. Defaults to
                "sojourn".

        Returns:
            LogBaseHazardFn: Returns the log normal base hazard function.
        """
        super().__init__()  # type: ignore

        self.mu = nn.Parameter(torch.tensor(mu))
        self.log_scale = nn.Parameter(torch.log(torch.tensor(scale)))
        self.clock_type = clock_type

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the log normal base hazard.

        Args:
            t0 (torch.Tensor): Former transition time.
            t1 (torch.Tensor): Current time

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        t = t1 - t0 if self.clock_type == "sojourn" else t1
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
