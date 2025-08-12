from typing import cast

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import LOGTWOPI, ClockMethod, Num, NumStrictlyPositive


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


class Exponential:
    r"""Implements the Exponential base hazard.

    Exponential base hazard is time independent.
    It is given by the formula:

    .. math::
        \lambda(t) = \lambda.

    This returns the base hazard in log scale.$

    Attributes:
        lmda (torch.Tensor): The rate factor.
        log_lmda (torch.Tensor): The log rate factor.
    """

    lmda: torch.Tensor
    log_lmda: torch.Tensor

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, lmda: NumStrictlyPositive):
        """Initializes the Exponential hazard.

        Args:
            lmda (NumStrictlyPositive): The rate factor.

        Raises:
            ValueError: If lmda is not strictly positive.
        """
        self.lmda = torch.tensor(lmda, dtype=torch.get_default_dtype())

        if self.lmda <= 0:
            raise ValueError(f"lmda must be strictly positive, got {self.lmda}")

        self.log_lmda = torch.log(self.lmda)

    def __call__(
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


class Weibull:
    r"""Implements the Weibull base hazard.

    Weibull base hazard is time dependent.
    It is given by the formula:

    .. math::
        \lambda(t) = \frac{k}{\lambda} \frac{x}{\lambda}^{k - 1}.

    This returns the base hazard in log scale.

    Attributes:
        k (torch.Tensor): The shape parameter.
        lmda (torch.Tensor): The scale parameter.
        clock_method (ClockMethod): The ClockMethod transformation.
        log_k (torch.Tensor): The log of the shape parameter.
        log_lmda (torch.Tensor): The log of the scale parameter.
    """

    k: torch.Tensor
    lmda: torch.Tensor
    clock_method: ClockMethod
    log_k: torch.Tensor
    log_lmda: torch.Tensor

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        k: NumStrictlyPositive,
        lmda: NumStrictlyPositive,
        clock_method: ClockMethod = clock_reset,
    ):
        """Initializes the Weivull base hazard.

        Args:
            k (NumStrictlyPositive): The shape parameter.
            lmda (NumStrictlyPositive): The scale parameter.
            clock_method (ClockMethod, optional): The ClockMethod transformation.
                Defaults to clock_reset.

        Raises:
            ValueError: If k is not strictly positive.
            ValueError: If lmda is not strictly positive.
        """
        self.k = torch.tensor(k, dtype=torch.get_default_dtype())
        self.lmda = torch.tensor(lmda, dtype=torch.get_default_dtype())
        self.clock_method = clock_method

        if self.k <= 0:
            raise ValueError(f"k must be strictly positive, got {self.k}")
        if self.lmda <= 0:
            raise ValueError(f"lmda must be strictly positive, got {self.lmda}")

        self.log_k = torch.log(self.k)
        self.log_lmda = torch.log(self.lmda)

    def __call__(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Calls the Weivull base hazard.

        Args:
            t0 (torch.Tensor): Past transition time.
            t1 (torch.Tensor): Current time

        Returns:
            torch.Tensor: The computed base hazard in log scale.
        """
        t = self.clock_method(t0, t1)
        log_t = torch.log(t)
        return self.log_k - self.log_lmda + (self.k - 1) * (log_t - self.log_lmda)


class Gompertz:
    r"""Implements the Gompertz base hazard.

    Gompertz base hazard is time dependent.
    It is given by the formula:

    .. math::
        \lambda(t) = a \exp{bt}.

    This returns the base hazard in log scale.

    Attributes:
        a (torch.Tensor) : The baseline hazard.
        b (int | float): The shape parameter.
        clock_method (ClockMethod): The ClockMethod transformation.
    """

    a: torch.Tensor
    b: int | float
    clock_method: ClockMethod

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
        self.a = torch.tensor(a, dtype=torch.get_default_dtype())
        self.b = b
        self.clock_method = clock_method

        if self.a <= 0:
            raise ValueError(f"a must be strictly positive, got {self.a}")

        self.log_a = torch.log(self.a)

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


class LogNormal:
    r"""Implements the log normal base hazard.

    Log normal base hazard is time dependent.
    It is given by the formula :

    .. math::
        \lambda(t) = \frac{\phi\left( \frac{\log t - \mu}{\sigma} \right)}{t \sigma
        \Phi\left( -\frac{\log t - \mu}{\sigma} \right)},
        \quad t > 0,

    where:

    .. math::
        \phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}, \quad
        \Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^z e^{-t^2/2} \, dt

    This returns the base hazard in log scale.

    Attributes:
        mu (int | float): The log time mean.
        scale (torch.Tensor): The log time scale.
        clock_method (ClockMethod): The ClockMethod transformation.
        log_scale (torch.Tensor): The log of scale.
    """

    mu: int | float
    scale: torch.Tensor
    clock_method: ClockMethod
    log_scale: torch.Tensor

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
        self.mu = mu
        self.scale = torch.tensor(scale, dtype=torch.get_default_dtype())
        self.clock_method = clock_method

        if self.scale <= 0:
            raise ValueError(f"scale must be strictly positive, got {self.scale}")

        self.log_scale = torch.log(self.scale)

    def __call__(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
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
        log_pdf = -log_t - self.log_scale - 0.5 * LOGTWOPI - 0.5 * z**2
        log_sf = cast(torch.Tensor, torch.special.log_ndtr(-z))  # type: ignore
        return log_pdf - log_sf
