from typing import cast

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import LOGTWOPI, ClockMethod, Num, NumStrictlyPositive


def clock_forward(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    """Time transformation for clock forward method.

    Args:
        t0 (torch.Tensor): Past transition time.
        t1 (torch.Tensor): Current time

    Returns:
        torch.Tensor: Current time.
    """
    return t1


def clock_reset(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    """Time transformation for clock reset method.

    Args:
        t0 (torch.Tensor): Past transition time.
        t1 (torch.Tensor): Current time

    Returns:
        torch.Tensor: Current time - Past transition time.
    """
    return t1 - t0


class Exponential:
    """Implements the exponential base hazard."""

    lmda: torch.Tensor
    log_lmda: torch.Tensor

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, lmda: NumStrictlyPositive):
        """Initializes the exponential hazard.

        Args:
            lmda (NumStrictlyPositive): The rate factor.

        Raises:
            ValueError: If lmda is not strictly positive.
        """
        self.lmda = torch.tensor(lmda, dtype=torch.get_default_dtype())

        if self.lmda <= 0:
            raise ValueError(f"lmda must be strictly positive, got {self.lmda}")

        self.log_lmda = torch.log(self.lmda)

    def __call__(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        return self.log_lmda


class Weibull:
    """Implements the weibull base hazard."""

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
        """Initializes the weibull base hazard.

        Args:
            k (NumStrictlyPositive): The shape parameter.
            lmda (NumStrictlyPositive): The scale parameter.
            clock_method (ClockMethod, optional): The ClockMethod transformation. Defaults to clock_reset.

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
        t = self.clock_method(t0, t1)
        log_t = torch.log(t)
        return self.log_k - self.log_lmda + (self.k - 1) * (log_t - self.log_lmda)


class Gompertz:
    """Implements the weibull base hazard."""

    a: torch.Tensor
    b: int | float
    clock_method: ClockMethod
    log_k: torch.Tensor
    log_lmda: torch.Tensor

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        a: NumStrictlyPositive,
        b: NumStrictlyPositive,
        clock_method: ClockMethod = clock_reset,
    ):
        """Initializes the gompertz base hazard.

        Args:
            a (NumStrictlyPositive): The baseline hazard.
            b (Num): The shape parameter.
            clock_method (ClockMethod, optional): The ClockMethod transformation. Defaults to clock_reset.

        Raises:
            ValueError: If a is not strictly positive.
        """
        self.a = torch.tensor(a, dtype=torch.get_default_dtype())
        self.b = b
        self.clock_method = clock_method

        if self.a <= 0:
            raise ValueError(f"a must be strictly positive, got {self.a}")

        self.log_a = torch.log(self.a)

    def __call__(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        t = self.clock_method(t0, t1)
        return self.log_a + self.b * t


class LogNormal:
    """Implements the weibull base hazard."""

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
            mu (Num):  log time mean.
            scale (NumStrictlyPositive): The log time scale.
            clock_method (ClockMethod, optional): The ClockMethod transformation. Defaults to clock_reset.

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
        t = self.clock_method(t0, t1)
        log_t = torch.log(t)
        z = (log_t - self.mu) / self.scale
        log_pdf = -log_t - self.log_scale - 0.5 * LOGTWOPI - 0.5 * z**2
        log_sf = cast(torch.Tensor, torch.special.log_ndtr(-z))  # type: ignore
        return log_pdf - log_sf
