from typing import SupportsFloat, cast

import torch
from beartype import beartype

from ..typedefs._defs import BaseHazardFn, ClockMethod, Tensor2D, TensorCol


def clock_forward(t0: TensorCol, t1: Tensor2D) -> Tensor2D:
    """Time transformation for clock forward method.

    Args:
        t0 (TensorCol): Past transition time.
        t1 (Tensor2D): Current time

    Returns:
        Tensor2D: Current time.
    """
    return t1


def clock_reset(t0: TensorCol, t1: Tensor2D) -> Tensor2D:
    """Time transformation for clock reset method.

    Args:
        t0 (TensorCol): Past transition time.
        t1 (Tensor2D): Current time

    Returns:
        Tensor2D: Current time - Past transition time.
    """
    return t1 - t0


def exponential(lmda: SupportsFloat) -> BaseHazardFn:
    """Returns the exponential base hazard function.

    Args:
        lmda (SupportsFloat): The scale parameter.

    Raises:
        ValueError: If lmda is not strictly positive.

    Returns:
        BaseHazardFn: The expoential base hazard function.
    """
    lmda = torch.as_tensor(lmda, dtype=torch.float64)

    # Checks
    if lmda <= 0:
        raise ValueError(f"lmda must be strictly positive, got {lmda}")

    log_lmda = torch.log(lmda)

    def _exponential(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        return log_lmda

    return _exponential


@beartype
def weibull(
    k: SupportsFloat,
    lmda: SupportsFloat,
    clock_method: ClockMethod = clock_reset,
) -> BaseHazardFn:
    """Returns the weibull base hazard function.

    Args:
        k (SupportsFloat): The shape parameter.
        lmda (SupportsFloat): The scale parameter.
        clock_method (ClockMethod, optional): The ClockMethod transformation. Defaults to clock_reset.

    Raises:
        ValueError: If k is not strictly positive.
        ValueError: If lmda is not strictly positive.

    Returns:
        BaseHazardFn: The weibull base hazard function.
    """
    k = torch.as_tensor(k, dtype=torch.float64)
    lmda = torch.as_tensor(lmda, dtype=torch.float64)

    # Checks
    if k <= 0:
        raise ValueError(f"k must be strictly positive, got {k}")
    if lmda <= 0:
        raise ValueError(f"lmda must be strictly positive, got {lmda}")

    log_k = torch.log(k)
    log_lmda = torch.log(lmda)

    def _weibull(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        t = clock_method(t0, t1)
        log_t = torch.log(t)
        return log_k - log_lmda + (k - 1) * (log_t - log_lmda)

    return _weibull


@beartype
def gompertz(
    a: SupportsFloat,
    b: SupportsFloat,
    clock_method: ClockMethod = clock_reset,
) -> BaseHazardFn:
    """Returns the gompertz base hazard function.

    Args:
        a (SupportsFloat): The baseline hazard.
        b (SupportsFloat): The shape parameter.
        clock_method (ClockMethod, optional): The ClockMethod transformation. Defaults to clock_reset.

    Raises:
        ValueError: If a is not strictly positive.

    Returns:
        BaseHazardFn: The gompertz base hazard function.
    """
    # Conversion en tenseur
    a = torch.as_tensor(a, dtype=torch.float64)

    # Checks
    if a <= 0:
        raise ValueError(f"a must be strictly positive, got {a}")

    b = torch.as_tensor(b, dtype=torch.float64)
    log_a = torch.log(a)

    def _gompertz(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        t = clock_method(t0, t1)
        # log h(t) = log a + b * t
        return log_a + b * t

    return _gompertz


@beartype
def log_normal(
    mu: SupportsFloat,
    scale: SupportsFloat,
    clock_method: ClockMethod = clock_reset,
) -> BaseHazardFn:
    """Returns the log normal base hazard function.

    Args:
        mu (SupportsFloat):  log time mean.
        scale (SupportsFloat): The log time scale.
        clock_method (ClockMethod, optional): The ClockMethod transformation. Defaults to clock_reset.

    Raises:
        ValueError: If scale is not strictly positive.

    Returns:
        BaseHazardFn: Returns the log normal base hazard function.
    """
    mu = torch.as_tensor(mu, dtype=torch.float64)
    scale = torch.as_tensor(scale, dtype=torch.float64)

    # Checks
    if scale <= 0:
        raise ValueError(f"sigma must be strictly positive, got {scale}")

    log_scale = torch.log(scale)
    log_2pi = torch.log(torch.tensor(2 * torch.pi, dtype=torch.float64))

    def _log_normal(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        t = clock_method(t0, t1)
        log_t = torch.log(t)
        z = (log_t - mu) / scale
        log_pdf = -log_t - log_scale - 0.5 * log_2pi - 0.5 * z**2
        log_sf = cast(torch.Tensor, torch.special.log_ndtr(-z))  # type: ignore
        return log_pdf - log_sf

    return _log_normal
