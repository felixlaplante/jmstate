from typing import cast

import torch
from beartype import beartype

from ..typedefs._defs import (
    LOGTWOPI,
    BaseHazardFn,
    ClockMethod,
    Num,
    NumStrictlyPositive,
    Tensor2D,
    TensorCol,
)


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


def exponential(lmda: NumStrictlyPositive) -> BaseHazardFn:
    """Returns the exponential base hazard function.

    Args:
        lmda (NumStrictlyPositive): The scale parameter.

    Raises:
        ValueError: If lmda is not strictly positive.

    Returns:
        BaseHazardFn: The expoential base hazard function.
    """
    lmda_t = torch.tensor(lmda, dtype=torch.get_default_dtype())

    # Checks
    if lmda_t <= 0:
        raise ValueError(f"lmda must be strictly positive, got {lmda_t}")

    log_lmda = torch.log(lmda_t)

    def _exponential(t0: TensorCol, t1: Tensor2D) -> torch.Tensor:
        return log_lmda

    return _exponential


@beartype
def weibull(
    k: NumStrictlyPositive,
    lmda: NumStrictlyPositive,
    clock_method: ClockMethod = clock_reset,
) -> BaseHazardFn:
    """Returns the weibull base hazard function.

    Args:
        k (NumStrictlyPositive): The shape parameter.
        lmda (NumStrictlyPositive): The scale parameter.
        clock_method (ClockMethod, optional): The ClockMethod transformation. Defaults to clock_reset.

    Raises:
        ValueError: If k is not strictly positive.
        ValueError: If lmda is not strictly positive.

    Returns:
        BaseHazardFn: The weibull base hazard function.
    """
    k_t = torch.tensor(k, dtype=torch.get_default_dtype())
    lmda_t = torch.tensor(lmda, dtype=torch.get_default_dtype())

    # Checks
    if k_t <= 0:
        raise ValueError(f"k must be strictly positive, got {k_t}")
    if lmda_t <= 0:
        raise ValueError(f"lmda must be strictly positive, got {lmda_t}")

    log_k = torch.log(k_t)
    log_lmda = torch.log(lmda_t)

    def _weibull(t0: TensorCol, t1: Tensor2D) -> torch.Tensor:
        t = clock_method(t0, t1)
        log_t = torch.log(t)
        return log_k - log_lmda + (k_t - 1) * (log_t - log_lmda)

    return _weibull


@beartype
def gompertz(
    a: NumStrictlyPositive,
    b: Num,
    clock_method: ClockMethod = clock_reset,
) -> BaseHazardFn:
    """Returns the gompertz base hazard function.

    Args:
        a (NumStrictlyPositive): The baseline hazard.
        b (Num): The shape parameter.
        clock_method (ClockMethod, optional): The ClockMethod transformation. Defaults to clock_reset.

    Raises:
        ValueError: If a is not strictly positive.

    Returns:
        BaseHazardFn: The gompertz base hazard function.
    """
    # Conversion en tenseur
    a_t = torch.tensor(a, dtype=torch.get_default_dtype())

    # Checks
    if a_t <= 0:
        raise ValueError(f"a must be strictly positive, got {a_t}")

    log_a = torch.log(a_t)

    def _gompertz(t0: TensorCol, t1: Tensor2D) -> torch.Tensor:
        t = clock_method(t0, t1)
        return log_a + b * t

    return _gompertz


@beartype
def log_normal(
    mu: Num,
    scale: NumStrictlyPositive,
    clock_method: ClockMethod = clock_reset,
) -> BaseHazardFn:
    """Returns the log normal base hazard function.

    Args:
        mu (Num):  log time mean.
        scale (NumStrictlyPositive): The log time scale.
        clock_method (ClockMethod, optional): The ClockMethod transformation. Defaults to clock_reset.

    Raises:
        ValueError: If scale is not strictly positive.

    Returns:
        BaseHazardFn: Returns the log normal base hazard function.
    """
    scale_t = torch.tensor(scale, dtype=torch.get_default_dtype())

    # Checks
    if scale_t <= 0:
        raise ValueError(f"scale must be strictly positive, got {scale_t}")

    log_scale = torch.log(scale_t)

    def _log_normal(t0: TensorCol, t1: Tensor2D) -> torch.Tensor:
        t = clock_method(t0, t1)
        log_t = torch.log(t)
        z = (log_t - mu) / scale
        log_pdf = -log_t - log_scale - 0.5 * LOGTWOPI - 0.5 * z**2
        log_sf = cast(torch.Tensor, torch.special.log_ndtr(-z))  # type: ignore
        return log_pdf - log_sf

    return _log_normal
