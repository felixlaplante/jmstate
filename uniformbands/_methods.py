from typing import cast

import numpy as np
from scipy.stats import rankdata  # type: ignore


def uniform(
    F_lo: np.ndarray,
    F_hi: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Implements the Uniform method for uniform control bands.

    Args:
        F_lo (np.ndarray): The lower high probability bounds.
        F_hi (np.ndarray):  The upper high probability bounds.
        alpha (float): The level of supplementary risk.

    Returns:
        dict[str, np.ndarray]: The uniform bands for level alpha
            of supplementary risk.
    """
    rank_lo, rank_hi = (
        rankdata(F_lo, method="max", axis=-2),
        rankdata(F_hi, method="min", axis=-2),
    )
    infZ, supZ = rank_lo.min(axis=-1) - 1, rank_hi.max(axis=-1) - 1

    q_lo = cast(int, np.quantile(infZ, alpha / 2, method="lower"))
    q_hi = cast(int, np.quantile(supZ, 1 - alpha / 2, method="higher"))

    sorted_lo = np.sort(F_lo, axis=-2)
    sorted_hi = np.sort(F_hi, axis=-2)

    return {
        "lower": sorted_lo.take(q_lo, axis=-2),  # type: ignore
        "upper": sorted_hi.take(q_hi, axis=-2),  # type: ignore
    }


def student(
    F_lo: np.ndarray,
    F_hi: np.ndarray,
    alpha: float,
    eps: float,
    min_val: float,
    max_val: float,
) -> dict[str, np.ndarray]:
    """Implements the Student method for uniform control bands.

    Args:
        F_lo (np.ndarray): The lower high probability bounds.
        F_hi (np.ndarray):  The upper high probability bounds.
        alpha (float): The level of supplementary risk.
        eps (float): The regularization parameter to ensure a well defined division.
        min_val (float): The minimum accepted values of the functions.
        max_val (float): The maximum accepted values of the functions.

    Returns:
        dict[str, np.ndarray]: The uniform bands for level alpha
            of supplementary risk.
    """
    mean_lo, mean_hi = F_lo.mean(axis=-2), F_hi.mean(axis=-2)
    std_lo, std_hi = F_lo.std(axis=-2) + eps, F_hi.std(axis=-2) + eps

    T_lo, T_hi = (F_lo - mean_lo) / std_lo, (F_hi - mean_hi) / std_hi
    infT, supT = T_lo.min(axis=-1), T_hi.max(axis=-1)

    q_lo = cast(float, np.quantile(infT, alpha / 2))
    q_hi = cast(float, np.quantile(supT, 1 - alpha / 2))

    return {
        "lower": np.clip(mean_lo + q_lo * std_lo, min_val, max_val),
        "upper": np.clip(mean_hi + q_hi * std_hi, min_val, max_val),
    }
