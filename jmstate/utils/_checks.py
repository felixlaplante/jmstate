from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch

from ..typedefs._defs import Trajectory

if TYPE_CHECKING:
    from ..typedefs._params import ModelParams


def check_trajectories_empty(trajectories: list[Trajectory]):
    """Check if the trajectories are not empty.

    Args:
        trajectories (list[Trajectory]): The trajectories.

    Raises:
        ValueError: If some trajectory is empty.
    """
    if any(len(trajectory) == 0 for trajectory in trajectories):
        raise ValueError("Trajectories must not be empty")


def check_trajectories_sorting(trajectories: list[Trajectory]):
    """Check if the trajectories are well sorted.

    Args:
        trajectories (list[Trajectory]): The trajectories.

    Raises:
        ValueError: If some trajectory is not sorted.
    """
    if any(
        not all(t0 <= t1 for t0, t1 in itertools.pairwise(t for t, _ in trajectory))
        for trajectory in trajectories
    ):
        raise ValueError(
            "Trajectories must be sorted by time, in ascending order. Also ensure "
            "there are no NaN values as this will trigger the check"
        )


def check_trajectories_c(trajectories: list[Trajectory], c: torch.Tensor | None):
    """Check if the trajectories are compatible with censoring times.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor | None): The censoring times.

    Raises:
        ValueError: If some trajectory is not compatible with the censoring.
    """
    if c is not None and any(
        trajectory[-1][0] > c for trajectory, c in zip(trajectories, c, strict=True)
    ):
        raise ValueError("Last trajectory time must not be greater than censoring time")


def check_matrix_dim(flat: torch.Tensor, dim: int, method: str):
    """Sets dimensions for matrix.

    Args:
        flat (torch.Tensor): The flat tensor.
        dim (int): The dimension of the matrix.
        method (str): The method to use.

    Raises:
        ValueError: If the number of elements is incompatible with method "full".
        ValueError: If the number of elements is incompatible with method "diag".
        ValueError: If the number of elements is not one and the method is "ball".
        ValueError: If the method is not in ("full", "diag", "ball").
    """
    match method:
        case "full":
            if flat.numel() != (dim * (dim + 1)) // 2:
                raise ValueError(
                    f"{flat.numel()} is incompatible with full matrix of dimension "
                    f"{dim}"
                )
        case "diag":
            if flat.numel() != dim:
                raise ValueError(
                    f"{flat.numel()} is incompatible with diag matrix of dimension "
                    f"{dim}"
                )
        case "ball":
            if flat.numel() != 1:
                raise ValueError(
                    f"Expected 1 element for flat, got {flat.numel()} for matrix of "
                    f" dimension {dim}"
                )
        case _:
            raise ValueError(
                f"Method must be be either 'full', 'diag' or 'ball', got {method}"
            )


def check_params_align(params1: ModelParams, params2: ModelParams):
    """Checks if the named parameters lists have the same length, names and methods.

    Args:
        params1 (ModelParams): The first instance of ModelParams.
        params2 (ModelParams): The second instance of ModelParams.

    Raises:
        ValueError: If the parameters do not have the same names.
        ValueError: If the parameters do not have the same shapes.
    """
    if params1.as_dict.keys() != params2.as_dict.keys():
        raise ValueError("All parameters must have the same names")
    if any(
        t1.shape != t2.shape
        for t1, t2 in zip(params1.as_list, params2.as_list, strict=True)
    ):
        raise ValueError("All parameters must have the same shapes")
