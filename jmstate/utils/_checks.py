import itertools

import torch

from ..typedefs._defs import (
    MatRepr,
    TensorCol,
    Trajectory,
)


def check_inf(tensors: tuple[torch.Tensor | None, ...]):
    """Check if any of the tensors contains infinity.

    Args:
        tensors (tuple[torch.Tensor | None, ...]): The tensors to check.

    Raises:
        ValueError: If one tensor contains infinity.
    """
    if any(t is not None and t.isinf().any() for t in tensors):
        raise ValueError("Tensors cannot contain inf values")


def check_consistent_size(
    groups: tuple[tuple[torch.Tensor | None, int], ...],
    ref: int | None = None,
):
    """Checks if all the tensors are consistent in size.

    Args:
        groups (tuple[tuple[torch.Tensor  |  None, int], ...]): The tuple of tensors.
        ref (int | None, optional): The regerence or None. Defaults to None.

    Raises:
        ValueError: If the sizes are inconsistent.
        ValueError: If the size does not match the reference.
    """
    sizes = {t.size(d) for t, d in groups if t is not None}
    if len(sizes) > 1:
        raise ValueError(f"Incoherent dimension, found sizes: {sorted(sizes)}")
    if ref is not None and ref not in sizes:
        raise ValueError(
            f"Tensor sizes {sizes} don't match the reference dimension {ref}"
        )


def check_trajectory_empty(trajectories: list[Trajectory]):
    """Check if the trajectories are not empty.

    Args:
        trajectories (list[Trajectory]): The trajectories.

    Raises:
        ValueError: If some trajectory is empty.
    """
    if any(len(trajectory) == 0 for trajectory in trajectories):
        raise ValueError("Trajectories must not be empty")


def check_trajectory_sorting(trajectories: list[Trajectory]):
    """Check if the trajectories are well sorted.

    Args:
        trajectories (list[Trajectory]): The trajectories.

    Raises:
        ValueError: If some trajectory is not sorted.
    """
    if any(
        any(t0 > t1 for t0, t1 in itertools.pairwise(t for t, _ in trajectory))
        for trajectory in trajectories
    ):
        raise ValueError("Trajectories must be sorted by time")


def check_trajectory_c(trajectories: list[Trajectory], c: TensorCol | None):
    """Check if the trajectories are compatible with censoring times.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        c (TensorCol | None): The censoring times.

    Raises:
        ValueError: If some trajectory is not compatible with the censoring.
    """
    if c is not None and any(
        trajectory[-1][0] > c for trajectory, c in zip(trajectories, c)
    ):
        raise ValueError("Last trajectory time must not be greater than c")


def check_matrix_dim(mat_repr: MatRepr):
    """Sets dimensions for matrix.

    Args:
        mat_repr (MatRepr): The matrix representation.

    Raises:
        ValueError: If flat is not flat.
        ValueError: If the number of elements is incompatible with method "full".
        ValueError: If the number of elements is icompatible with method "diag".
        ValueError: If the number of elements is not one and the method is "ball".
        ValueError: If the method is unknown.
    """
    flat, dim, method = mat_repr

    if flat.ndim != 1:
        raise ValueError(f"flat must be flat tensor, got shpe {flat.shape}")

    match method:
        case "full":
            if flat.numel() != (dim * (dim + 1)) // 2:
                raise ValueError(
                    f"{flat.numel()} is incompatible with full matrix of dimension {dim}"
                )
        case "diag":
            if flat.numel() != dim:
                raise ValueError(
                    f"{flat.numel()} is incompatible with diag matrix of dimension {dim}"
                )
        case "ball":
            if flat.numel() != 1:
                raise ValueError(f"Expected 1 element for flat, got {flat.numel()}")
        case _:
            raise ValueError(f"Got method {method} unknown")
