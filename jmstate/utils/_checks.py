import itertools

import torch

from ..typedefs._defs import MatRepr, Trajectory


def check_inf(tensors: tuple[tuple[torch.Tensor | None, str], ...]):
    """Check if any of the tensors contain infinity.

    Args:
        tensors (tuple[tuple[torch.Tensor | None, str] ,...]): The tuples to check.

    Raises:
        ValueError: If any of the tensors contain infinity.
    """
    for t, name in tensors:
        if t is not None and t.isinf().any():
            raise ValueError(f"Tensor {name} cannot contain inf values")


def check_nan(tensors: tuple[tuple[torch.Tensor | None, str], ...]):
    """Check if any of the tensors contain NaNs.

    Args:
        tensors (tuple[tuple[torch.Tensor | None, str] ,...]): The tuples to check.

    Raises:
        ValueError: If any of the tensors contain NaNs.
    """
    for t, name in tensors:
        if t is not None and t.isnan().any():
            raise ValueError(f"Tensor {name} cannot contain NaN values")


def check_consistent_size(
    groups: tuple[tuple[torch.Tensor | int | None, int | None, str], ...],
):
    """Checks if all the tensors are consistent in size.

    Args:
        groups (tuple[tuple[torch.Tensor | int | None, int | None, str], ...]):
            The tuples to check.

    Raises:
        ValueError: If any of the sizes are inconsistent.
    """
    for (t0, d0, name0), (t1, d1, name1) in itertools.pairwise(groups):
        if t0 is None or t1 is None:
            continue
        dim0 = t0 if isinstance(t0, int) else t0.size(d0)
        dim1 = t1 if isinstance(t1, int) else t1.size(d1)

        if dim0 != dim1:
            raise ValueError(
                f"{dim0} != {dim1} at dims {d0, d1} for args {name0, name1}"
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
        not all(t0 <= t1 for t0, t1 in itertools.pairwise(t for t, _ in trajectory))
        for trajectory in trajectories
    ):
        raise ValueError(
            "Trajectories must be sorted by time, in ascending order. Also ensure "
            "there are no NaN values as this will trigger the check"
        )


def check_trajectory_c(trajectories: list[Trajectory], c: torch.Tensor | None):
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


def check_matrix_dim(mat_repr: MatRepr, name: str):
    """Sets dimensions for matrix.

    Args:
        mat_repr (MatRepr): The matrix representation.
        name (str): The matrix name.

    Raises:
        ValueError: If flat tensor is not flat.
        ValueError: If the number of elements is incompatible with method "full".
        ValueError: If the number of elements is incompatible with method "diag".
        ValueError: If the number of elements is not one and the method is "ball".
        ValueError: If the method is not in ("full", "diag", "ball").
    """
    flat, dim, method = mat_repr

    if flat.ndim != 1:
        raise ValueError(
            f"flat must be flat tensor, got shape {flat.shape} for matrix {name}"
        )

    match method:
        case "full":
            if flat.numel() != (dim * (dim + 1)) // 2:
                raise ValueError(
                    f"{flat.numel()} is incompatible with full matrix {name} of "
                    f"dimension {dim}"
                )
        case "diag":
            if flat.numel() != dim:
                raise ValueError(
                    f"{flat.numel()} is incompatible with diag matrix {name} of "
                    f"dimension {dim}"
                )
        case "ball":
            if flat.numel() != 1:
                raise ValueError(
                    f"Expected 1 element for flat, got {flat.numel()} for matrix {name}"
                )
        case _:
            raise ValueError(
                f"Method must be be either 'full', 'diag' or 'ball', got {method}"
            )
