import itertools

import torch

from ..typedefs._defs import Tensor1D, Tensor2D, Trajectory


def check_inf(tensors: list[torch.Tensor | None]):
    """Check if any of the tensors contains infinity.

    Args:
        tensors (list[torch.Tensor  |  None]): The tensors to check.

    Raises:
        ValueError: If one tensor contains infinity.
    """
    if any(t is not None and t.isinf().any() for t in tensors):
        raise ValueError("Tensors cannot contain inf values")


def check_consistent_size(tensors: list[torch.Tensor | None], n: int):
    """Checks it the number of individuals is consistent.

    Args:
        tensors (list[torch.Tensor  |  None]): The tensors to check.
        n (int): The expected size.

    Raises:
        ValueError: If the number of inconsistent.
    """
    if any(tensor is not None and tensor.shape[0] != n for tensor in tensors):
        raise ValueError("Inconsistent number of individuals")


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


def check_trajectory_c(trajectories: list[Trajectory], c: Tensor1D | Tensor2D | None):
    """Check if the trajectories are compatible with censoring times.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        c (Tensor1D | Tensor2D | None): The censoring times.

    Raises:
        ValueError: If some trajectory is not compatible with the censoring.
    """
    if c is not None and any(
        trajectory[-1][0] > c for trajectory, c in zip(trajectories, c)
    ):
        raise ValueError("Last trajectory time must not be greater than c")
