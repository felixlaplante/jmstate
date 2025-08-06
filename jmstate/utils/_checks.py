import itertools

import torch

from ..typedefs._defs import MatRepr, Tensor2D, Tensor3D, TensorCol, Trajectory


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
    tensors: tuple[Tensor2D | Tensor3D | None, ...], dims: tuple[int, ...], n: int
):
    """Checks it the number of individuals is consistent.

    Args:
        tensors (tuple[Tensor2D | Tensor3D | None, ...]): The tensors to check.
        dims (tuple[int, ...]): The dimensions to check.
        n (int): The expected size.

    Raises:
        ValueError: If the number of inconsistent.
    """
    if any(
        tensor is not None and tensor.size(dim) != n
        for tensor, dim in zip(tensors, dims)
    ):
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


def check_sampler(
    n_chains: int, init_step_size: float, adapt_rate: float, target_accept_rate: float
):
    """Check if every input is valid.

    Args:
        n_chains (int): Number of parallel MCMC chains.
        init_step_size (float): Kernel step size.
        adapt_rate (float): Adaptation rate of the step size.
        target_accept_rate (float): Target acceptance rate.

    Raises:
        TypeError: If n_chains is not strictly positive.
        ValueError: If init_step_size is not strictly positive.
        ValueError: If adapt_rate is not positive.
        ValueError: If target_accept_rate is not in (0, 1).
    """
    if n_chains <= 0:
        raise ValueError(f"n_chains must be strictly positive, got {n_chains}")
    if init_step_size <= 0:
        raise ValueError(
            f"init_step_size must be strictly positive, got {init_step_size}"
        )
    if adapt_rate < 0:
        raise ValueError(f"adapt_rate must be positive, got {adapt_rate}")
    if not 0 < target_accept_rate < 1:
        raise ValueError(
            f"target_accept_rate must be in (0, 1), got {target_accept_rate}"
        )
