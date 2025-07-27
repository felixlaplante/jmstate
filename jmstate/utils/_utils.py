import itertools
from collections import defaultdict
from typing import Any, cast

import numpy as np
import torch

from ._defs import TWO, BaseHazardFn, BucketData, LinkFn, Trajectory


def legendre_quad(n_quad: int) -> tuple[torch.Tensor, ...]:
    """Get the Legendre quadrature nodes and weights.

    Args:
        n_quad (int, optional): The number of quadrature points.

    Returns:
        tuple[torch.Tensor, ...]: The nodes and weights.
    """
    nodes, weights = cast(
        tuple[
            np.ndarray[Any, np.dtype[np.float32]],
            np.ndarray[Any, np.dtype[np.float32]],
        ],
        np.polynomial.legendre.leggauss(n_quad),  # type: ignore
    )

    std_nodes = torch.tensor(nodes, dtype=torch.float32)
    std_weights = torch.tensor(weights, dtype=torch.float32)

    return std_nodes, std_weights


def flat_from_tril(L: torch.Tensor) -> torch.Tensor:
    """Flatten the lower triangular part (including the diagonal) of a square matrix.

    Into a 1D tensor, in row-wise order.

    Args:
        L (torch.Tensor): Square lower-triangular matrix of shape (n, n).

    Raises:
        ValueError: If the input is not square.
        RuntimeError: If the flattening fails.

    Returns:
        torch.Tensor: Flattened 1D tensor containing the lower triangular entries.
    """
    try:
        if L.ndim != TWO or L.shape[0] != L.shape[1]:
            raise ValueError("Input must be a square matrix")

        n = L.shape[0]
        i, j = torch.tril_indices(n, n)

        return L[i, j]

    except Exception as e:
        raise RuntimeError(f"Failed to flatten matrix: {e}") from e


def tril_from_flat(flat: torch.Tensor, n: int) -> torch.Tensor:
    """Generate the lower triangular matrix associated with flat tensor.

    Args:
        flat (torch.Tensor): Flat tehsnro
        n (int): Dimension of the matrix.

    Raises:
        ValueError: Error if the the dimensions do not allow matrix computation.
        RuntimeError: Error if the computation fails.

    Returns:
        torch.Tensor: The lower triangular matrix.
    """
    if flat.numel() != (n * (n + 1)) // 2:
        raise ValueError("Incompatible dimensions for lower triangular matrix")

    L = torch.zeros(n, n, dtype=flat.dtype).index_put_(
        tuple(torch.tril_indices(n, n)), flat
    )

    return L


def log_cholesky_from_flat(
    flat: torch.Tensor, n: int, method: str = "full"
) -> torch.Tensor:
    """Computes log cholesky from flat tensor according to choice of method.

    Args:
        flat (torch.Tensor): The flat tensor parameter.
        n (int): The dimension of the matrix.
        method (str, optional): The method, full, diagonal or ball. Defaults to "full".

    Raises:
        ValueError: If the array is not flat.
        ValueError: If the number of parameters is inconsistent with n.
        ValueError: If the number of parameters does not equal one.
        ValueError: If the method is not in ("full", "diag", "ball").

    Returns:
        torch.Tensor: The log cholesky representation.
    """
    if flat.ndim != 1:
        raise ValueError(f"flat should be flat, got shape {flat.shape}")

    match method:
        case "full":
            return tril_from_flat(flat, n)
        case "diag":
            if flat.numel() != n:
                raise ValueError(f"flat has {flat.numel()} elements, expected {n}")
            return torch.diag(flat)
        case "ball":
            if flat.numel() != 1:
                f"flat has {flat.numel()} elements, expected 1"
            return flat * torch.eye(n)
        case _:
            raise ValueError(f"Got method {method} unknown")


def flat_from_log_cholesky(L: torch.Tensor, method: str = "full") -> torch.Tensor:
    """Computes flat tensor from log cholesky matrix according to choice of method.

    Args:
        L (torch.Tensor): The square lower triangular matrix parameter.
        method (str, optional): The method, full, diagonal or ball. Defaults to "full".

    Raises:
        ValueError: If the method is not in ("full", "diag", "ball").

    Returns:
        torch.Tensor: The flat representation.
    """
    if L.ndim != TWO or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square, got shape {L.shape}")

    match method:
        case "full":
            return flat_from_tril(L)
        case "diag":
            return L.diagonal()
        case "ball":
            return L[0, 0]
        case _:
            raise ValueError(f"Got method {method} unknown")


def build_buckets(
    trajectories: list[Trajectory],
) -> dict[tuple[int, int], BucketData]:
    """Builds buckets from trajectories for user convenience.

    Args:
        trajectories (list[Trajectory]): The list of individual trajectories.

    Raises:
        RuntimeError: If the construction of the buckets fails.

    Returns:
        dict[tuple[int, int], BucketData]: A dictionnary of transition keys with a triplet of tensors (idxs, t0, t1).
    """
    try:
        # Process each individual trajectory
        buckets: defaultdict[tuple[int, int], list[list[Any]]] = defaultdict(
            lambda: [[], [], []]
        )

        for i, trajectory in enumerate(trajectories):
            for (t0, s0), (t1, s1) in itertools.pairwise(trajectory):
                key = (s0, s1)
                buckets[key][0].append(i)
                buckets[key][1].append(t0)
                buckets[key][2].append(t1)

        return {
            key: BucketData(
                torch.tensor(vals[0], dtype=torch.int64),
                torch.tensor(vals[1], dtype=torch.float32),
                torch.tensor(vals[2], dtype=torch.float32),
            )
            for key, vals in buckets.items()
            if vals[0]  # skip empty
        }

    except Exception as e:
        raise RuntimeError(f"Failed to construct buckets: {e}") from e


def build_vec_rep(
    trajectories: list[Trajectory],
    c: torch.Tensor,
    surv: dict[tuple[int, int], tuple[BaseHazardFn, LinkFn]],
) -> dict[tuple[int, int], tuple[torch.Tensor, ...]]:
    """Build vectorizable bucket representation.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor): Censoring times.
        surv (dict[tuple[int, int], tuple[BaseHazardFn, LinkFn]]) : The model survival dict.

    Raises:
        ValueError: If some keys are not in surv.
        RuntimeError: If the building fails.

    Returns:
        dict[tuple[int, int], tuple[torch.Tensor, ...]]: The vectorizable buckets representation.
    """
    try:
        # Get survival transitions defined in the model
        trans = set(surv.keys())

        # Build alternative state mapping
        alt_map: defaultdict[int, list[int]] = defaultdict(list)
        for from_state, to_state in trans:
            alt_map[from_state].append(to_state)

        # Initialize buckets
        buckets: dict[tuple[int, int], list[list[Any]]] = defaultdict(
            lambda: [[], [], [], []]
        )

        # Process each individual trajectory
        for i, trajectory in enumerate(trajectories):
            # Add censoring
            ext_trajectory = [*trajectory, (float(c[i]), None)]

            for (t0, s0), (t1, s1) in itertools.pairwise(ext_trajectory):
                if t0 >= t1:
                    continue

                if s1 is not None and (s0, s1) not in trans:
                    raise ValueError(
                        f"Transition {(s0, s1)} must be in model_design.surv keys"
                    )

                for alt_state in alt_map[cast(int, s0)]:
                    key = (cast(int, s0), alt_state)
                    buckets[key][0].append(i)
                    buckets[key][1].append(t0)
                    buckets[key][2].append(t1)
                    buckets[key][3].append(alt_state == s1)

        return {
            key: (
                torch.tensor(vals[0], dtype=torch.int64),
                torch.tensor(vals[1], dtype=torch.float32),
                torch.tensor(vals[2], dtype=torch.float32),
                torch.tensor(vals[3], dtype=torch.bool),
            )
            for key, vals in buckets.items()
            if vals[0]
        }

    except Exception as e:
        raise RuntimeError(f"Error building survival buckets: {e}") from e
