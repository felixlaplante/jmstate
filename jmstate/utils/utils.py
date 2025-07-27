import itertools
from collections import defaultdict
from typing import Any, DefaultDict

import torch

from ._utils import flat_from_tril, tril_from_flat
from .types import Trajectory


def log_cholesky_from_flat(
    flat: torch.Tensor, n: int, method: str = "full"
) -> torch.Tensor:
    """Computes log cholesky from flat tensor according to choice of method.

    Args:
        flat (torch.Tensor): The flat tensor parameter.
        n (int): The dimension of the matrix.
        method (str, optional): The method, either for full, diagonal or isotropic covariance matrix. Defaults to "full".

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
                raise ValueError(
                    f"Inocrrect number of elements for flat, got {flat.numel()} but expected {n}"
                )
            return torch.diag(flat)
        case "ball":
            if flat.numel() != 1:
                f"Inocrrect number of elements for flat, got {flat.numel()} but expected {1}"
            return flat * torch.eye(n)
        case _:
            raise ValueError(f"Got method {method} unknown")


def flat_from_log_cholesky(L: torch.Tensor, method: str = "full") -> torch.Tensor:
    """Computes flat tensor from log cholesky matrix according to choice of method.

    Args:
        L (torch.Tensor): The square lower triangular matrix parameter.
        method (str, optional): The method, either for full, diagonal or isotropic covariance matrix. Defaults to "full".

    Raises:
        ValueError: If the method is not in ("full", "diag", "ball").

    Returns:
        torch.Tensor: The flat representation.
    """
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
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
) -> dict[tuple[int, int], tuple[torch.Tensor, ...]]:
    """Builds buckets from trajectories for user convenience.

    Args:
        trajectories (list[Trajectory]): The list of individual trajectories.

    Raises:
        RuntimeError: If the construction of the buckets fails.

    Returns:
        dict[tuple[int, int], tuple[torch.Tensor, ...]]: A dictionnary of transition keys with a triplet of tensors (idxs, t0, t1).
    """
    try:
        # Process each individual trajectory
        buckets: DefaultDict[tuple[int, int], list[list[Any]]] = defaultdict(
            lambda: [[], [], []]
        )

        for i, trajectory in enumerate(trajectories):
            for (t0, s0), (t1, s1) in itertools.pairwise(trajectory):
                key = (s0, s1)
                buckets[key][0].append(i)
                buckets[key][1].append(t0)
                buckets[key][2].append(t1)

        processed_buckets = {
            key: (
                torch.tensor(vals[0], dtype=torch.int64),
                torch.tensor(vals[1], dtype=torch.float32),
                torch.tensor(vals[2], dtype=torch.float32),
            )
            for key, vals in buckets.items()
            if vals[0]  # skip empty
        }

        return processed_buckets

    except Exception as e:
        raise RuntimeError(f"Failed to construct buckets: {e}") from e
