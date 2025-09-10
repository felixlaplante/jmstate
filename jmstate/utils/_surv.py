import itertools
from collections import defaultdict
from functools import lru_cache
from typing import Any

import torch

from ..typedefs._defs import BucketData, Trajectory, TrajRepr


def build_buckets(
    trajectories: list[Trajectory],
) -> dict[tuple[Any, Any], BucketData]:
    """Builds buckets from trajectories for user convenience.

    This yeilds a `NamedTuple` containing transition information containing:
        idxs (Tensor1D): The individual indices.
        t0 (TensorCol): A column vector of previous transition times.
        t1 (TensorCol): A column vecotr of next transition times.

    Args:
        trajectories (list[Trajectory]): The list of individual trajectories.

    Returns:
        dict[tuple[Any, Any], BucketData]: Transition keys with values (idxs, t0, t1).
    """
    # Process each individual trajectory
    accumulator: defaultdict[tuple[Any, Any], list[tuple[int, float, float]]] = (
        defaultdict(list)
    )

    for i, trajectory in enumerate(trajectories):
        for (t0, s0), (t1, s1) in itertools.pairwise(trajectory):
            accumulator[(s0, s1)].append((i, t0, t1))

    buckets = {
        key: tuple(zip(*vals, strict=False)) for key, vals in accumulator.items()
    }

    return {
        key: BucketData(
            torch.tensor(vals[0], dtype=torch.int64),
            torch.tensor(vals[1]).view(-1, 1),
            torch.tensor(vals[2]).view(-1, 1),
        )
        for key, vals in buckets.items()
    }


@lru_cache
def _build_alt_map(
    surv_keys: tuple[tuple[Any, Any], ...],
) -> dict[Any, list[tuple[Any, Any]]]:
    """Builds alternative state mapping.

    Args:
        surv_keys (tuple[tuple[Any, Any], ...]): The survival keys.

    Returns:
        dict[Any, list[tuple[Any, Any]]]: The alternative state mapping.
    """
    alt_map: dict[Any, list[tuple[Any, Any]]] = defaultdict(list)
    for s0, s1 in surv_keys:
        alt_map[s0].append((s0, s1))

    return alt_map


def build_traj_repr(
    trajectories: list[Trajectory],
    c: torch.Tensor,
    surv_keys: tuple[tuple[Any, Any], ...],
) -> dict[tuple[Any, Any], TrajRepr]:
    """Build vectorizable bucket representation.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor): Censoring times.
        surv_keys (tuple[tuple[Any, Any], ...]): The survival keys.

    Returns:
        dict[tuple[Any, Any], TrajRepr]: The vectorizable buckets representation.
    """
    alt_map = _build_alt_map(surv_keys)

    # Initialize buckets
    accumulator: dict[tuple[Any, Any], list[tuple[int, float, float, bool]]] = (
        defaultdict(list)
    )

    # Process each individual trajectory
    for i, trajectory in enumerate(trajectories):
        for (t0, s0), (t1, s1) in itertools.pairwise(trajectory):
            for key in alt_map[s0]:
                accumulator[key].append((i, t0, t1, key[1] == s1))

        (last_t, last_s), c_i = trajectory[-1], c[i].item()

        if last_t >= c_i:
            continue

        for key in alt_map[last_s]:
            accumulator[key].append((i, last_t, c_i, False))

    buckets = {
        key: tuple(zip(*vals, strict=False)) for key, vals in accumulator.items()
    }

    return {
        key: TrajRepr(
            torch.tensor(vals[0], dtype=torch.int64),
            torch.tensor(vals[1]).view(-1, 1),
            torch.tensor(vals[2]).view(-1, 1),
            torch.tensor(vals[3], dtype=torch.bool),
        )
        for key, vals in buckets.items()
    }
