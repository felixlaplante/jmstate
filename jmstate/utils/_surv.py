import itertools
from collections import defaultdict
from collections.abc import KeysView
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
        defaultdict(lambda: [])
    )

    for i, trajectory in enumerate(trajectories):
        for (t0, s0), (t1, s1) in itertools.pairwise(trajectory):
            key = (s0, s1)
            accumulator[key].append((i, t0, t1))

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


def build_traj_repr(
    trajectories: list[Trajectory],
    c: torch.Tensor,
    surv_keys: KeysView[tuple[Any, Any]],
) -> dict[tuple[Any, Any], TrajRepr]:
    """Build vectorizable bucket representation.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor): Censoring times.
        surv_keys (KeysView[tuple[Any, Any]]): The survival keys.

    Returns:
        dict[tuple[Any, Any], TrajRepr]: The vectorizable buckets representation.
    """
    # Build alternative state mapping
    alt_map: defaultdict[Any, list[Any]] = defaultdict(list)
    for from_state, to_state in surv_keys:
        alt_map[from_state].append(to_state)

    # Initialize buckets
    accumulator: dict[tuple[Any, Any], list[tuple[int, float, float, bool]]] = (
        defaultdict(lambda: [])
    )

    # Process each individual trajectory
    for i, trajectory in enumerate(trajectories):
        # Add censoring
        ext_trajectory = [*trajectory, (c[i].item(), None)]

        for (t0, s0), (t1, s1) in itertools.pairwise(ext_trajectory):
            if t0 >= t1:
                continue

            for alt_state in alt_map[s0]:
                key = (s0, alt_state)
                accumulator[key].append((i, t0, t1, alt_state == s1))

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
