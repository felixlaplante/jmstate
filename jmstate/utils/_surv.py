import itertools
from collections import defaultdict
from typing import Any, cast

import torch

from ..typedefs._defs import (
    BucketData,
    HazardFns,
    Trajectory,
    TrajRepr,
)


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
    buckets: defaultdict[tuple[Any, Any], list[list[Any]]] = defaultdict(
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
            torch.tensor(vals[1]).view(-1, 1),
            torch.tensor(vals[2]).view(-1, 1),
        )
        for key, vals in buckets.items()
        if vals != []
    }


def build_traj_repr(
    trajectories: list[Trajectory],
    c: torch.Tensor,
    surv: dict[tuple[Any, Any], HazardFns],
) -> dict[tuple[Any, Any], TrajRepr]:
    """Build vectorizable bucket representation.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor): Censoring times.
        surv (dict[tuple[Any, Any], tuple[BaseHazardFn, LinkFn]]) : The survival dict.

    Raises:
        ValueError: If some keys are not in surv.

    Returns:
        dict[tuple[Any, Any], TrajRepr]: The vectorizable buckets representation.
    """
    # Get survival transitions defined in the model
    trans = set(surv.keys())

    # Build alternative state mapping
    alt_map: defaultdict[int, list[int]] = defaultdict(list)
    for from_state, to_state in trans:
        alt_map[from_state].append(to_state)

    # Initialize buckets
    buckets: dict[tuple[Any, Any], list[list[Any]]] = defaultdict(
        lambda: [[], [], [], []]
    )

    # Process each individual trajectory
    for i, trajectory in enumerate(trajectories):
        # Add censoring
        ext_trajectory = [*trajectory, (c[i].item(), None)]

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
        key: TrajRepr(
            torch.tensor(vals[0], dtype=torch.int64),
            torch.tensor(vals[1]).view(-1, 1),
            torch.tensor(vals[2]).view(-1, 1),
            torch.tensor(vals[3], dtype=torch.bool),
        )
        for key, vals in buckets.items()
        if vals != []
    }
