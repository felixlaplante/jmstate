import itertools
from collections import defaultdict
from typing import Any, cast

import torch

from ..types._defs import BaseHazardFn, BucketData, LinkFn, Trajectory


def build_buckets(
    trajectories: list[Trajectory],
) -> dict[tuple[int, int], BucketData]:
    """Builds buckets from trajectories for user convenience.

    Args:
        trajectories (list[Trajectory]): The list of individual trajectories.


    Returns:
        dict[tuple[int, int], BucketData]: A dictionnary of transition keys with a triplet of tensors (idxs, t0, t1).
    """
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

    Returns:
        dict[tuple[int, int], tuple[torch.Tensor, ...]]: The vectorizable buckets representation.
    """
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
