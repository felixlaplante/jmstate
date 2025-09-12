import itertools
from array import array
from collections import defaultdict
from functools import lru_cache
from typing import Any

import torch

from ..typedefs._defs import BucketData, Trajectory
from ..utils._dtype import get_dtype


def build_buckets(
    trajectories: list[Trajectory],
) -> dict[tuple[Any, Any], BucketData]:
    """Builds buckets from trajectories for user convenience.

    This yeilds a `NamedTuple` containing transition information containing:
        idxs (Tensor1D): The individual indices.
        t0 (TensorCol): A column vector of previous transition times.
        t1 (TensorCol): A column vector of next transition times.

    Args:
        trajectories (list[Trajectory]): The list of individual trajectories.

    Returns:
        dict[tuple[Any, Any], BucketData]: Transition keys with values (idxs, t0, t1).
    """
    dtype = get_dtype()
    typecode = "f" if dtype == torch.float32 else "d"

    # Process each individual trajectory
    buckets: defaultdict[
        tuple[Any, Any], tuple[array[int], array[float], array[float]]
    ] = defaultdict(lambda: (array("q"), array(typecode), array(typecode)))

    for i, trajectory in enumerate(trajectories):
        for (t0, s0), (t1, s1) in itertools.pairwise(trajectory):
            buckets[(s0, s1)][0].append(i)
            buckets[(s0, s1)][1].append(t0)
            buckets[(s0, s1)][2].append(t1)

    return {
        key: BucketData(
            torch.frombuffer(vals[0], dtype=torch.int64),
            torch.frombuffer(vals[1], dtype=dtype).view(-1, 1),
            torch.frombuffer(vals[2], dtype=dtype).view(-1, 1),
        )
        for key, vals in buckets.items()
    }


@lru_cache
def _build_alt_map(
    surv_keys: tuple[tuple[Any, Any], ...],
) -> defaultdict[Any, list[tuple[Any, Any]]]:
    """Builds alternative state mapping.

    Args:
        surv_keys (tuple[tuple[Any, Any], ...]): The survival keys.

    Returns:
        defaultdict[Any, list[tuple[Any, Any]]]: The alternative state mapping.
    """
    alt_map: defaultdict[Any, list[tuple[Any, Any]]] = defaultdict(list)
    for s0, s1 in surv_keys:
        alt_map[s0].append((s0, s1))

    return alt_map


def build_all_buckets(
    trajectories: list[Trajectory],
    c: torch.Tensor,
    surv_keys: tuple[tuple[Any, Any], ...],
) -> dict[tuple[Any, Any], tuple[torch.Tensor, ...]]:
    """Build vectorizable bucket representation.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor): Censoring times.
        surv_keys (tuple[tuple[Any, Any], ...]): The survival keys.

    Returns:
        dict[tuple[Any, Any], tuple[torch.Tensor, ...]]: The vectorizable buckets
            representation.
    """
    alt_map = _build_alt_map(surv_keys)
    dtype = get_dtype()
    typecode = "f" if dtype == torch.float32 else "d"

    # Initialize buckets
    buckets: defaultdict[
        tuple[Any, Any], tuple[array[int], array[float], array[float], array[bool]]
    ] = defaultdict(lambda: (array("q"), array(typecode), array(typecode), array("b")))

    # Process each individual trajectory
    for i, trajectory in enumerate(trajectories):
        for (t0, s0), (t1, s1) in itertools.pairwise(trajectory):
            for key in alt_map[s0]:
                buckets[key][0].append(i)
                buckets[key][1].append(t0)
                buckets[key][2].append(t1)
                buckets[key][3].append(key[1] == s1)

        (last_t, last_s), c_i = trajectory[-1], c[i].item()

        if last_t >= c_i:
            continue

        for key in alt_map[last_s]:
            buckets[key][0].append(i)
            buckets[key][1].append(last_t)
            buckets[key][2].append(c_i)
            buckets[key][3].append(False)

    return {
        key: (
            torch.frombuffer(vals[0], dtype=torch.int64),
            torch.frombuffer(vals[1], dtype=dtype).view(-1, 1),
            torch.frombuffer(vals[2], dtype=dtype).view(-1, 1),
            torch.frombuffer(vals[3], dtype=torch.bool),
        )
        for key, vals in buckets.items()
    }


def build_possible_buckets(
    trajectories: list[Trajectory],
    c: torch.Tensor,
    surv_keys: tuple[tuple[Any, Any], ...],
) -> dict[tuple[Any, Any], tuple[torch.Tensor, ...]]:
    """Build possible bucket representation.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor): Censoring times.
        surv_keys (tuple[tuple[Any, Any], ...]): The survival keys.

    Returns:
        dict[tuple[Any, Any], tuple[torch.Tensor, ...]]: The possible buckets
            representation.
    """
    alt_map = _build_alt_map(surv_keys)
    dtype = get_dtype()
    typecode = "f" if dtype == torch.float32 else "d"

    # Initialize buckets
    buckets: defaultdict[tuple[Any, Any], tuple[array[int], array[float]]] = (
        defaultdict(lambda: (array("q"), array(typecode)))
    )

    # Process each individual trajectory
    for i, trajectory in enumerate(trajectories):
        last_t, last_s = trajectory[-1]

        if last_t >= c[i].item():
            continue

        for key in alt_map[last_s]:
            buckets[key][0].append(i)
            buckets[key][1].append(last_t)

    return {
        key: (
            idxs := torch.frombuffer(vals[0], dtype=torch.int64),
            torch.frombuffer(vals[1], dtype=dtype).view(-1, 1),
            c.index_select(0, idxs),
        )
        for key, vals in buckets.items()
    }
