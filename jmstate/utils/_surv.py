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
            idxs, t0s, t1s = buckets[(s0, s1)]
            idxs.append(i)
            t0s.append(t0)
            t1s.append(t1)

    result = {
        key: BucketData(
            torch.frombuffer(idxs, dtype=torch.int64),
            torch.frombuffer(t0s, dtype=dtype).reshape(-1, 1),
            torch.frombuffer(t1s, dtype=dtype).reshape(-1, 1),
        )
        for key, (idxs, t0s, t1s) in buckets.items()
    }

    return dict(sorted(result.items(), key=lambda kv: str(kv[0])))


@lru_cache
def _build_alt_map(
    surv_keys: tuple[tuple[Any, Any], ...],
) -> defaultdict[Any, tuple[tuple[Any, Any], ...]]:
    """Builds alternative state mapping as tuples in a defaultdict.

    Args:
        surv_keys (tuple[tuple[Any, Any], ...]): The survival keys.

    Returns:
        defaultdict[Any, tuple[tuple[Any, Any], ...]]: The alternative state map.
    """
    return defaultdict(
        lambda: (),
        {
            s0: tuple((k, v) for k, v in surv_keys if k == s0)
            for s0 in {s0 for s0, _ in surv_keys}
        },
    )


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
                idxs, t0s, t1s, obs = buckets[key]
                idxs.append(i)
                t0s.append(t0)
                t1s.append(t1)
                obs.append(key[1] == s1)

        (last_t, last_s), c_i = trajectory[-1], c[i].item()

        if last_t >= c_i:
            continue

        for key in alt_map[last_s]:
            idxs, t0s, t1s, obs = buckets[key]
            idxs.append(i)
            t0s.append(last_t)
            t1s.append(c_i)
            obs.append(False)

    return {
        key: (
            torch.frombuffer(idxs, dtype=torch.int64),
            torch.frombuffer(t0s, dtype=dtype).reshape(-1, 1),
            torch.frombuffer(t1s, dtype=dtype).reshape(-1, 1),
            torch.frombuffer(obs, dtype=torch.bool),
        )
        for key, (idxs, t0s, t1s, obs) in buckets.items()
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
            idxs, t0s = buckets[key]
            idxs.append(i)
            t0s.append(last_t)

    return {
        key: (
            idxs_tensor := torch.frombuffer(idxs, dtype=torch.int64),
            torch.frombuffer(t0s, dtype=dtype).reshape(-1, 1),
            c.index_select(0, idxs_tensor),
        )
        for key, (idxs, t0s) in buckets.items()
    }


def build_remaining_buckets(
    trajectories: list[Trajectory],
    surv_keys: tuple[tuple[Any, Any], ...],
) -> dict[tuple[Any, Any], tuple[torch.Tensor, ...]]:
    """Build remaining bucket representation.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        surv_keys (tuple[tuple[Any, Any], ...]): The survival keys.

    Returns:
        dict[tuple[Any, Any], tuple[torch.Tensor, ...]]: The remaining buckets
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
        for key in alt_map[last_s]:
            idxs, t1s = buckets[key]
            idxs.append(i)
            t1s.append(last_t)

    return {
        key: (
            torch.frombuffer(idxs, dtype=torch.int64),
            torch.frombuffer(t1s, dtype=dtype).reshape(-1, 1),
        )
        for key, (idxs, t1s) in buckets.items()
    }
