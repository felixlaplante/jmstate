from typing import Any

import torch

from ..types._structures import CallbackFn


def Scheduling(
    scheduler: type[torch.optim.lr_scheduler.LRScheduler],
    scheduler_params: dict[str, Any],
) -> type[CallbackFn]:
    """Gets a scheduling callback.

    Args:
        scheduler (type[torch.optim.lr_scheduler.LRScheduler]): The desired LRScheduler.
        scheduler_params (dict[str, Any]): The scheduler parameters.

    Returns:
        CallbackFn: The callback function.
    """

    class _Scheduling(CallbackFn):
        def init(
            self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
        ) -> None:
            tmp["scheduler"] = scheduler(info["optimizer"], **scheduler_params)

        def run(
            self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
        ) -> None:
            tmp["scheduler"].step()

        def end(
            self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
        ) -> None:
            tmp.pop("scheduler")

    return _Scheduling
