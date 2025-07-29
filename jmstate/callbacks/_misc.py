from typing import Any

import torch

from ..types._structures import CallbackFn


def call_callbacks(
    method: str,
    callbacks: CallbackFn | list[CallbackFn] | None,
    info: dict[str, Any],
    metrics: dict[str, Any],
    tmp: dict[str, Any],
):
    """Init one or multiple functions.

    Args:
        callbacks (CallbackFn | list[CallbackFn] | None): The function(s) to call, or None.
    """
    if callbacks is None:
        return
    if isinstance(callbacks, CallbackFn):
        callbacks = [callbacks]
    with torch.no_grad():
        match method:
            case "init":
                for callback in callbacks:
                    callback.init(info=info, metrics=metrics, tmp=tmp)
            case "run":
                for callback in callbacks:
                    callback.run(info=info, metrics=metrics, tmp=tmp)
            case "end":
                for callback in callbacks:
                    callback.end(info=info, metrics=metrics, tmp=tmp)
            case _:
                pass
