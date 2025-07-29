from typing import Any

import torch

from ..types._structures import CallbackFn


def L1Proximal(lmda: float, group: str = "betas") -> type[CallbackFn]:
    """Gets the L1 proximal operator.

    Args:
        lmda (float): The penalty.
        group (str, optional): Either alphas or betas. Defaults to "betas".

    Raises:
        ValueError: If the group is not alphas nor betas.

    Returns:
        CallbackFn: The L1 proximal operator.
    """
    if group not in ("alphas", "betas"):
        raise ValueError(f"Group must be either 'alphas' or 'betas', got {group}")

    class _L1Proximal(CallbackFn):
        def init(
            self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
        ) -> None:
            pass

        def run(
            self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
        ) -> None:
            attr = getattr(info["params"], group)
            for key, val in attr.items():
                attr[key] = torch.sign(val) * torch.clamp(val.abs() - lmda, min=0.0)

        def end(
            self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
        ) -> None:
            pass

    return _L1Proximal
