from typing import Any

import torch

from ..types._defs import ALPHAS_POS
from ..types._structures import CallbackFn


def AdamL1Proximal(lmda: float, group: str = "betas") -> type[CallbackFn]:
    """Gets the L1 proximal operator for Adam.

    Args:
        lmda (float): The penalty.
        group (str, optional): Must be either "alphas" or "betas". Defaults to "betas".

    Raises:
        ValueError: If the group is not alphas nor betas.

    Returns:
        CallbackFn: The L1 proximal operator.
    """
    if group not in ("alphas", "betas"):
        raise ValueError(f"Group must be either 'alphas' or 'betas', got {group}")

    class _AdamL1Proximal(CallbackFn):
        def init(
            self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
        ) -> None:
            if not isinstance(info["optimizer"], torch.optim.Adam):
                raise ValueError("Optimizer must be set as Adam for AdamL1Proximal")

            self.offset = (
                ALPHAS_POS
                if group == "alphas"
                else ALPHAS_POS + len(info["model"].model_design.surv)
            )

        def run(
            self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
        ) -> None:
            g = info["optimizer"].param_groups[0]

            attr = getattr(info["params"], group)
            for i, key in enumerate(attr):
                p = g["params"][i + self.offset]

                if p.grad is None:
                    continue

                state = info["optimizer"].state[p]
                if len(state) == 0:
                    continue

                effective_lr = g["lr"] / torch.sqrt(state["exp_avg_sq"] + g["eps"])

                attr[key].data = torch.clamp(
                    attr[key].abs() - lmda * effective_lr, min=0.0
                )

        def end(
            self, info: dict[str, Any], metrics: dict[str, Any], tmp: dict[str, Any]
        ) -> None:
            pass

    return _AdamL1Proximal
