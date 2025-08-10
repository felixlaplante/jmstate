from abc import ABC, abstractmethod
from typing import Any, Final

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import Info, Job, Metrics, NumPositive

ADAM_LIKE: Final[tuple[type[torch.optim.Optimizer], ...]] = (
    torch.optim.Adam,
    torch.optim.AdamW,
    torch.optim.NAdam,
)


class _BaseL1Proximal(Job, ABC):
    group: str
    lmda: int | float
    param_groups: list[dict[str, Any]]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, lmda: NumPositive, group: str = "betas"):
        self.group = group
        self.lmda = lmda
        if group not in ("alphas", "betas"):
            raise ValueError(
                f"Group must be either 'alphas' or 'betas', got {self.group}"
            )

    def init(self, info: Info):
        if not hasattr(info, "optimizer"):
            raise ValueError("Optimizer must be initialized before proximal job")
        if getattr(info.model.params_, self.group) is None:
            raise ValueError(f"{self.group} is None")

        self.check_optimizer(info.optimizer)

        self.param_groups = [
            g for g in info.optimizer.param_groups if g.get("group") == self.group
        ]
        if not self.param_groups:
            raise ValueError(f"Optimizer does not optimize group {self.group}")

    def run(self, info: Info):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue

                state = info.optimizer.state[p]
                if len(state) == 0:
                    continue

                eff_lr = self.get_effective_lr(g, state)
                p.data = p.sign() * torch.clamp(p.abs() - self.lmda * eff_lr, min=0.0)

    def end(self, info: Info, metrics: Metrics):
        pass

    @staticmethod
    @abstractmethod
    def check_optimizer(optimizer: torch.optim.Optimizer): ...

    @staticmethod
    @abstractmethod
    def get_effective_lr(g: dict[str, Any], state: dict[str, Any]) -> float: ...


class AdamL1Proximal(_BaseL1Proximal):
    @staticmethod
    def check_optimizer(optimizer: torch.optim.Optimizer):
        if not isinstance(optimizer, ADAM_LIKE):
            raise ValueError("Optimizer must be Adam or Adam-like")

    @staticmethod
    def get_effective_lr(g: dict[str, Any], state: dict[str, Any]):
        return g["lr"] / torch.sqrt(state["exp_avg_sq"] + g["eps"])
