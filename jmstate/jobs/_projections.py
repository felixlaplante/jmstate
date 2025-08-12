from abc import ABC, abstractmethod
from typing import Any, Final

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import Info, Job, Metrics, NumNonNegative

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
    def __init__(self, lmda: NumNonNegative, group: str = "betas", *, info: Info):
        """Initialize the proximal operator.

        Args:
            lmda (NumNonNegative): The penalty.
            group (str, optional): The group to penalize, either the link or covariate
                parameters. Defaults to "betas".
            info (Info): The job information object.

        Raises:
            ValueError: If the optimizer has not been initialized before the proximal.
            ValueError: If the group is not in `alphas` nor `betas`.
            ValueError: If the group is not optimized by the optimizer.
        """
        self.group = group
        self.lmda = lmda
        if group not in ("alphas", "betas"):
            raise ValueError(
                f"Group must be either 'alphas' or 'betas', got {self.group}"
            )

        if not hasattr(info, "opt"):
            raise ValueError("Optimizer must be initialized before proximal job")
        if getattr(info.model.params_, self.group) is None:
            raise ValueError(f"{self.group} is None")

        self.check_optimizer(info.opt)

        self.param_groups = [
            g for g in info.opt.param_groups if g.get("group") == self.group
        ]
        if not self.param_groups:
            raise ValueError(f"Optimizer does not optimize group {self.group}")

    def run(self, info: Info):
        """Projects the current parameter value.

        Args:
            info (Info): The job information object.
        """
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
        """Empty method.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The metrics information object.
        """

    @staticmethod
    @abstractmethod
    def check_optimizer(optimizer: torch.optim.Optimizer):
        """Checks if the optimizer is supported.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer object.
        """

    @staticmethod
    @abstractmethod
    def get_effective_lr(g: dict[str, Any], state: dict[str, Any]) -> float:
        """Get the effective learning rate.

        Args:
            g (dict[str, Any]): The parameter group/
            state (dict[str, Any]): The optimizer state.
        """


class AdamL1Proximal(_BaseL1Proximal):
    """Adam proximal operator.

    Args:
        lmda (NumNonNegative): The penalty.
        group (str, optional): The group to penalize, either the link or covariate
            parameters. Defaults to "betas".

    Raises:
        ValueError: If the optimizer is not supported.
    """

    @staticmethod
    def check_optimizer(optimizer: torch.optim.Optimizer):
        """Checks if the optimizer is supported.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer object.

        Raises:
            ValueError: If the optimizer is not Adam-like.
        """
        if not isinstance(optimizer, ADAM_LIKE):
            raise ValueError("Optimizer must be Adam or Adam-like")

    @staticmethod
    def get_effective_lr(g: dict[str, Any], state: dict[str, Any]) -> float:
        """Gets the effective learning rate.

        Args:
            g (dict[str, Any]): The parameter group.
            state (dict[str, Any]): The optimizer state.

        Returns:
            float: The effective learning rate.
        """
        return g["lr"] / torch.sqrt(state["exp_avg_sq"] + g["eps"])
