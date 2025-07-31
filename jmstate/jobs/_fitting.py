import warnings
from typing import Any

import torch

from ..typedefs._defs import Info, Job, Metrics


class Fit(Job):
    """Job to fit the model."""

    optimizer_factory: type[torch.optim.Optimizer]
    optimizer_params: dict[str, Any] | None
    retain_graph: bool

    def __init__(
        self,
        optimizer_factory: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_params: dict[str, Any] | None = None,
        retain_graph: bool = False,
    ) -> None:
        self.optimizer_factory = optimizer_factory
        self.optimizer_params = optimizer_params
        self.retain_graph = retain_graph

    def init(self, info: Info, metrics: Metrics) -> None:
        info.model.data = info.data
        info.model.params_.require_grad(True)
        param_groups = [
            {
                "params": val,
                "group": key,
            }
            for key, val in info.model.params_.as_groups.items()
        ]
        info.optimizer = self.optimizer_factory(
            param_groups,
            **(self.optimizer_params or {"lr": 0.01}),
        )

    def run(self, info: Info, metrics: Metrics) -> None:
        loss = (
            -info.logpdfs.mean() + info.model.pen(info.model.params_)
            if info.model.pen is not None
            else -info.logpdfs.mean()
        )

        info.optimizer.zero_grad(set_to_none=True)  # type: ignore
        loss.backward(retain_graph=self.retain_graph)  # type: ignore
        info.optimizer.step()  # type: ignore

    def end(self, info: Info, metrics: Metrics) -> None:
        params_flat_tensor = info.model.params_.as_flat_tensor
        if (
            torch.isnan(params_flat_tensor).any()
            or torch.isinf(params_flat_tensor).any()
        ):
            warnings.warn("Error infering model parameters", stacklevel=2)

        info.model.params_.require_grad(False)
        info.model.fit_ = True


class Scheduling(Job):
    """Job to run a scheduler during the optimization."""

    scheduler_factory: type[torch.optim.lr_scheduler.LRScheduler]
    scheduler_params: dict[str, Any]
    scheduler: torch.optim.lr_scheduler.LRScheduler

    def __init__(
        self,
        scheduler_factory: type[torch.optim.lr_scheduler.LRScheduler],
        scheduler_params: dict[str, Any],
    ) -> None:
        self.scheduler_factory = scheduler_factory
        self.scheduler_params = scheduler_params

    def init(self, info: Info, metrics: Metrics) -> None:
        if not hasattr(info, "optimizer"):
            raise ValueError("Optimizer must be initialized before scheduler")
        self.scheduler = self.scheduler_factory(info.optimizer, **self.scheduler_params)

    def run(self, info: Info, metrics: Metrics) -> None:
        self.scheduler.step()

    def end(self, info: Info, metrics: Metrics) -> None:
        pass


class AdamL1Proximal(Job):
    """Job to do proximal gradient decent and variable selection."""

    group: str
    lmda: float
    param_groups: list[dict[str, Any]]

    def __init__(self, lmda: float, group: str = "betas") -> None:
        self.group = group
        self.lmda = lmda
        if group not in ("alphas", "betas"):
            raise ValueError(
                f"Group must be either 'alphas' or 'betas', got {self.group}"
            )

    def init(self, info: Info, metrics: Metrics) -> None:
        if not hasattr(info, "optimizer"):
            raise ValueError("Optimizer must be initialized before AdamL1Proximal")
        if not isinstance(info.optimizer, torch.optim.Adam):
            raise ValueError("Optimizer must be set as Adam for AdamL1Proximal")
        if getattr(info.model.params_, self.group) is None:
            raise ValueError(f"{self.group} is None")

        self.param_groups = [
            g for g in info.optimizer.param_groups if g.get("group") == self.group
        ]

    def run(self, info: Info, metrics: Metrics) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue

                state = info.optimizer.state[p]
                if len(state) == 0:
                    continue

                eff_lr = g["lr"] / torch.sqrt(state["exp_avg_sq"] + g["eps"])
                p.data = torch.clamp(p.abs() - self.lmda * eff_lr, min=0.0)

    def end(self, info: Info, metrics: Metrics) -> None:
        pass
