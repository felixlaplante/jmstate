import warnings
from typing import Any, cast

import torch

from ..typedefs._defs import ALPHAS_POS, Info, Job, Metrics


class Fit(Job):
    """Job to fit the model."""

    retain_graph: bool
    optimizer_factory: type[torch.optim.Optimizer]
    optimizer_params: dict[str, Any] | None

    def __init__(
        self,
        retain_graph: bool = False,
        optimizer_factory: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_params: dict[str, Any] | None = None,
    ) -> None:
        self.retain_graph = retain_graph
        self.optimizer_factory = optimizer_factory
        self.optimizer_params = optimizer_params

    def init(self, info: Info, metrics: Metrics) -> None:
        info.model.data = info.data
        info.model.params_.require_grad(True)
        info.optimizer = self.optimizer_factory(
            info.model.params_.as_list, **(self.optimizer_params or {"lr": 1e-2})
        )

    def run(self, info: Info, metrics: Metrics) -> None:
        loss = (
            -info.logpdfs.sum() + info.model.pen(info.model.params_)
            if info.model.pen is not None
            else -info.logpdfs.sum()
        )

        info.optimizer.zero_grad()  # type: ignore
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
        if not isinstance(info.extra.optimizer, torch.optim.Optimizer):
            raise ValueError("Optimizer must be initialized before scheduler")
        self.scheduler = self.scheduler_factory(
            info.extra.optimizer, **self.scheduler_params
        )

    def run(self, info: Info, metrics: Metrics) -> None:
        self.scheduler.step()

    def end(self, info: Info, metrics: Metrics) -> None:
        pass


class AdamL1Proximal(Job):
    """Job to do proximal gradient decent and variable selection."""

    group: str
    lmda: float
    offset: int

    def __init__(self, lmda: float, group: str = "betas") -> None:
        self.group = group
        self.lmda = lmda
        if group not in ("alphas", "betas"):
            raise ValueError(
                f"Group must be either 'alphas' or 'betas', got {self.group}"
            )

    def init(self, info: Info, metrics: Metrics) -> None:
        if not isinstance(info.extra.optimizer, torch.optim.Adam):
            raise ValueError("Optimizer must be set as Adam for AdamL1Proximal")
        if getattr(info.model.params_, self.group) is None:
            raise ValueError(f"{self.group} is None")

        self.offset = (
            ALPHAS_POS
            if self.group == "alphas"
            else ALPHAS_POS + len(info.model.params_.alphas)
        )

    def run(self, info: Info, metrics: Metrics) -> None:
        g = cast(torch.optim.Adam, info.extra.optimizer).param_groups[0]

        attr = getattr(info.model.params_, self.group)
        for i, key in enumerate(attr):
            p = g["params"][i + self.offset]

            if p.grad is None:
                continue

            state = cast(torch.optim.Adam, info.extra.optimizer).state[p]
            if len(state) == 0:
                continue

            eff_lr = g["lr"] / torch.sqrt(state["exp_avg_sq"] + g["eps"])

            attr[key].data = torch.clamp(
                attr[key].abs() - (self.lmda * info.data.size) * eff_lr, min=0.0
            )

    def end(self, info: Info, metrics: Metrics) -> None:
        pass
