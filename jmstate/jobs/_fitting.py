import warnings
from typing import Any, Final

import torch
from beartype import beartype

from ..typedefs._defs import Info, Job, Metrics

# Constants
DEFAULT_OPT_FACTORY: Final[type[torch.optim.Optimizer]] = torch.optim.Adam
DEFAULT_OPT_KWARGS: Final[dict[str, Any]] = {"lr": 0.1, "fused": True}
NO_GROUPS_OPT: Final[tuple[type[torch.optim.Optimizer], ...]] = (torch.optim.LBFGS,)


class Fit(Job):
    """Job to fit the model."""

    optimizer_factory: type[torch.optim.Optimizer]
    optimizer: torch.optim.Optimizer
    kwargs: Any

    @beartype
    def __init__(
        self,
        optimizer_factory: type[torch.optim.Optimizer] = DEFAULT_OPT_FACTORY,
        **kwargs: Any,
    ):
        self.optimizer_factory = optimizer_factory

        if self.optimizer_factory == DEFAULT_OPT_FACTORY:
            self.kwargs = {
                **DEFAULT_OPT_KWARGS,
                **kwargs,
            }
        else:
            self.kwargs = kwargs

    def init(self, info: Info):
        info.model.data = info.data
        info.model.params_.requires_grad_(True)

        parameters = (
            [
                {
                    "params": val,
                    "group": key,
                }
                for key, val in info.model.params_.as_groups.items()
            ]
            if self.optimizer_factory not in NO_GROUPS_OPT
            else info.model.params_.as_list
        )

        self.optimizer = self.optimizer_factory(
            parameters,
            **self.kwargs,
        )
        info.optimizer = self.optimizer

    def run(self, info: Info):
        def closure():
            self.optimizer.zero_grad(set_to_none=True)  # type: ignore
            logpdfs = info.logpdfs_fn(info.model.params_, info.b)
            loss = (
                -logpdfs.mean() + info.model.pen(info.model.params_)
                if info.model.pen is not None
                else -logpdfs.mean()
            )
            loss.backward()  # type: ignore
            return loss

        self.optimizer.step(closure)  # type: ignore

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        params_flat_tensor = info.model.params_.as_flat_tensor
        if (
            torch.isnan(params_flat_tensor).any()
            or torch.isinf(params_flat_tensor).any()
        ):
            warnings.warn("Error infering model parameters", stacklevel=2)

        info.model.params_.requires_grad_(False)
        info.model.fit_ = True


class Scheduling(Job):
    """Job to run a scheduler during the optimization."""

    scheduler_factory: type[torch.optim.lr_scheduler.LRScheduler]
    scheduler: torch.optim.lr_scheduler.LRScheduler
    kwargs: Any

    @beartype
    def __init__(
        self,
        scheduler_factory: type[torch.optim.lr_scheduler.LRScheduler],
        **kwargs: Any,
    ):
        self.scheduler_factory = scheduler_factory
        self.kwargs = kwargs

    def init(self, info: Info):
        if not hasattr(info, "optimizer"):
            raise ValueError("Optimizer must be initialized before scheduler")

        self.scheduler = self.scheduler_factory(info.optimizer, **self.kwargs)

    def run(self, info: Info):  # noqa: ARG002
        self.scheduler.step()

    def end(self, info: Info, metrics: Metrics):
        pass
