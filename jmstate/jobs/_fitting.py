import itertools
import warnings
from abc import ABC, abstractmethod
from typing import Any, Final, cast

import torch
from beartype import beartype

from ..typedefs._defs import Info, Job, Metrics, Tensor0D

# Constants
NO_GROUPS_OPT: Final[tuple[type[torch.optim.Optimizer], ...]] = (torch.optim.LBFGS,)
DEFAULT_DETERMINISTIC_OPT_FACTORY: Final[type[torch.optim.Optimizer]] = torch.optim.Adam
DEFAULT_RANDOM_OPT_FACTORY: Final[type[torch.optim.Optimizer]] = torch.optim.Adam
DEFAULT_DETERMINISTIC_KWARGS: Final[dict[str, Any]] = {"lr": 0.1, "fused": True}
DEFAULT_RANDOM_KWARGS: Final[dict[str, Any]] = {"lr": 0.01, "fused": True}


class _BaseFit(Job, ABC):
    """Base job to fit the model."""

    optimizer_factory: type[torch.optim.Optimizer]
    optimizer: torch.optim.Optimizer
    kwargs: Any

    default_opt_factory: type[torch.optim.Optimizer]
    default_kwargs: Any
    is_fitting: bool

    @beartype
    def __init__(
        self,
        optimizer_factory: type[torch.optim.Optimizer] | None = None,
        **kwargs: Any,
    ):
        self.optimizer_factory = optimizer_factory or self.default_opt_factory

        if optimizer_factory is None:
            self.kwargs = {
                **self.default_kwargs,
                **kwargs,
            }
        else:
            self.kwargs = kwargs

    def init(self, info: Info):
        info.model.data = info.data
        info.model.params_.requires_grad_(True)

        param_groups = [
            {
                "params": val,
                "group": key,
            }
            for key, val in info.model.params_.as_groups.items()
        ]

        param_list = list(
            itertools.chain.from_iterable(
                cast(torch.Tensor, g["params"]) for g in param_groups
            )
        )

        self.optimizer = self.optimizer_factory(
            param_list if self.optimizer_factory in NO_GROUPS_OPT else param_groups,
            **self.kwargs,
        )
        info.optimizer = self.optimizer

    def run(self, info: Info):
        self.optimizer.step(lambda: self.closure(info))  # type: ignore

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        params_flat_tensor = info.model.params_.as_flat_tensor
        if (
            torch.isnan(params_flat_tensor).any()
            or torch.isinf(params_flat_tensor).any()
        ):
            warnings.warn("Error infering model parameters", stacklevel=2)

        info.model.params_.requires_grad_(False)
        info.model.fit_ = info.model.fit_ or self.is_fitting

    @abstractmethod
    def closure(self, info: Info) -> Tensor0D:
        pass


class DeterministicFit(_BaseFit):
    """Job to fit the model without random effects."""

    default_opt_factory: type[torch.optim.Optimizer] = DEFAULT_DETERMINISTIC_OPT_FACTORY
    default_kwargs: Any = DEFAULT_DETERMINISTIC_KWARGS
    is_fitting: bool = False

    def closure(self, info: Info) -> Tensor0D:
        self.optimizer.zero_grad()  # type: ignore
        logliks = info.logliks_fn(info.model.params_, info.b)
        loss = (
            -logliks.mean() + info.model.pen(info.model.params_)
            if info.model.pen is not None
            else -logliks.mean()
        )
        loss.backward()  # type: ignore
        return loss


class RandomFit(_BaseFit):
    """Job to fit the model with random effects."""

    default_opt_factory: type[torch.optim.Optimizer] = DEFAULT_RANDOM_OPT_FACTORY
    default_kwargs: Any = DEFAULT_RANDOM_KWARGS
    is_fitting: bool = True

    def closure(self, info: Info) -> Tensor0D:
        self.optimizer.zero_grad()  # type: ignore
        logpdfs = info.logpdfs_fn(info.model.params_, info.b)
        loss = (
            -logpdfs.mean() + info.model.pen(info.model.params_)
            if info.model.pen is not None
            else -logpdfs.mean()
        )
        loss.backward()  # type: ignore
        return loss


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
