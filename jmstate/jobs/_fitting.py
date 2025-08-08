import itertools
import warnings
from abc import ABC, abstractmethod
from typing import Any, Final, cast

import torch
from beartype import beartype

from ..typedefs._defs import Info, Job, Metrics, Tensor0D

# Constants
NO_GROUPS_OPT: Final[tuple[type[torch.optim.Optimizer], ...]] = (torch.optim.LBFGS,)
DEFAULT_DETERMINISTIC_OPT_FACTORY: Final[type[torch.optim.Optimizer]] = (
    torch.optim.LBFGS
)
DEFAULT_RANDOM_OPT_FACTORY: Final[type[torch.optim.Optimizer]] = torch.optim.Adam


class _BaseFit(Job, ABC):
    """Base job to fit the model."""

    opt_factory: type[torch.optim.Optimizer]
    fit_extra: bool
    opt: torch.optim.Optimizer
    kwargs: Any

    default_opt_factory: type[torch.optim.Optimizer]
    is_fitting: bool

    @beartype
    def __init__(
        self,
        opt_factory: type[torch.optim.Optimizer] | None = None,
        fit_extra: bool = True,
        **kwargs: Any,
    ):
        self.opt_factory = opt_factory or self.default_opt_factory
        self.fit_extra = fit_extra
        self.kwargs = kwargs

    def init(self, info: Info):
        info.model.data = info.data
        info.model.params_.requires_grad_(True)
        if self.fit_extra:
            info.model.params_.extra_requires_grad_(True)

        param_groups = [
            {"params": params, "group": name}
            for name, params in info.model.params_.as_groups.items()
        ]

        extra = info.model.params_.extra
        if extra is not None and self.fit_extra:
            param_groups.append({"params": extra, "group": "extra"})

        param_list = list(
            itertools.chain.from_iterable(
                cast(torch.Tensor, g["params"]) for g in param_groups
            )
        )

        self.opt = self.opt_factory(
            param_list if self.opt_factory in NO_GROUPS_OPT else param_groups,
            **self.kwargs,
        )
        info.opt = self.opt

    def run(self, info: Info):
        self.opt.step(lambda: self.closure(info))  # type: ignore

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        params_flat_tensor = info.model.params_.as_flat_tensor
        if (
            torch.isnan(params_flat_tensor).any()
            or torch.isinf(params_flat_tensor).any()
        ):
            warnings.warn("Error infering model parameters", stacklevel=2)

        info.model.params_.requires_grad_(False)
        if self.fit_extra:
            info.model.params_.extra_requires_grad_(False)

        info.model.fit_ = info.model.fit_ or self.is_fitting

    @abstractmethod
    def closure(self, info: Info) -> Tensor0D:
        pass


class DeterministicFit(_BaseFit):
    """Job to fit the model without random effects."""

    default_opt_factory: type[torch.optim.Optimizer] = DEFAULT_DETERMINISTIC_OPT_FACTORY
    is_fitting: bool = False

    def closure(self, info: Info) -> Tensor0D:
        self.opt.zero_grad()  # type: ignore
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
    is_fitting: bool = True

    def closure(self, info: Info) -> Tensor0D:
        self.opt.zero_grad()  # type: ignore
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

    sched_factory: type[torch.optim.lr_scheduler.LRScheduler]
    sched: torch.optim.lr_scheduler.LRScheduler
    kwargs: Any

    @beartype
    def __init__(
        self,
        sched_factory: type[torch.optim.lr_scheduler.LRScheduler],
        **kwargs: Any,
    ):
        self.sched_factory = sched_factory
        self.kwargs = kwargs

    def init(self, info: Info):
        if not hasattr(info, "opt"):
            raise ValueError("Optimizer must be initialized before scheduler")

        self.sched = self.sched_factory(info.opt, **self.kwargs)

    def run(self, info: Info):  # noqa: ARG002
        self.sched.step()

    def end(self, info: Info, metrics: Metrics):
        pass
