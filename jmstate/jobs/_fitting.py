import itertools
import warnings
from collections.abc import Callable
from typing import Any, Final, cast

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import Info, Job

# Constants
NO_GROUPS_OPT: Final[tuple[type[torch.optim.Optimizer], ...]] = (torch.optim.LBFGS,)
DEFAULT_OPT_FACTORY: Final[type[torch.optim.Optimizer]] = torch.optim.Adam


class Fit(Job):
    """Job to fit the model with random effects.

    This is a fitting job. It sets the `fit_` attribute when finished.

    It can be used with any optimizer factory built on the base class
    `torch.optim.Optimizer`. If None, it defaults to Adam.

    Use kwargs to pass defaults to the optimizer factory.

    Change the value of `fit_extra` if you do not want to fit extra parameters.

    It warns the user if the parameters contain infinite or NaN values.

    Attributes:
        fit_extra (bool): Whether to set the model `fit` attribute or not.
        opt (torch.optim.Optimizer): The optimizer object.

    Examples:
        >>> Fit(torch.optim.Adam, lr=0.01)
    """

    fit_extra: bool
    opt: torch.optim.Optimizer

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __new__(
        cls,
        opt_factory: type[torch.optim.Optimizer] | None = None,
        fit_extra: bool = True,
        **kwargs: Any,
    ) -> Callable[[Info], Job]:
        """Creates the fitting class.

        Args:
            opt_factory (type[torch.optim.Optimizer] | None, optional): The optimizer
                factory, if None, it defaults to the current default set by a file
                constant. Defaults to None.
            fit_extra (bool, optional): An option to fit extra parameters or not.
                Defaults to True.
            kwargs (Any): Additional kwargs passed to the optimizer factory, such as lr.
        """
        return super().__new__(cls, opt_factory, fit_extra, **kwargs)

    def __init__(  # type: ignore
        self,
        opt_factory: type[torch.optim.Optimizer] | None = None,
        fit_extra: bool = True,
        *,
        info: Info,
        **kwargs: Any,
    ):
        """Initializes the fitting class.

        Args:
            opt_factory (type[torch.optim.Optimizer] | None, optional): The optimizer
                factory, if None, it defaults to the current default set by a file
                constant. Defaults to None.
            fit_extra (bool, optional): An option to fit extra parameters or not.
                Defaults to True.
            info (Info): The job information object.
            kwargs (Any): Additional kwargs passed to the optimizer factory, such as lr.
        """
        self.fit_extra = fit_extra

        info.model.data = info.data
        info.model.params_.requires_grad_(True)
        if self.fit_extra:
            info.model.params_.extra_requires_grad_(True)

        param_groups = [
            {
                "params": list(p.values()) if isinstance(p, dict) else [p],
                "group": name,
            }
            for name, p in info.model.params_.as_dict.items()
        ]

        extra = info.model.params_.extra
        if extra is not None and self.fit_extra:
            param_groups.append({"params": extra, "group": "extra"})

        param_list = list(
            itertools.chain.from_iterable(
                cast(torch.Tensor, g["params"]) for g in param_groups
            )
        )

        opt_factory = opt_factory or DEFAULT_OPT_FACTORY
        self.opt = opt_factory(
            param_list if opt_factory in NO_GROUPS_OPT else param_groups,
            **kwargs,
        )
        info.opt = self.opt

    def run(self, info: Info):
        """Performs a single optimization step using the optimizer.

        This calls the optimizer's `step` function, which in turn calls the `closure`
        method to compute the loss and gradients.

        Args:
            info (Info): The job information object.
        """

        def _closure():
            self.opt.zero_grad()  # type: ignore
            logpdfs, _ = info.logpdfs_aux_fn(info.model.params_, info.sampler.b)
            loss = (
                -logpdfs.mean()
                if info.model.pen is None
                else -logpdfs.mean() + info.model.pen(info.model.params_)
            )
            loss.backward()  # type: ignore
            return loss.detach()

        self.opt.step(_closure)  # type: ignore

        info.sampler.logpdfs, info.sampler.aux = info.sampler.logpdfs_aux_fn(
            info.sampler.b
        )

    def end(self, info: Info, **_kwargs: Any):
        """Ends fitting cycle.

        This resets the parameters gradient and sets the `fit_` model attribute to
        True.

        Args:
            info (Info): The job information object.
        """
        params_flat_tensor = info.model.params_.as_flat_tensor
        if (
            torch.isnan(params_flat_tensor).any()
            or torch.isinf(params_flat_tensor).any()
        ):
            warnings.warn("Error infering model parameters", stacklevel=2)

        info.model.params_.requires_grad_(False)
        if self.fit_extra:
            info.model.params_.extra_requires_grad_(False)

        info.model.fit_ = True


class Scheduling(Job):
    """Job to run a scheduler during the optimization.

    It can be used with any optimizer factory built on the base class
    `torch.optim.lr_scheduler.LRScheduler`.

    Use kwargs to pass defaults to the scheduler factory.

    Attributes:
        sched (torch.optim.lr_scheduler.LRScheduler): The scheduler object.
    """

    sched: torch.optim.lr_scheduler.LRScheduler

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __new__(
        cls,
        sched_factory: type[torch.optim.lr_scheduler.LRScheduler],
        **kwargs: Any,
    ) -> Callable[[Info], Job]:
        """Creates the scheduler.

        Args:
            sched_factory (type[torch.optim.lr_scheduler.LRScheduler]): The scheduler
                factory.
            kwargs (Any): Additional kwargs passed to the scheduler factory.
        """
        return super().__new__(cls, sched_factory, **kwargs)

    def __init__(
        self,
        sched_factory: type[torch.optim.lr_scheduler.LRScheduler],
        *,
        info: Info,
        **kwargs: Any,
    ):
        """Initializes the scheduler.

        Raises:
            ValueError: If the optimizer has not been initialized before the scheduler.

        Args:
            sched_factory (type[torch.optim.lr_scheduler.LRScheduler]): The scheduler
                factory.
            info (Info): The job information object.
            kwargs (Any): Additional kwargs passed to the scheduler factory.
        """
        if not hasattr(info, "opt"):
            raise ValueError("Optimizer must be initialized before scheduler")

        self.sched = sched_factory(info.opt, **kwargs)

    def run(self, **_kwargs: Any):
        """Performs a single scheduling step using the optimizer."""
        self.sched.step()
