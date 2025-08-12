import itertools
import warnings
from abc import ABC, abstractmethod
from typing import Any, Final, cast

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import Info, Job, Metrics

# Constants
NO_GROUPS_OPT: Final[tuple[type[torch.optim.Optimizer], ...]] = (torch.optim.LBFGS,)
DEFAULT_DETERMINISTIC_OPT_FACTORY: Final[type[torch.optim.Optimizer]] = torch.optim.Adam
DEFAULT_RANDOM_OPT_FACTORY: Final[type[torch.optim.Optimizer]] = torch.optim.Adam


class _BaseFit(Job, ABC):
    """Base job to fit the model."""

    fit_extra: bool
    opt: torch.optim.Optimizer

    default_opt_factory: type[torch.optim.Optimizer]
    is_fitting: bool

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        opt_factory: type[torch.optim.Optimizer] | None = None,
        fit_extra: bool = True,
        *,
        info: Info,
        **kwargs: Any,
    ):
        """Initializes the fitting base class.

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

        opt_factory = opt_factory or self.default_opt_factory
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
        self.opt.step(lambda: self.closure(info))  # type: ignore

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        """Ends fitting cycle.

        This resets the parameters gradient and sets the `fit_` model attribute to
        the default attribute value.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The job metrics object.
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

        info.model.fit_ = info.model.fit_ or self.is_fitting

    @abstractmethod
    def closure(self, info: Info) -> torch.Tensor:
        """Implements closure for the optimizer.

        Args:
            info (Info): The job information object.

        Returns:
            torch.Tensor: The loss value.
        """


class DeterministicFit(_BaseFit):
    """Job to fit the model without random effects.

    This is a deterministic fitting job, but it does not set the `fit_` attribute as it
    does not take random effects into account. This job helps to find probably good
    initial values for the paramaters.

    It can be used with any optimizer factory built on the base class
    `torch.optim.Optimizer`. If None, it defaults to Adam.

    Use kwargs to pass defaults to the optimizer factory.

    Change the value of `fit_extra` if you do not want to fit extra parameters.

    It warns the user if the paramaters contain infinite or NaN values.

    Please not its use is discouraged when using in conjunction with a neural network.

    Attributes:
        fit_extra (bool): Whether to set the model `fit` attribute or not.
        opt (torch.optim.Optimizer): The optimizer object.
        is_fitting (bool): False.

    Examples:
        >>> DeterministicFit(torch.optim.Adam, lr=0.01)
    """

    default_opt_factory: type[torch.optim.Optimizer] = DEFAULT_DETERMINISTIC_OPT_FACTORY
    is_fitting: bool = False

    def closure(self, info: Info) -> torch.Tensor:
        """Closure of the loss.

        Args:
            info (Info): The job information object.

        Returns:
            torch.Tensor: The loss equal to the penalized negative log likeliihod
            without random effects.
        """
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
    """Job to fit the model with random effects.

    This is a random fitting job and it does set the `fit_` attribute when finished.

    It can be used with any optimizer factory built on the base class
    `torch.optim.Optimizer`. If None, it defaults to Adam.

    Use kwargs to pass defaults to the optimizer factory.

    Change the value of `fit_extra` if you do not want to fit extra parameters.

    It warns the user if the paramaters contain infinite or NaN values.


    Attributes:
        fit_extra (bool): Whether to set the model `fit` attribute or not.
        opt (torch.optim.Optimizer): The optimizer object.
        is_fitting (bool): False.

    Examples:
        >>> RandomFit(torch.optim.Adam, lr=0.01)
    """

    default_opt_factory: type[torch.optim.Optimizer] = DEFAULT_RANDOM_OPT_FACTORY
    is_fitting: bool = True

    def closure(self, info: Info) -> torch.Tensor:
        """Closure of the loss.

        Args:
            info (Info): The job information object.

        Returns:
            torch.Tensor: The loss equal to the penalized negative log likeliihod with
            random effects.
        """
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
    """Job to run a scheduler during the optimization.

    It can be used with any optimizer factory built on the base class
    `torch.optim.lr_scheduler.LRScheduler`.

    Use kwargs to pass defaults to the scheduler factory.

    Attributes:
        sched (torch.optim.lr_scheduler.LRScheduler): The scheduler object.
    """

    sched: torch.optim.lr_scheduler.LRScheduler

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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

    def run(self, info: Info):  # noqa: ARG002
        """Performs a single scheduling step using the optimizer.

        Args:
            info (Info): The job information object.
        """
        self.sched.step()

    def end(self, info: Info, metrics: Metrics):
        """Empty method.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The job metrics object.
        """
