import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import (
    Info,
    IntStrictlyPositive,
    Job,
    NumNonNegative,
    NumProbability,
    Tensor1DPositive,
)
from ..utils._checks import check_consistent_size

# Constants
NOT_CONVERGED_WARNING = (
    "The parameters are not converged, try increasing the maximum number "
    "of iterations, decreasing the learning rate, or increasing n_chains"
)


class NoStop(Job):
    """Job to disallow convergence until a minimum number of iterations is reached.

    All stopping jobs have to agree and return True for the process to stop. Other
    jobs return None.

    Attributes:
        min_steps (int): The minimum number of steps to allow convergence.
    """

    min_steps: int

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __new__(cls, min_steps: IntStrictlyPositive) -> Callable[[Info], Job]:
        """Creates the `NoStop` class.

        Args:
            min_steps (IntStrictlyPositive): The minimum number of steps before allowing
                convergence checking.
        """
        return super().__new__(cls, min_steps)

    def __init__(  # type: ignore
        self, min_steps: IntStrictlyPositive, **_kwargs: Any
    ):
        """Initializes the `NoStop` class.

        Args:
            min_steps (IntStrictlyPositive): The minimum number of steps before allowing
                convergence checking.
        """
        self.min_steps = min_steps

    def run(self, info: Info) -> bool | None:
        """Returns True if convergence checking is allowed, else None.

        Args:
            info (Info): The job information object.

        Returns:
            bool: True if convergence checking is allowed, else None.
        """
        return None if info.iteration >= self.min_steps else False


class _BaseEMAStop(Job, ABC):
    """Job base class to test the convergence based on exponential moving averages.

    Attributes:
        atol (int | float | torch.Tensor, optional): Absolute tolerance,
            either scalar or element-wise.
        rtol (int | float | torch.Tensor, optional): Relative tolerance,
        either scalar or element-wise.
        min_consecutive (IntStrictlyPositive, optional): The minimum consecutive
            iterations with grad difference less than tolerance.
        betas (tuple[NumProbability, NumProbability], optional): Exponential moving
            averages' forget parameters.
        m (torch.Tensor): The first moment estimate.
        v (torch.Tensor): The second moment estimate.
        n_consecutive (int): The number of iterations with convergence criterion met.
    """

    atol: int | float | torch.Tensor
    rtol: int | float | torch.Tensor
    min_consecutive: int
    betas: tuple[int | float, int | float]
    m: torch.Tensor
    v: torch.Tensor
    n_consecutive: int

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __new__(
        cls,
        atol: NumNonNegative | Tensor1DPositive = 1e-6,
        rtol: NumNonNegative | Tensor1DPositive = 0.1,
        min_consecutive: IntStrictlyPositive = 1,
        *,
        betas: tuple[NumProbability, NumProbability] = (0.99, 0.99),
    ) -> Callable[[Info], Job]:
        """Creates the class.

        Args:
            atol (NumNonNegative | Tensor1DPositive, optional): Absolute tolerance,
            either scalar or element-wise. Defaults to 1e-6.
            rtol (NumNonNegative | Tensor1DPositive, optional): Relative tolerance,
            either scalar or element-wise. Defaults to 0.1.
            min_consecutive (IntStrictlyPositive, optional): The minimum consecutive
                iterations with grad difference less than tolerance. Defaults to 1.
            betas (tuple[NumProbability, NumProbability], optional): Exponential moving
                averages' forget parameters. Defaults to (0.99, 0.99).
        """
        return super().__new__(cls, atol, rtol, min_consecutive, betas=betas)

    def __init__(  # type: ignore
        self,
        atol: NumNonNegative | Tensor1DPositive = 1e-6,
        rtol: NumNonNegative | Tensor1DPositive = 0.1,
        min_consecutive: IntStrictlyPositive = 1,
        *,
        betas: tuple[NumProbability, NumProbability] = (0.99, 0.99),
        **_kwargs: Any,
    ):
        """Initializes the class.

        Args:
            atol (NumNonNegative | Tensor1DPositive, optional): Absolute tolerance,
            either scalar or element-wise. Defaults to 1e-6.
            rtol (NumNonNegative | Tensor1DPositive, optional): Relative tolerance,
            either scalar or element-wise. Defaults to 0.1.
            min_consecutive (IntStrictlyPositive, optional): The minimum consecutive
                iterations with grad difference less than tolerance. Defaults to 1.
            betas (tuple[NumProbability, NumProbability], optional): Exponential moving
                averages' forget parameters. Defaults to (0.99, 0.99).

        Raises:
            ValueError: If the shapes of the tolerances are not compatible.
        """
        self.atol = atol
        self.rtol = rtol
        self.min_consecutive = min_consecutive
        self.betas = betas

        self.n_consecutive = 0

    def run(self, info: Info) -> bool:
        """Updates the moment estimates stochastically and checks for convergence.

        Args:
            info (Info): The job information object.

        Returns:
            bool: True if parameters have convergence, else False.
        """
        grads_list: list[torch.Tensor] = []
        for p in info.model.params_.as_list:
            if p.grad is None:
                grads_list.append(torch.zeros_like(p))
            else:
                grads_list.append(p.grad)

        q = self.quantity(info)
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * q
        self.v = self.betas[0] * self.v + (1 - self.betas[1]) * q**2

        m_hat = self.m / (1 - self.betas[0] ** (info.iteration + 1))
        v_hat = self.v / (1 - self.betas[1] ** (info.iteration + 1))

        # Check convergence
        if (m_hat.abs() <= self.atol + self.rtol * v_hat.sqrt()).all():
            self.n_consecutive += 1
        else:
            self.n_consecutive = 0

        return self.n_consecutive >= self.min_consecutive

    def end(self, **_kwargs: Any):
        """Checks and warns the user if the parameters have not yet converged.

        All stopping jobs have to agree and return True for the process to stop. Other
        jobs return None.
        """
        if self.n_consecutive < self.min_consecutive:
            warnings.warn(
                NOT_CONVERGED_WARNING,
                stacklevel=2,
            )

    @abstractmethod
    def quantity(self, info: Info) -> torch.Tensor:
        """Gets the interest quantity.

        Returns:
            torch.Tensor: The quantity of interest.
        """


class GradStop(_BaseEMAStop):
    r"""Job to test the convergence based on gradient.

    All stopping jobs have to agree and return True for the process to stop. Other
    jobs return None.

    Mathematically, estimates of the first and second moments of the gradient are
    stochastically computed, using exponential moving averages with the formula:

    .. math::
        m_i^{(t)} \gets \beta_i m_i^{(t-1)} + (1 - \beta_i) \left( \nabla \log
        \mathcal{L}(\theta^{(t)}) \right)^i, \quad \hat{m}_i^{(t)} = \frac{m_i^{(t)}}
        {1 - \beta_i^t}.

    The `run` method returns true when for `min_consecutive` iterations:

    .. math::
        \vert \hat{m}_1^{(t)} \vert \leq \text{atol} + \text{rtol} \odot
        \sqrt{\hat{m}_2^{(t)}}.

    Please note the tolerances must both be non-negative and can represent element-wise
    tolerances or global tolerance common to all parameters.

    Attributes:
        atol (int | float | torch.Tensor): Absolute tolerance.
        rtol (int | float | torch.Tensor): Relative tolerance.
        min_consecutive (int): Minimum consecutive iterations to declare convergence.
        betas (tuple[int | float, int | float]): The forget parameters
        m (torch.Tensor): The first moment estimate.
        v (torch.Tensor): The second moment estimate.
        n_consecutive (int): The number of iterations with convergence criterion met.
        stopped (bool): Indicator whether or not convergence has happened.
    """

    @wraps(_BaseEMAStop.__init__)
    def __init__(self, *args: Any, info: Info, **kwargs: Any):
        super().__init__(*args, **kwargs)

        d = info.model.params_.numel
        if isinstance(self.atol, torch.Tensor):
            check_consistent_size(((self.atol, 0, "atol"), (d, None, "params.numel")))
        if isinstance(self.rtol, torch.Tensor):
            check_consistent_size(((self.rtol, 0, "rtol"), (d, None, "params.numel")))

        self.m = torch.zeros(d)
        self.v = torch.zeros(d)

    def quantity(self, info: Info) -> torch.Tensor:
        """Gets the gradients.

        Args:
            info (Info): The job information object.

        Returns:
            torch.Tensor: The flat gradients.
        """
        grads_list: list[torch.Tensor] = []
        for p in info.model.params_.as_list:
            if p.grad is None:
                grads_list.append(torch.zeros_like(p))
            else:
                grads_list.append(p.grad)

        return torch.cat([g.reshape(-1) for g in grads_list]).detach()


class ParamStop(_BaseEMAStop):
    r"""Job to test the convergence based on the parameters' evolution.

    All stopping jobs have to agree and return True for the process to stop. Other
    jobs return None.

    Mathematically, estimates of the first and the second moments of the parameters'
    difference are stochastically computed, using exponential moving averages with the
    formula:

    .. math::
        m_i^{(t)} \gets \beta_i m_i^{(t-1)} + (1 - \beta_i) (\theta^{(t)} -
        \theta^{(t-1)})^i, \quad \hat{m}_i^{(t)} = \frac{m_i^{(t)}}{1 - \beta_i^t}.

    The `run` method returns true when for `min_consecutive` iterations:

    .. math::
        \vert \hat{m}_1^{(t)} \vert \leq \text{atol} + \text{rtol} \odot
        \sqrt{\hat{m}_2^{(t)}}.

    Please note the tolerances must both be non-negative and can represent element-wise
    tolerances or global tolerance common to all parameters.

    Attributes:
        atol (int | float): Absolute tolerance.
        min_consecutive (int): Minimum consecutive iterations to declare convergence.
        betas (tuple[int | float, int | float]): The forget parameters
        m (torch.Tensor): The first moment estimate of the log likelihood gain.
        v (torch.Tensor): The second moment estimate of the log likelihood gain.
        prev_loglik (torch.Tensor): The previous log likelihood value.
        n_consecutive (int): The number of iterations with convergence criterion met.
        stopped (bool): Indicator whether or not convergence has happened.
    """

    prev_params: torch.Tensor

    @wraps(_BaseEMAStop.__init__)
    def __init__(self, *args: Any, info: Info, **kwargs: Any):
        super().__init__(*args, **kwargs)

        d = info.model.params_.numel
        if isinstance(self.atol, torch.Tensor):
            check_consistent_size(((self.atol, 0, "atol"), (d, None, "params.numel")))
        if isinstance(self.rtol, torch.Tensor):
            check_consistent_size(((self.rtol, 0, "rtol"), (d, None, "params.numel")))

        self.m = torch.zeros(d)
        self.v = torch.zeros(d)

        self.prev_params = info.model.params_.as_flat_tensor

    def quantity(self, info: Info) -> torch.Tensor:
        """Gets the parameters' difference.

        Args:
            info (Info): The job information object.

        Returns:
            torch.Tensor: The flat parameters' difference.
        """
        new_params = info.model.params_.as_flat_tensor
        diff = new_params - self.prev_params
        self.prev_params = new_params
        return diff
