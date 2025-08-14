import warnings

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import (
    Info,
    IntStrictlyPositive,
    Job,
    Metrics,
    NumNonNegative,
    NumProbability,
    Tensor1DPositive,
)
from ..utils._checks import check_consistent_size

# Constants
DEFAULT_NOT_CONVERGED_WARNING = (
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
    def __init__(
        self,
        min_steps: IntStrictlyPositive,
        info: Info,  # noqa: ARG002
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

    def end(self, info: Info, metrics: Metrics):
        """Empty method.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The metrics information object.
        """


class GradStop(Job):
    r"""Job to test the convergence based on gradient.

    All stopping jobs have to agree and return True for the process to stop. Other
    jobs return None.

    Mathematically, an estimate of the first and second moments of the gradient are
    stochastically computed, using exponential moving averages with the formula:

    .. math::
        m_i^{(t)} \gets \beta_i m_i^{(t-1)} + (1 - \beta_i) \nabla^{(t)}, \quad
        \hat{m}_i^{(t)} = \frac{m_i^{(t)}}{1 - \beta_i^t}.

    The `run` method returns true when for `min_consecutive` iterations:

    .. math::
        \hat{m}_1^{(t)} \leq \text{atol} + \text{rtol} \odot
        \sqrt{\hat{m}_2^{(t)}}.

    Please note the tolerance must both be positive and can represent element-wise
    tolerances or global tolerance common to all parameters.

    Attributes:
        atol (int | float | torch.Tensor): Absolute tolerance.
        rtol (int | float | torch.Tensor): Relative tolerance.
        min_consecutive (int): Minimum consecutive iterations to declare convergence.
        betas (tuple[int | float, int | float]): The forget parameters.
        m (torch.Tensor): The first moment estimate.
        v (torch.Tensor): The second moment estimate.
        n_consecutive (int): The number of iterations with convergence criterion met.
        stopped (bool): Indicator whether or not convergence has happened.
    """

    atol: int | float | torch.Tensor
    rtol: int | float | torch.Tensor
    min_consecutive: int
    betas: tuple[int | float, int | float]
    m: torch.Tensor
    v: torch.Tensor
    n_consecutive: int
    stopped: bool

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        atol: NumNonNegative | Tensor1DPositive = 0.01,
        rtol: NumNonNegative | Tensor1DPositive = 0.01,
        min_consecutive: IntStrictlyPositive = 50,
        betas: tuple[NumProbability, NumProbability] = (0.9, 0.999),
        *,
        info: Info,
    ):
        """Initializes the `GradStop` class.

        Args:
            atol (NumNonNegative | Tensor1DPositive, optional): Absolute tolerance,
            either scalar or element-wise. Defaults to 0.01.
            rtol (NumNonNegative | Tensor1DPositive, optional): Relative tolerance,
            either scalar or element-wise. Defaults to 0.01.
            min_consecutive (IntStrictlyPositive, optional): The minimum consecutive
                iterations with grad difference less than tolerance. Defaults to 50.
            betas (tuple[NumProbability, NumProbability], optional): Exponential moving
                averages forget parameters. Defaults to (0.9, 0.999).
            info (Info): The job information object.
        """
        self.atol = atol
        self.rtol = rtol
        self.min_consecutive = min_consecutive
        self.betas = betas

        self.n_consecutive = 0
        self.stopped = False

        d = info.model.params_.numel
        if isinstance(self.atol, torch.Tensor):
            check_consistent_size(((self.atol, 0, "atol"), (d, None, "params.numel")))
        if isinstance(self.rtol, torch.Tensor):
            check_consistent_size(((self.rtol, 0, "rtol"), (d, None, "params.numel")))

        self.m = torch.zeros(d)
        self.v = torch.zeros(d)

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

        grads = torch.cat([g.view(-1) for g in grads_list]).detach()
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * grads
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * grads**2

        m_hat = self.m / (1 - self.betas[0] ** (info.iteration + 1))
        v_hat = self.v / (1 - self.betas[1] ** (info.iteration + 1))

        # Check convergence
        if (m_hat.abs() <= self.atol + self.rtol * v_hat.sqrt()).all():
            self.n_consecutive += 1
            if self.n_consecutive >= self.min_consecutive:
                self.stopped = True
        else:
            self.n_consecutive = 0

        return self.stopped

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        """Checks and warns the user if the parameters have not yet converged.

        All stopping jobs have to agree and return True for the process to stop. Other
        jobs return None.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The metrics information object.
        """
        if not self.stopped:
            warnings.warn(
                DEFAULT_NOT_CONVERGED_WARNING,
                stacklevel=2,
            )


class ValueStop(Job):
    r"""Job to test the convergence based on log likelihood value.

    All stopping jobs have to agree and return True for the process to stop. Other
    jobs return None.

    Mathematically, an estimate of the first moment of the log likelihood and its
    difference is stochastically computed, using exponential moving averages with the
    formulae:

    .. math::
        l^{(t)} \gets \beta_1 l^{(t-1)} + (1 - \beta_1)
        \frac{\text{log likelihood}^{(t)}}{n}, \quad \hat{l}^{(t)} =
        \frac{l^{(t)}}{1 - \beta_1^t}.

    .. math::
        d^{(t)} \gets \beta_2 d^{(t-1)} + (1 - \beta_2)
        \frac{\text{log likelihood}^{(t)} - \text{log likelihood}^{(t - 1)}}{n},
        \quad \hat{d}^{(t)} = \frac{m^{(t)}}{1 - \beta_2^t}.

    The `run` method returns true when for `min_consecutive` iterations:

    .. math::
        \vert \hat{d}^{(t)} \vert \leq \text{atol} + \text{rtol} \odot n^{-1}
        \vert l^{(t)} \vert.

    Please note the tolerance must both be positive.

    Attributes:
        atol (int | float): Absolute tolerance.
        rtol (int | float): Relative tolerance.
        min_consecutive (int): Minimum consecutive iterations to declare convergence.
        betas (tuple[int | float, int | float]): The forget parameters.
        l (float): The first moment estimate of the log likelihood.
        d (float): The first moment estimate of the log likelihood difference.
        prev_loglik (float): The previous log likelihood value.
        n_consecutive (int): The number of iterations with convergence criterion met.
        stopped (bool): Indicator whether or not convergence has happened.
    """

    atol: int | float
    rtol: int | float
    min_consecutive: int
    betas: tuple[int | float, int | float]
    m: float
    v: float
    prev_loglik: float
    n_consecutive: int
    stopped: bool

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        atol: NumNonNegative = 0.001,
        rtol: NumNonNegative = 0.001,
        min_consecutive: IntStrictlyPositive = 50,
        betas: tuple[NumProbability, NumProbability] = (0.99, 0.99),
        *,
        info: Info,
    ):
        """Initializes the `ValueStop` class.

        Args:
            atol (NumNonNegative, optional): Absolute tolerance. Defaults to 0.001.
            rtol (NumNonNegative, optional): Relative tolerance. Defaults to 0.001.
            min_consecutive (IntStrictlyPositive, optional): The minimum consecutive
                iterations with grad difference less than tolerance. Defaults to 50.
            betas (tuple[NumProbability, NumProbability], optional): Exponential moving
                averages forget parameters. Defaults to (0.99, 0.99).
            info (Info): The job information object.
        """
        self.atol = atol
        self.rtol = rtol
        self.min_consecutive = min_consecutive
        self.betas = betas

        self.n_consecutive = 0
        self.stopped = False

        self.l = 0.0
        self.d = 0.0
        self.prev_loglik = info.logliks.mean().item()

    def run(self, info: Info) -> bool:
        """Updates the moment estimates stochastically and checks for convergence.

        Args:
            info (Info): The job information object.

        Returns:
            bool: True if parameters have convergence, else False.
        """
        loglik = info.logliks.mean().item()
        diff = loglik - self.prev_loglik
        self.prev_loglik = loglik

        self.l = self.betas[0] * self.l + (1 - self.betas[0]) * loglik
        self.d = self.betas[1] * self.d + (1 - self.betas[1]) * diff

        l_hat = self.l / (1 - self.betas[0] ** (info.iteration + 1))
        d_hat = self.d / (1 - self.betas[1] ** (info.iteration + 1))

        # Check convergence
        if abs(d_hat) <= self.atol + self.rtol * abs(l_hat):
            self.n_consecutive += 1
            if self.n_consecutive >= self.min_consecutive:
                self.stopped = True
        else:
            self.n_consecutive = 0

        return self.stopped

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        """Checks and warns the user if the parameters have not yet converged.

        All stopping jobs have to agree and return True for the process to stop. Other
        jobs return None.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The metrics information object.
        """
        if not self.stopped:
            warnings.warn(
                DEFAULT_NOT_CONVERGED_WARNING,
                stacklevel=2,
            )
