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


class GradStop(Job):
    r"""Job to test the convergence based on gradient.

    Mathematically, an estimate of the first and second moments of the gradient are
    stochastically computed, using exponential moving averages with the formula:

    .. math::
        m_i^{(t+1)} \gets \beta_i m_i^{(t)} + (1 - \beta_i) \nabla^{(t)}, \quad
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
        min_consecutive: IntStrictlyPositive = 20,
        betas: tuple[NumProbability, NumProbability] = (0.9, 0.999),
        *,
        info: Info,
    ):
        """Initializes the ValueStop class.

        Args:
            atol (NumNonNegative | Tensor1DPositive, optional): Absolute tolerance,
            either scalar or element-wise. Defaults to 0.01.
            rtol (NumNonNegative | Tensor1DPositive, optional): Relative tolerance,
            either scalar or element-wise. Defaults to 0.01.
            min_consecutive (IntStrictlyPositive, optional): The minimum consecutive
                iterations with grad difference less than tolerance. Defaults to 20.
            betas (tuple[NumProbability, NumProbability], optional): Exponential moving
                averages forget parameters. Defaults to (0.9, 0.999).
            info (Info): The job information object.

        Raises:
            ValueError: If the optimizer has not been set before GradStop.
            ValueError: If `atol` is not all non-negative.
            ValueError: If `rtol` is not all non-negative.
        """
        self.atol = atol
        self.rtol = rtol
        self.min_consecutive = min_consecutive
        self.betas = betas

        self.n_consecutive = 0
        self.stopped = False

        if not hasattr(info, "opt"):
            raise ValueError("Optimizer must be set for GradStop")

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

        grads = torch.cat([g.view(-1) for g in grads_list])
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
    r"""Job to test the convergence based on parameter values.

    Mathematically, an estimate of the first moments of the parameter values are
    stochastically computed, using exponential moving averages with the formula:

    .. math::
        m^{(t+1)} \gets \beta m^{(t)} + (1 - \beta) \theta^{(t)}, \quad
        \hat{m}^{(t)} = \frac{m^{(t)}}{1 - \beta^t}.

    The `run` method returns true when for `min_consecutive` iterations:

    .. math::
        \vert \theta^{(t)} - \hat{m}_1^{(t)} \vert \leq \text{atol} + \text{rtol} \odot
        \hat{m}^{(t)}.

    Please note the tolerance must both be positive and can represent element-wise
    tolerances or global tolerance common to all parameters.

    Attributes:
        atol (int | float | torch.Tensor): Absolute tolerance.
        rtol (int | float | torch.Tensor): Relative tolerance.
        min_consecutive (int): Minimum consecutive iterations to declare convergence.
        beta (int | float): The forget parameter.
        p (torch.Tensor): The first moment estimate.
        n_consecutive (int): The number of iterations with convergence criterion met.
        stopped (bool): Indicator whether or not convergence has happened.
    """

    atol: int | float | torch.Tensor
    rtol: int | float | torch.Tensor
    min_consecutive: int
    beta: int | float
    p: torch.Tensor
    n_consecutive: int
    stopped: bool

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        atol: NumNonNegative | Tensor1DPositive = 0.01,
        rtol: NumNonNegative | Tensor1DPositive = 0.01,
        min_consecutive: IntStrictlyPositive = 20,
        beta: NumProbability = 0.9,
        *,
        info: Info,
    ):
        """Initializes the ValueStop class.

        Args:
            atol (NumNonNegative | Tensor1DPositive, optional): Absolute tolerance,
            either scalar or element-wise. Defaults to 0.01.
            rtol (NumNonNegative | Tensor1DPositive, optional): Relative tolerance,
            either scalar or element-wise. Defaults to 0.01.
            min_consecutive (IntStrictlyPositive, optional): The minimum consecutive
                iterations with grad difference less than tolerance. Defaults to 20.
            beta (tuple[NumProbability, NumProbability], optional): Exponential moving
                averages forget parameters. Defaults to 0.9.
            info (Info): The job information object.

        Raises:
            ValueError: If the optimizer has not been set before ValueStop.
            ValueError: If `atol` is not all non-negative.
            ValueError: If `rtol` is not all non-negative.
        """
        self.atol = atol
        self.rtol = rtol
        self.min_consecutive = min_consecutive
        self.beta = beta

        self.n_consecutive = 0
        self.stopped = False

        d = info.model.params_.numel
        if isinstance(self.atol, torch.Tensor):
            check_consistent_size(((self.atol, 0, "atol"), (d, None, "params.numel")))
        if isinstance(self.rtol, torch.Tensor):
            check_consistent_size(((self.rtol, 0, "rtol"), (d, None, "params.numel")))

        self.p = torch.zeros(d)

    def run(self, info: Info) -> bool:
        """Updates the moment estimate stochastically and checks for convergence.

        Args:
            info (Info): The job information object.

        Returns:
            bool: True if parameters have convergence, else False.
        """
        params = info.model.params_.as_flat_tensor
        self.p = self.beta * self.p + (1 - self.beta) * params
        p_hat = self.p / (1 - self.beta ** (info.iteration + 1))

        # Check convergence
        if ((params - p_hat).abs() <= self.atol + self.rtol * p_hat.abs()).all():
            self.n_consecutive += 1
            if self.n_consecutive >= self.min_consecutive:
                self.stopped = True
        else:
            self.n_consecutive = 0

        return self.stopped

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        """Checks and warns the user if the parameters have not yet converged.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The metrics information object.
        """
        if not self.stopped:
            warnings.warn(
                DEFAULT_NOT_CONVERGED_WARNING,
                stacklevel=2,
            )
