from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Final

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import Info, Job, NumNonNegative

ADAM_LIKE: Final[tuple[type[torch.optim.Optimizer], ...]] = (
    torch.optim.Adam,
    torch.optim.AdamW,
    torch.optim.NAdam,
)


class _BaseL1Proximal(Job, ABC):
    """Base class for proximal operators.

    Attributes:
        lmda (int | float): The penalty.
        group (str): The group to penalize, either the link or covariate
            parameters.
        unique_params (list[torch.Tensor]): The unique parameters to penalize.
    """

    group: str
    lmda: int | float
    unique_params: list[torch.Tensor]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __new__(
        cls, lmda: NumNonNegative, group: str = "betas"
    ) -> Callable[[Info], Job]:
        """Creates the proximal operator.

        Args:
            lmda (NumNonNegative): The penalty.
            group (str, optional): The group to penalize, either the link or covariate
                parameters. Defaults to "betas".
        """
        return super().__new__(cls, lmda, group)

    def __init__(self, lmda: NumNonNegative, group: str = "betas", *, info: Info):  # type: ignore
        """Initialize the proximal operator.

        Args:
            lmda (NumNonNegative): The penalty.
            group (str, optional): The group to penalize, either the link or covariate
                parameters. Defaults to "betas".
            info (Info): The job information object.

        Raises:
            ValueError: If the optimizer has not been initialized before the proximal.
            ValueError: If the group is not in `alphas` nor `betas`.
            ValueError: If the group is None.
        """
        self.group = group
        self.lmda = lmda
        if group not in ("alphas", "betas"):
            raise ValueError(
                f"Group must be either 'alphas' or 'betas', got {self.group}"
            )

        if not hasattr(info, "opt"):
            raise ValueError("Optimizer must be initialized before proximal job")
        if getattr(info.model.params_, self.group) is None:
            raise ValueError(f"{self.group} is None")

        self.check_optimizer(info.opt)

        self.unique_params = list(set(getattr(info.model.params_, group).values()))

    def run(self, info: Info):
        """Projects the current parameter value.

        Args:
            info (Info): The job information object.
        """
        g = info.opt.param_groups[0]
        for p in self.unique_params:
            if p.grad is None:
                continue

            state = info.opt.state[p]
            if len(state) == 0:
                continue

            eff_lr = self.get_effective_lr(g, state)
            p.data = p.sign() * torch.clamp(p.abs() - self.lmda * eff_lr, min=0.0)

    @staticmethod
    @abstractmethod
    def check_optimizer(optimizer: torch.optim.Optimizer):
        """Checks if the optimizer is supported.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer object.
        """

    @staticmethod
    @abstractmethod
    def get_effective_lr(g: dict[str, Any], state: dict[str, Any]) -> float:
        """Get the effective learning rate.

        Args:
            g (dict[str, Any]): The parameter group/
            state (dict[str, Any]): The optimizer state.
        """


class AdamL1Proximal(_BaseL1Proximal):
    r"""Adam proximal operator.

    This proximal operator aims at bringing variable selection to joint modeling.
    This proximal operator only works with Adam or Adam-like optimizers that compute
    exponential moving averages of the order 1 and 2 moments of the gradient.

    Mathematically, consider the debiased estimates of the gradient and its element-wise
    square:

    .. math::
        \hat{m}_1^{(t)}, \quad \hat{m}_2^{(t)}.

    The proximal operator is given by the following formula. If :math:`\theta` are the
    model parameters, define for a L1 penalty :math:`\lambda \geq 0`:

    .. math::
        \theta_\text{pre}^{(t+1)} \gets \theta^{(t+1)} + \frac{\text{lr}}{\epsilon +
        \sqrt{\hat{m}_2^{(t)}}} \hat{m}_1^{(t)},

    then compute the projection:

    .. math::
        \theta^{(t+1)} \gets \operatorname{Proxi}_{\lambda \frac{\text{lr}}{\epsilon
        + \sqrt{\hat{m}_2^{(t)}}}}(\theta_\text{pre}^{(t+1)}).

    The proximal operator is defined by the element-wise soft thresholding:

    .. math::
        \theta_i^{(t+1)} \gets \operatorname{sgn}(\theta_{\text{pre}, i}^{(t+1)})
        \max\bigl(\vert \theta_{\text{pre}, i}^{(t+1)} \vert - \lambda \frac{\text{lr}}
        {\epsilon + \sqrt{\hat{m}_{2, i}^{(t)}}}, 0\bigr).

    Attributes:
        lmda (int | float): The penalty.
        group (str): The group to penalize, either the link or covariate
            parameters.
        unique_params (list[torch.Tensor]): The unique parameters to penalize.
    """

    @staticmethod
    def check_optimizer(optimizer: torch.optim.Optimizer):
        """Checks if the optimizer is supported.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer object.

        Raises:
            ValueError: If the optimizer is not Adam-like.
        """
        if not isinstance(optimizer, ADAM_LIKE):
            raise ValueError("Optimizer must be Adam or Adam-like")

    @staticmethod
    def get_effective_lr(g: dict[str, Any], state: dict[str, Any]) -> float:
        """Gets the effective learning rate.

        Args:
            g (dict[str, Any]): The parameter group.
            state (dict[str, Any]): The optimizer state.

        Returns:
            float: The effective learning rate.
        """
        return g["lr"] / torch.sqrt(state["exp_avg_sq"] + g["eps"])
