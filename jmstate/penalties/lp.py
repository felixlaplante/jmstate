from typing import cast

import torch

from ..utils.structures import ModelParams


def l2_pen(params: ModelParams, lambda_reg: float = 1.0) -> torch.Tensor:
    """Implements L2 penalty.

    Args:
        params (ModelParams): The model parameters.
        lambda_reg (float, optional): The regularization strength. Defaults to 1.0.

    Returns:
        torch.Tensor: The penalty.
    """
    return cast(torch.Tensor, lambda_reg * sum(torch.sum(p**2) for p in params.as_list))


def l1_pen(params: ModelParams, lambda_reg: float = 1.0) -> torch.Tensor:
    """Implements L1 penalty.

    Args:
        params (ModelParams): The model parameters.
        lambda_reg (float, optional): The regularization strength. Defaults to 1.0.

    Returns:P
        torch.Tensor: The penalty.
    """
    return cast(
        torch.Tensor, lambda_reg * sum(torch.sum(p.abs()) for p in params.as_list)
    )
