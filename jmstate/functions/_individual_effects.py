import torch

from ..typedefs._defs import Tensor2D, Tensor3D


def identity(
    gamma: torch.Tensor, x: Tensor2D, b: Tensor2D | Tensor3D
) -> Tensor2D | Tensor3D:
    """The standard linear transformation b.

    Args:
        gamma (torch.Tensor): The population parameters.
        x (Tensor2D): The (fixed) covariates.
        b (Tensor2D | Tensor3D): The random effects.

    Returns:
        Tensor2D | Tensor3D: The identity b
    """
    return b


def gamma_x_plus_b(
    gamma: torch.Tensor, x: Tensor2D, b: Tensor2D | Tensor3D
) -> Tensor2D | Tensor3D:
    """The standard linear transformation x @ gamma + b.

    Args:
        gamma (torch.Tensor): The population parameters.
        x (Tensor2D): The (fixed) covariates.
        b (Tensor2D | Tensor3D): The random effects.

    Returns:
        Tensor2D | Tensor3D: The transformation x @ gamma + b
    """
    return x @ gamma + b


def gamma_plus_b(
    gamma: torch.Tensor,
    x: Tensor2D,  # noqa: ARG001
    b: Tensor2D | Tensor3D,
) -> Tensor2D | Tensor3D:
    """The linear transformation gamma + b.

    Args:
        gamma (torch.Tensor): The population parameters.
        x (torch.Tensor): The (fixed) covariates.
        b (Tensor2D | Tensor3D): The random effects.

    Returns:
        Tensor2D | Tensor3D: The transformation gamma + b
    """
    return gamma + b
