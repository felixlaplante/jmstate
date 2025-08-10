import torch


def identity(gamma: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    """The standard linear transformation b.

    Args:
        gamma (torch.Tensor): The population parameters.
        x (torch.Tensor): The (fixed) covariates.
        b (torch.Tensor): The random effects.

    Returns:
        torch.Tensor: The identity b
    """
    return b


def gamma_x_plus_b(
    gamma: torch.Tensor, x: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """The standard linear transformation x @ gamma + b.

    Args:
        gamma (torch.Tensor): The population parameters.
        x (torch.Tensor): The (fixed) covariates.
        b (torch.Tensor): The random effects.

    Returns:
        torch.Tensor: The transformation x @ gamma + b
    """
    return x @ gamma + b


def gamma_plus_b(
    gamma: torch.Tensor,
    x: torch.Tensor,  # noqa: ARG001
    b: torch.Tensor,
) -> torch.Tensor:
    """The linear transformation gamma + b.

    Args:
        gamma (torch.Tensor): The population parameters.
        x (torch.Tensor): The (fixed) covariates.
        b (torch.Tensor): The random effects.

    Returns:
        torch.Tensor: The transformation gamma + b
    """
    return gamma + b
