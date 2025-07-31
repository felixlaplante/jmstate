import torch


def gamma_x_cat_b(
    gamma: torch.Tensor, x: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """The standard linear transformation [x @ gamma, b].

    Args:
        gamma (torch.Tensor): The population parameters.
        x (torch.Tensor): The (fixed) covariates.
        b (torch.Tensor): The random effects.

    Returns:
        torch.Tensor: The transformation [x @ gamma, b]
    """
    return torch.cat([x @ gamma, b], dim=1)

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


def gamma_plus_b(gamma: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The linear transformation gamma + b.

    Args:
        gamma (torch.Tensor): The population parameters.
        x (torch.Tensor): The (fixed) covariates.
        b (torch.Tensor): The random effects.

    Returns:
        torch.Tensor: The transformation [x @ gamma, b]
    """
    return gamma + b
