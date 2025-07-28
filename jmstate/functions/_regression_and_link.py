import torch


def linear(t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Implements the linear transformation.

    Args:
        t (torch.Tensor): The time points.
        psi (torch.Tensor): The individual effects (parameters).

    Returns:
        torch.Tensor: The computed transformation.
    """
    return psi.unsqueeze(1).repeat(1, t.shape[1], 1)


def sigmoid(t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Implements the sigmoid transformation.

    Args:
        t (torch.Tensor): The time points.
        psi (torch.Tensor): The individual effects (parameters).

    Returns:
        torch.Tensor: The computed transformation.
    """
    a, b, c = psi.chunk(3, dim=1)
    return (a * torch.sigmoid((t - c) / b)).unsqueeze(2)
