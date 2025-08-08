import torch
from torch import nn

from ..typedefs._defs import Tensor1D, Tensor2D, Tensor3D, Tensor4D


def linear(t: Tensor1D | Tensor2D, psi: Tensor2D | Tensor3D) -> Tensor3D | Tensor4D:
    """Implements the linear transformation.

    Args:
        t (Tensor1D | Tensor2D): The time points.
        psi (Tensor2D | Tensor3D): The individual effects (parameters).

    Returns:
        Tensor3D | Tensor4D: The computed transformation.
    """
    return psi.unsqueeze(-2).expand(*psi.shape[:-1], t.size(-1), -1)


def sigmoid(t: Tensor1D | Tensor2D, psi: Tensor2D | Tensor3D) -> Tensor3D | Tensor4D:
    """Implements the sigmoid transformation.

    Args:
        t (Tensor1D | Tensor2D): The time points.
        psi (Tensor2D | Tensor3D): The individual effects (parameters).

    Returns:
        Tensor3D | Tensor4D: The computed transformation.
    """
    a, b, c = psi.chunk(3, dim=-1)
    return (a * torch.sigmoid((t - c) / b)).unsqueeze(-1)


class Net(nn.Module):
    """Implements a neural network."""

    net: nn.Module

    def __init__(self, net: nn.Module):
        super().__init__()  # type: ignore
        self.net = net
        self.net.requires_grad_(False)

    def forward(
        self, t: Tensor1D | Tensor2D, psi: Tensor2D | Tensor3D
    ) -> Tensor3D | Tensor4D:
        """Implements the neural transformation.

        Args:
            t (Tensor1D | Tensor2D): The time points.
            psi (Tensor2D | Tensor3D): The individual effects (parameters).

        Returns:
            Tensor3D | Tensor4D: The computed transformation.
        """
        psi_ext = psi.unsqueeze(-2).expand(*psi.shape[:-1], t.size(-1), -1)
        t_ext = t.unsqueeze(-1).broadcast_to(*psi_ext.shape[:-1], 1)
        x = torch.cat([t_ext, psi_ext], dim=-1)
        return self.net(x)
