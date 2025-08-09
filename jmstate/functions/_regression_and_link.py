import warnings

import torch
from beartype import beartype
from torch import nn

from ..typedefs._defs import (
    IntPositive,
    LinkFn,
    RegressionFn,
    Tensor1D,
    Tensor2D,
    Tensor3D,
    Tensor4D,
)


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

    @beartype
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
        t_ext = t.unsqueeze(-1).expand(*psi_ext.shape[:-1], 1)
        x = torch.cat([t_ext, psi_ext], dim=-1)
        return self.net(x)

    @beartype
    def derivatives(self, degs: tuple[IntPositive, ...]) -> RegressionFn | LinkFn:
        """Gets a function returning multiple derivatives of the neural network.

        Args:
            degs (tuple[IntPositive, ...]): The degrees.

        Returns:
            RegressionFn | LinkFn: A regresion/link function.
        """
        max_deg = max(degs)

        if max_deg < 1:
            warnings.warn(
                "Do not use derivatives if you do not need them, use __call__ instead",
                stacklevel=2,
            )

        @torch.enable_grad()  # type: ignore
        def _derivatives(
            t: Tensor1D | Tensor2D, psi: Tensor2D | Tensor3D
        ) -> Tensor3D | Tensor4D:
            psi_ext = psi.unsqueeze(-2).expand(*psi.shape[:-1], t.size(-1), -1)
            t_ext = t.unsqueeze(-1).expand(*psi_ext.shape[:-1], 1).requires_grad_()
            x = torch.cat([t_ext, psi_ext], dim=-1)
            y = self.net(x)

            outputs: list[Tensor3D | Tensor4D] = []
            for i in range(max_deg + 1):
                if i in degs:
                    outputs.append(y)
                if i < max_deg:
                    y, *_ = torch.autograd.grad(
                        y,
                        t_ext,
                        torch.ones_like(y),
                        create_graph=psi.requires_grad or i < max_deg - 1,
                    )

            outputs = torch.cat(outputs, dim=-1)
            return outputs if psi.requires_grad else outputs.detach()

        return _derivatives
