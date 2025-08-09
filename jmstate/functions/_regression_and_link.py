import torch
from beartype import beartype
from torch import nn

from ..typedefs._defs import IntPositive, LinkFn, RegressionFn


def linear(t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Implements the linear transformation.

    Args:
        t (torch.Tensor): The time points.
        psi (torch.Tensor): The individual effects (parameters).

    Returns:
        torch.Tensor: The computed transformation.
    """
    return psi.unsqueeze(-2).expand(*psi.shape[:-1], t.size(-1), -1)


def sigmoid(t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Implements the sigmoid transformation.

    Args:
        t (torch.Tensor): The time points.
        psi (torch.Tensor): The individual effects (parameters).

    Returns:
        torch.Tensor: The computed transformation.
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

    def forward(self, t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Implements the neural transformation.

        Args:
            t (torch.Tensor): The time points.
            psi (torch.Tensor): The individual effects (parameters).

        Returns:
            torch.Tensor: The computed transformation.
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

        @torch.enable_grad()  # type: ignore
        def _derivatives(t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
            psi_ext = psi.unsqueeze(-2).expand(*psi.shape[:-1], t.size(-1), -1)
            t_ext = t.unsqueeze(-1).expand(*psi_ext.shape[:-1], 1).requires_grad_()
            x = torch.cat([t_ext, psi_ext], dim=-1)
            y = self.net(x)

            ones = torch.ones_like(y)
            out_list: list[torch.Tensor] = []
            for i in range(max_deg + 1):
                if i in degs:
                    out_list.append(y)
                if i < max_deg:
                    y = torch.autograd.grad(y, t_ext, ones, create_graph=True)[0]

            out = torch.cat(out_list, dim=-1)
            needs_grad = any(p.requires_grad for p in (*self.net.parameters(), t, psi))
            return out if needs_grad else out.detach()

        return _derivatives
