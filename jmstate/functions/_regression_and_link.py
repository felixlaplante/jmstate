import torch
from pydantic import ConfigDict, validate_call
from torch import nn

from ..typedefs._defs import IntNonNegative, LinkFn, RegressionFn
from ..utils._dtype import get_dtype


def linear(t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    r"""Implements the linear regression or link function.

    When reverting to a linear joint model, this gives the mapping:

    .. math::
        h(t, \psi) = \psi,

    where :math:`\psi` are the individual parameters.

    Args:
        t (torch.Tensor): The time points.
        psi (torch.Tensor): The individual effects (parameters).

    Returns:
        torch.Tensor: The computed transformation.
    """
    return psi.unsqueeze(-2).expand(*psi.shape[:-1], t.size(-1), -1)


class Net(nn.Module):
    r"""Implements a neural network.

    This neural network is very flexible, and any nn.Module may be used, in particular
    sequentials. When not knowing which link or regression function to use, try Net.
    You can use derivatives of arbitrary order for the link function using the
    derivatives method.
    If the input layer is in :math:`\mathbb{R}^d`, then one dimension is used for the
    input :math:`t`, and the rest for the individual parameters.

    Attributes:
        net (nn.Module): The neural network module.

    Examples:
        >>> from torch import nn
        >>> net = Net(nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1)))
        >>> link = net.derivatives((0, 1)) # derivatives of order 0 and 1
    """

    net: nn.Module

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, net: nn.Module):
        super().__init__()  # type: ignore
        self.net = net.to(get_dtype())
        self.requires_grad_(False)

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

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def derivatives(self, degs: tuple[IntNonNegative, ...]) -> RegressionFn | LinkFn:
        """Gets a function returning multiple derivatives of the neural network.

        Args:
            degs (tuple[IntNonNegative, ...]): The degrees.

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

            needs_grad = y.requires_grad
            ones = torch.ones_like(y)
            out_list = [y] if 0 in degs else []
            for i in range(1, max_deg + 1):
                y = torch.autograd.grad(
                    y, t_ext, ones, create_graph=i < max_deg or needs_grad
                )[0]
                if i in degs:
                    out_list.append(y)

            out = torch.cat(out_list, dim=-1)
            return out if needs_grad else out.detach()

        return _derivatives
