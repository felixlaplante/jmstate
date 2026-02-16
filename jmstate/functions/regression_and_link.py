"""Regression and link functions."""

__all__ = ["linear"]

import torch


def linear(t: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    r"""Implements the linear regression or link function.

    When reverting to a linear joint model, this gives the mapping

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
