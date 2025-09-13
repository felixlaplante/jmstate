from typing import cast

import torch


def identity(
    gamma: torch.Tensor | None,  # noqa: ARG001
    x: torch.Tensor | None,  # noqa: ARG001
    b: torch.Tensor,
) -> torch.Tensor:
    r"""The standard identity transformation.

    It is useful when only random effects are considered as individual parameters.
    This is simply the mapping:

    .. math::
        (\gamma, x, b) \mapsto b.

    Args:
        gamma (torch.Tensor | None): The population parameters.
        x (torch.Tensor | None): The (fixed) covariates.
        b (torch.Tensor): The random effects.

    Returns:
        torch.Tensor: The identity b
    """
    return b


def gamma_x_plus_b(
    gamma: torch.Tensor | None, x: torch.Tensor | None, b: torch.Tensor
) -> torch.Tensor:
    r"""The standard linear transformation.

    It is useful when a linear combination of covariates and random effects are
    considered as individual parameters.
    This is simply the mapping:

    .. math::
        (\gamma, x, b) \mapsto \gamma x + b

    Args:
        gamma (torch.Tensor | None): The population parameters. It must not be None.
        x (torch.Tensor | None): The (fixed) covariates. It must not be None.
        b (torch.Tensor): The random effects.

    Returns:
        torch.Tensor: The transformation x @ gamma + b
    """
    return cast(torch.Tensor, x) @ cast(torch.Tensor, gamma) + b


def gamma_plus_b(
    gamma: torch.Tensor | None,
    x: torch.Tensor | None,  # noqa: ARG001
    b: torch.Tensor,
) -> torch.Tensor:
    r"""The standard linear transformation.

    It is useful when a linear combination of population parameters and random effects
    are considered as individual parameters.
    This is simply the mapping:

    .. math::
        (\gamma, x, b) \mapsto \gamma + b

    Args:
        gamma (torch.Tensor | None): The population parameters. It must not be None.
        x (torch.Tensor | None): The (fixed) covariates.
        b (torch.Tensor): The random effects.

    Returns:
        torch.Tensor: The transformation gamma + b
    """
    return cast(torch.Tensor, gamma) + b
