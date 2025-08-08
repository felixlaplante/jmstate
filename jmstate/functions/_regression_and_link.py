from functools import cached_property
from typing import Callable

import torch
from beartype import beartype

from ..typedefs._defs import IntStrictlyPositive, Tensor1D, Tensor2D, Tensor3D, Tensor4D


def linear(t: Tensor1D | Tensor2D, psi: Tensor2D | Tensor3D) -> Tensor3D | Tensor4D:
    """Implements the linear transformation.

    Args:
        t (Tensor1D | Tensor2D): The time points.
        psi (Tensor2D | Tensor3D): The individual effects (parameters).

    Returns:
        Tensor3D | Tensor4D: The computed transformation.
    """
    return psi.unsqueeze(-1).expand(-1, t.size(-1), -1)


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


class Perceptron:
    """Implements perceptron."""

    hidden_dim: IntStrictlyPositive
    output_dim: IntStrictlyPositive
    hidden_activation: Callable[[torch.Tensor], torch.Tensor]
    output_activation: Callable[[torch.Tensor], torch.Tensor]

    @beartype
    def __init__(
        self,
        hidden_dim: IntStrictlyPositive,
        output_dim: IntStrictlyPositive = 1,
        *,
        hidden_activation: Callable[[torch.Tensor], torch.Tensor],
        output_activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.Identity(),
    ):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def __call__(
        self, t: Tensor1D | Tensor2D, psi: Tensor2D | Tensor3D
    ) -> Tensor3D | Tensor4D:
        """Gets a single layer activation.

        Args:
        t (Tensor1D | Tensor2D): The time points.
        psi (Tensor2D | Tensor3D): The individual effects (parameters).

        Returns:
            Tensor3D | Tensor4D: The computed transformation.
        """
        t = t.unsqueeze(-1)
        psi = psi.unsqueeze(-2)

        split_sizes = [
            self.hidden_dim,
            self.hidden_dim,
            self.output_dim * self.hidden_dim,
            self.output_dim,
        ]
        W1, b1, W2_flat, b2 = torch.split(psi, split_sizes, dim=-1)

        W2 = W2_flat.view(*W2_flat.shape[:-1], self.output_dim, self.hidden_dim)
        h = self.hidden_activation(t * W1 + b1)
        y = torch.einsum("...h,...mh->...m", h, W2) + b2
        return self.output_activation(y)

    @cached_property
    def dim(self) -> IntStrictlyPositive:
        """Returns the dimension of the neural network.

        Returns:
            IntStrictlyPositive: The dimension.
        """
        return self.hidden_dim * (2 + self.output_dim) + self.output_dim

    def glorot_init(self) -> Tensor1D:
        """Returns the initialization values for the weights and biases.

        Returns:
            Tensor1D: The initial values.
        """
        bound_W1 = torch.sqrt(torch.tensor(6.0 / (1 + self.hidden_dim))).item()
        W1 = torch.empty(self.hidden_dim).uniform_(-bound_W1, bound_W1)
        b1 = torch.zeros_like(W1)

        bound_W2 = torch.sqrt(
            torch.tensor(6.0 / (self.hidden_dim + self.output_dim))
        ).item()
        W2 = torch.empty(self.output_dim, self.hidden_dim).uniform_(-bound_W2, bound_W2)
        b2 = torch.zeros(self.output_dim)

        return torch.cat([W1.view(-1), b1.view(-1), W2.view(-1), b2.view(-1)], dim=0)
