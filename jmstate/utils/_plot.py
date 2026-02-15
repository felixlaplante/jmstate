from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import atleast_1d
from sklearn.utils._param_validation import validate_params  # type: ignore

if TYPE_CHECKING:
    from ..model._base import MultiStateJointModel


@validate_params(
    {
        "figsize": [tuple],
    },
    prefer_skip_nested_validation=True,
)
def plot_model_parameters_history(
    model: MultiStateJointModel,
    *,
    figsize: tuple[int, int] = (10, 8),
) -> tuple[plt.Figure, np.ndarray]:  # type: ignore
    """Plot the history of the model parameters during optimization.

    This function plots the history of the parameters in a grid of subplots.

    Args:
        model (MultiStateJointModel): The model to plot the parameters history of.
        figsize (tuple[int, int], optional): The figure size. Defaults to (10, 8).

    Raises:
        ValueError: If the model has less than two parameter history.

    Returns:
        tuple[plt.Figure, np.ndarray]: The figure and axes as a flat array.
    """
    from ..model._base import MultiStateJointModel  # noqa: PLC0415

    validate_params(
        {
            "model": [MultiStateJointModel],
        },
        prefer_skip_nested_validation=True,
    )

    if len(model.vector_model_parameters_history_) <= 1:
        raise ValueError("Only one parameter history provided")

    # Get the names
    named_parameters_dict = dict(model.model_parameters.named_parameters())
    nsubplots = len(named_parameters_dict)
    ncols = math.ceil(math.sqrt(nsubplots))
    nrows = math.ceil(nsubplots / ncols)

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)  # type: ignore
    axes = atleast_1d(axes).ravel()

    Y = torch.stack(model.vector_model_parameters_history_)
    i = 0
    for ax, (name, val) in zip(axes, named_parameters_dict.items(), strict=False):
        history = Y[:, i : (i := i + val.numel())]
        ax.plot(history, label=[f"{name}[{j}]" for j in range(val.numel())])
        ax.set(title=name, xlabel="Iteration", ylabel="Value")
        ax.legend()

    # Remove unused subplots
    for j in range(nsubplots, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Stochastic optimization of the parameters")  # type: ignore
    plt.tight_layout()

    return fig, axes
