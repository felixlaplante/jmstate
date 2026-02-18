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
def plot_params_history(
    model: MultiStateJointModel,
    *,
    figsize: tuple[int, int] = (10, 8),
) -> tuple[plt.Figure, np.ndarray]:  # type: ignore
    r"""Visualize the evolution of model parameters during fitting.

    This function generates a grid of subplots showing the trajectory of each model
    parameter across iterations, allowing assessment of convergence and exploration
    of the optimization process.

    Args:
        model (MultiStateJointModel): The fitted model whose parameter history is to
            be plotted.
        figsize (tuple[int, int], optional): Figure dimensions `(width, height)`.
            Defaults to `(10, 8)`.

    Raises:
        ValueError: If the model contains fewer than two recorded parameter states,
            preventing visualization.

    Returns:
        tuple[plt.Figure, np.ndarray]: A tuple containing the matplotlib `Figure`
            object and a flattened array of `Axes` objects corresponding to the
            subplots.
    """
    from ..model._base import MultiStateJointModel  # noqa: PLC0415

    validate_params(
        {
            "model": [MultiStateJointModel],
        },
        prefer_skip_nested_validation=True,
    )

    if len(model.params_history_) <= 1:
        raise ValueError("Only one parameter history provided")

    # Get the names
    named_parameters_dict = dict(model.params.named_parameters())
    nsubplots = len(named_parameters_dict)
    ncols = math.ceil(math.sqrt(nsubplots))
    nrows = math.ceil(nsubplots / ncols)

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)  # type: ignore
    axes = atleast_1d(axes).ravel()

    Y = torch.stack(model.params_history_)
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
