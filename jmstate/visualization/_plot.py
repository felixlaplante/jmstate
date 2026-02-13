from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch
from numpy import atleast_1d

from ..typedefs._params import ModelParams
from ..utils._checks import check_params_align

if TYPE_CHECKING:
    from ..model._base import MultiStateJointModel


def plot_params_history(
    model: MultiStateJointModel,
    *,
    true_params: ModelParams | None = None,
    figsize: tuple[int, int] = (10, 8),
    show: bool = True,
):
    """Plot the history of the parameters.

    This function plots the history of the parameters in a grid of subplots.

    Args:
        model (MultiStateJointModel): The model to plot the parameters history of.
        true_params (ModelParams | None, optional): The real parameters. Defaults to
            None.
        figsize (tuple[int, int], optional): The figure size. Defaults to (10, 8).
        show (bool, optional): Whether to show the plot. Defaults to True.

    Raises:
        ValueError: If the model has less than two parameter history.
        ValueError: If the parameters do not have the same names.
        ValueError: If the parameters do not have the same shapes.
    """
    if len(model.params_history_) <= 1:
        raise ValueError("Only one parameter history provided")

    # Get the names
    params_dict = model.params_.as_dict
    nsubplots = len(params_dict)
    ncols = math.ceil(math.sqrt(nsubplots))
    nrows = math.ceil(nsubplots / ncols)

    # Check alignment between model params and true params
    if true_params is not None:
        check_params_align(model.params_, true_params)

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)  # type: ignore
    axes = atleast_1d(axes).flat

    for ax, name in zip(axes, params_dict.keys(), strict=True):
        history = torch.cat(
            [p.as_dict[name].reshape(1, -1) for p in model.params_history_], dim=0
        )
        labels = [f"{name}[{i}]" for i in range(history.size(1))]

        if true_params is not None:
            lines = ax.plot(history, label=labels)
            for line, p in zip(lines, true_params.as_dict[name], strict=True):
                ax.axhline(p, linestyle="--", color=line.get_color())

        ax.set(title=name, xlabel="Iteration", ylabel="Value")
        ax.legend()

    # Remove unused subplots
    for j in range(nsubplots, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Stochastic optimization of the parameters")  # type: ignore
    plt.tight_layout()
    if show:
        plt.show()  # type: ignore
