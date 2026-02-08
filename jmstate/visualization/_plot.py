import math

import matplotlib.pyplot as plt
import torch
from numpy import atleast_1d

from ..typedefs._params import ModelParams


def plot_history(
    param_history: list[ModelParams],
    figsize: tuple[int, int] = (10, 8),
    show: bool = True,
):
    """Plot the history of the parameters.

    This function plots the history of the parameters in a grid of subplots.

    Args:
        param_history (list[ModelParams]): The parameter history as given in
            `Metrics.params_history`.
        figsize (tuple[int, int], optional): The figure size. Defaults to (10, 8).
        show (bool, optional): Whether to show the plot. Defaults to True.

    Raises:
        ValueError: If the parameter history is empty.
    """
    if not param_history:
        raise ValueError("Empty parameter history provided")

    # Get the names
    names = param_history[0].as_named_list
    nsubplots = len(names)
    ncols = math.ceil(math.sqrt(nsubplots))
    nrows = math.ceil(nsubplots / ncols)

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)  # type: ignore
    axes = atleast_1d(axes).flat

    for i, (ax, (name, _)) in enumerate(zip(axes, names, strict=True)):
        history = torch.cat([p.as_list[i].reshape(1, -1) for p in param_history], dim=0)

        labels = (
            [f"{name}[{i}]" for i in range(1, history.size(1) + 1)]
            if history.size(1) > 1
            else name
        )

        ax.plot(history, label=labels)
        ax.set(title=name, xlabel="Iteration", ylabel="Value")
        ax.legend()

    # Remove unused subplots
    for j in range(nsubplots, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Stochastic optimization of the parameters")  # type: ignore
    if show:
        plt.tight_layout()
        plt.show()  # type: ignore
