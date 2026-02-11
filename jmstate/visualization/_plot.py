import math

import matplotlib.pyplot as plt
import torch
from numpy import atleast_1d

from ..typedefs._params import ModelParams
from ..utils._checks import check_params_align


def plot_params_history(
    params_history: list[ModelParams],
    *,
    true_params: ModelParams | None = None,
    figsize: tuple[int, int] = (10, 8),
    show: bool = True,
):
    """Plot the history of the parameters.

    This function plots the history of the parameters in a grid of subplots.

    Args:
        params_history (list[ModelParams]): The parameter history as given in
            `Metrics.params_history`.
        true_params (ModelParams | None, optional): The real parameters. Defaults to
            None.
        figsize (tuple[int, int], optional): The figure size. Defaults to (10, 8).
        show (bool, optional): Whether to show the plot. Defaults to True.

    Raises:
        ValueError: If the parameters' history is empty.
        ValueError: If the parameters do not have the same names.
        ValueError: If the parameters do not have the same methods for Q.
        ValueError: If the parameters do not have the same methods for R.
    """
    if not params_history:
        raise ValueError("Empty parameters' history provided")

    # Get the names
    params_dict = params_history[0].as_dict
    nsubplots = len(params_dict)
    ncols = math.ceil(math.sqrt(nsubplots))
    nrows = math.ceil(nsubplots / ncols)

    # Check alignment between true params and history
    if true_params is not None:
        check_params_align(params_history[0], true_params)

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)  # type: ignore
    axes = atleast_1d(axes).flat

    for ax, (name, _) in zip(axes, params_dict.items(), strict=True):
        history = torch.cat(
            [p.as_dict[name].reshape(1, -1) for p in params_history], dim=0
        )

        labels = (
            [f"{name}[{i}]" for i in range(1, history.size(1) + 1)]
            if history.size(1) > 1
            else name
        )

        lines = ax.plot(history, label=labels)
        if true_params is not None:
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
