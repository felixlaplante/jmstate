import math

import matplotlib.pyplot as plt
import torch
from numpy import atleast_1d

from ..typedefs._params import ModelParams
from ..utils._checks import check_params_size_and_names


def plot_history(
    params_history: list[ModelParams],
    *,
    real_params: ModelParams | None = None,
    figsize: tuple[int, int] = (10, 8),
    show: bool = True,
):
    """Plot the history of the parameters.

    This function plots the history of the parameters in a grid of subplots.

    Args:
        params_history (list[ModelParams]): The parameter history as given in
            `Metrics.params_history`.
        real_params (ModelParams | None, optional): The real parameters. Defaults to
            None.
        figsize (tuple[int, int], optional): The figure size. Defaults to (10, 8).
        show (bool, optional): Whether to show the plot. Defaults to True.

    Raises:
        ValueError: If the parameters' history is empty.
        ValueError: If the real parameters do not match the parameters' history length.
        ValueError: If the real parameters do not match the parameters' history names.
    """
    if not params_history:
        raise ValueError("Empty parameters' history provided")

    # Get the names
    named_params_list = params_history[0].as_named_list
    nsubplots = len(named_params_list)
    ncols = math.ceil(math.sqrt(nsubplots))
    nrows = math.ceil(nsubplots / ncols)

    # Check naming is correct as well as sizes
    if real_params is not None:
        named_real_params_list = real_params.as_named_list
        check_params_size_and_names(named_params_list, named_real_params_list)

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)  # type: ignore
    axes = atleast_1d(axes).flat

    for i, (ax, (name, _)) in enumerate(zip(axes, named_params_list, strict=True)):
        history = torch.cat(
            [p.as_list[i].reshape(1, -1) for p in params_history], dim=0
        )

        labels = (
            [f"{name}[{i}]" for i in range(1, history.size(1) + 1)]
            if history.size(1) > 1
            else name
        )

        lines = ax.plot(history, label=labels)
        if real_params is not None:
            for line, p in zip(lines, real_params.as_list[i], strict=True):
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

