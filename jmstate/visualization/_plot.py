from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch
from numpy import atleast_1d
from sklearn.utils._param_validation import validate_params  # type: ignore

from ..typedefs._params import ModelParams
from ..utils._checks import check_params_align

if TYPE_CHECKING:
    from ..model._base import MultiStateJointModel


@validate_params(
    {
        "true_params": [ModelParams, None],
        "figsize": [tuple],
        "show": [bool],
    },
    prefer_skip_nested_validation=True,
)
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
    from ..model._base import MultiStateJointModel  # noqa: PLC0415

    validate_params(
        {
            "model": [MultiStateJointModel],
        },
        prefer_skip_nested_validation=True,
    )

    if len(model.vector_params_history_) <= 1:
        raise ValueError("Only one parameter history provided")

    # Get the names
    named_params_dict = dict(model.params.named_parameters())
    nsubplots = len(named_params_dict)
    ncols = math.ceil(math.sqrt(nsubplots))
    nrows = math.ceil(nsubplots / ncols)

    # Check alignment between model params and true params
    if true_params is not None:
        check_params_align(model.params, true_params)
        named_true_params_dict = dict(true_params.named_parameters())

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)  # type: ignore
    axes = atleast_1d(axes).flat

    params_matrix_history = torch.stack(model.vector_params_history_)
    i = 0
    for ax, (name, val) in zip(axes, named_params_dict.items(), strict=True):
        history = params_matrix_history[:, i : (i := i + val.numel())]
        lines = ax.plot(history, label=[f"{name}[{j}]" for j in range(val.numel())])

        if true_params is not None:
            for line, p in zip(
                lines,
                named_true_params_dict[name].detach(),  # type: ignore
                strict=True,
            ):
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
