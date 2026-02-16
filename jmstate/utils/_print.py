from __future__ import annotations

from bisect import bisect_left
from typing import TYPE_CHECKING, Final

import torch
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector

if TYPE_CHECKING:
    from ..model._base import MultiStateJointModel


# Constants
SIGNIFICANCE_LEVELS: Final[tuple[float, ...]] = (
    0.001,
    0.01,
    0.05,
    0.1,
    float("inf"),
)
SIGNIFICANCE_CODES: Final[tuple[str, ...]] = (
    "[red3]***[/]",
    "[orange3]**[/]",
    "[yellow3]*[/]",
    ".",
    "",
)


def summary(model: MultiStateJointModel):
    """Prints a summary of the fitted model.

    This function prints the (nullity) p-values of the parameters as well as values and
    standard error. Also prints the log likelihood, AIC, BIC with lovely colors.

    Raises:
        ValueError: If the model is not fitted.
    """
    vector = parameters_to_vector(model.params.parameters())
    stderr = model.stderr
    zvalues = torch.abs(vector / stderr)
    pvalues = 2 * (1 - Normal(0, 1).cdf(zvalues))

    table = Table()
    table.add_column("Parameter name", justify="left")
    table.add_column("Value", justify="center")
    table.add_column("Standard Error", justify="center")
    table.add_column("z-value", justify="center")
    table.add_column("p-value", justify="center")
    table.add_column("Significance level", justify="center")

    i = 0
    for name, val in model.params.named_parameters():
        for j in range(val.numel()):
            code = SIGNIFICANCE_CODES[
                bisect_left(SIGNIFICANCE_LEVELS, pvalues[i].item())
            ]

            table.add_row(
                f"{name}[{j}]",
                f"{vector[i].item():.3f}",
                f"{stderr[i].item():.3f}",
                f"{zvalues[i].item():.3f}",
                f"{pvalues[i].item():.3f}",
                code,
            )
            i += 1

    criteria = Text(
        f"Log-likelihood: {model.loglik_:.3f}\n"
        f"AIC: {model.aic_:.3f}\n"
        f"BIC: {model.bic_:.3f}",
        style="bold cyan",
    )

    content = Group(table, Rule(style="dim"), criteria, Rule(style="dim"))
    panel = Panel(content, title="Model Summary", border_style="green", expand=False)

    Console().print(panel)
