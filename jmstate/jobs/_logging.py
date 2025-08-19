from collections.abc import Callable
from typing import Any

from ..typedefs._defs import Info, Job, Metrics
from ..typedefs._params import ModelParams


class LogParamsHistory(Job):
    """Job to log the evolution of the paramters during fit.

    This yields a list of `ModelParams`.

    Attributes:
        params_history (list[ModelParams]): The history of the model parameters.
    """

    params_history: list[ModelParams]

    def __new__(cls) -> Callable[[Info], Job]:
        """Creates the parameter logging job."""
        return super().__new__(cls)

    def __init__(self, **_kwargs: Any):  # type: ignore
        """Initializes the history to an empty list."""
        self.params_history = []

    def run(self, info: Info):
        """Appends the current parameter values to the parameter list.

        Args:
            info (Info): The job information object.
        """
        self.params_history.append(info.model.params_.clone().detach())

    def end(self, metrics: Metrics, **_kwargs: Any):
        """Adds by concatenation or erases the metrics history.

        Args:
            metrics (Metrics): The metrics object.
        """
        if not hasattr(metrics, "params_history"):
            metrics.params_history = self.params_history
        else:
            metrics.params_history += self.params_history


class MCMCDiagnostics(Job):
    """Job to log the evolution of the MCMC sampler.

    This yields a list of dicts containing MCMC diagnostics, such as step sizes and
    acceptance rates.

    Attributes:
        mcmc_diagnostics (list[dict[str, Any]]): A list of MCMC diagnostic information.
    """

    mcmc_diagnostics: list[dict[str, Any]]

    def __new__(cls) -> Callable[[Info], Job]:
        """Creates the MCMC diagnostics logging job."""
        return super().__new__(cls)

    def __init__(self, **_kwargs: Any):  # type: ignore
        """Initializes the diagnostics to an empty list."""
        self.mcmc_diagnostics = []

    def run(self, info: Info):
        """Appends the current diagnostics dict to the parameter list.

        Args:
            info (Info): The job information object.
        """
        self.mcmc_diagnostics.append(info.sampler.diagnostics)

    def end(self, metrics: Metrics, **_kwargs: Any):
        """Adds by concatenation or erases the metrics MCMC diagnostics.

        Args:
            metrics (Metrics): The metrics object.
        """
        if not hasattr(metrics, "mcmc_diagnostics"):
            metrics.mcmc_diagnostics = self.mcmc_diagnostics
        else:
            metrics.mcmc_diagnostics += self.mcmc_diagnostics
