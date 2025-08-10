from typing import Any

from ..typedefs._defs import Info, Job, Metrics
from ..typedefs._params import ModelParams, params_like_from_flat


class LogParamsHistory(Job):
    """Job to log the evolution of the paramters during fit."""

    params_history: list[ModelParams]

    def __init__(self):
        self.params_history = []

    def init(self, info: Info):
        pass

    def run(self, info: Info):
        self.params_history.append(
            params_like_from_flat(info.model.params_, info.model.params_.as_flat_tensor)
        )

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        if not hasattr(metrics, "params_history"):
            metrics.params_history = self.params_history
        else:
            metrics.params_history += self.params_history


class MCMCDiagnostics(Job):
    """Job to log the evolution of the MCMC sampler."""

    mcmc_diagnostics: list[dict[str, Any]]

    def __init__(self):
        self.mcmc_diagnostics = []

    def init(self, info: Info):
        pass

    def run(self, info: Info):
        self.mcmc_diagnostics.append(info.sampler.diagnostics)

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        if not hasattr(metrics, "mcmc_diagnostics"):
            metrics.mcmc_diagnostics = self.mcmc_diagnostics
        else:
            metrics.mcmc_diagnostics += self.mcmc_diagnostics
