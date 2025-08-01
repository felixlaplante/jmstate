from ..typedefs._defs import Info, Job, Metrics


class LogParamsHistory(Job):
    """Job to log the evolution of the paramters during fit."""

    def init(self, info: Info, metrics: Metrics) -> None:
        metrics.params_history = []

    def run(self, info: Info, metrics: Metrics) -> None:
        metrics.params_history.append(
            info.model.params_.as_flat_tensor.detach().clone()
        )

    def end(self, info: Info, metrics: Metrics) -> None:
        pass


class MCMCDiagnostics(Job):
    """Job to log the evolution of the MCMC sampler."""

    def init(self, info: Info, metrics: Metrics) -> None:
        metrics.mcmc_diagnostics = []

    def run(self, info: Info, metrics: Metrics) -> None:
        metrics.mcmc_diagnostics.append(info.sampler.diagnostics)

    def end(self, info: Info, metrics: Metrics) -> None:
        pass
