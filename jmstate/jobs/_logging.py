from ..typedefs._defs import Info, Job, Metrics
from ..utils._misc import params_like_from_flat


class LogParamsHistory(Job):
    """Job to log the evolution of the paramters during fit."""

    def init(self, info: Info, metrics: Metrics) -> None:  # noqa: ARG002
        metrics.params_history = []

    def run(self, info: Info, metrics: Metrics) -> None:
        metrics.params_history.append(
            params_like_from_flat(
                info.model.params_, info.model.params_.as_flat_tensor.detach().clone()
            )
        )

    def end(self, info: Info, metrics: Metrics) -> None:
        pass


class MCMCDiagnostics(Job):
    """Job to log the evolution of the MCMC sampler."""

    def init(self, info: Info, metrics: Metrics) -> None:  # noqa: ARG002
        metrics.mcmc_diagnostics = []

    def run(self, info: Info, metrics: Metrics) -> None:
        metrics.mcmc_diagnostics.append(info.sampler.diagnostics)

    def end(self, info: Info, metrics: Metrics) -> None:
        pass
