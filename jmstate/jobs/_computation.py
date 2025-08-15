import warnings
from collections.abc import Callable

import torch

from ..typedefs._defs import Info, Job, Metrics


class ComputeFIM(Job):
    r"""Job to compute the Fisher Information Matrix.

    Many methods exist for computing the Fisher Information Matrix in latent variable
    models. In particular, this class leverages the expected Fisher Information Matrix
    using the identity:

    .. math::
        \mathcal{I}(\theta) = \mathbb{E}_{b \sim p(\cdot \mid x, \theta)} \left(\nabla
        \log \mathcal{L}(x, b ; \theta) \nabla \log \mathcal{L}(x, b ; \theta)^T \right)

    Please note that this is a stochastic approximation.

    For more information, see PMLR 202:1430-1453, 2023.

    Attributes:
        grad_m1 (torch.Tensor): The moment of order 1 estimate of the gradient.
        grad_m2 (torch.Tensor): The moment of order 2 estimate of the gradient.
        jac_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The Jacobian
            function. Useful because of batching.
    """

    grad_m1: torch.Tensor
    grad_m2: torch.Tensor
    jac_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(self, info: Info):
        """Initializes the moment estimates to 0.

        Warns the user if the model has not yet been fitted through the `fit`
        attribute.

        Args:
            info (Info): The job information object.
        """
        if not info.model.fit_:
            warnings.warn(
                (
                    "Model should be (random) fitted before computing Fisher "
                    "Information Matrix"
                ),
                stacklevel=2,
            )

        d = info.model.params_.numel
        self.grad_m1 = torch.zeros(d)
        self.grad_m2 = torch.zeros(d, d)

        def _jac_fn(params_flat_tensor: torch.Tensor, b: torch.Tensor):
            params = info.model.params_.from_flat_tensor(params_flat_tensor)
            return info.logpdfs_fn(params, b).sum(dim=1)

        self.jac_fn = torch.func.jacrev(_jac_fn)  # type: ignore

    def run(self, info: Info):
        """Updates the moment estimates by stochastic approximation.

        Args:
            info (Info): The job information object.
        """
        jac = self.jac_fn(info.model.params_.as_flat_tensor, info.b)

        self.grad_m1 += jac.mean(dim=0)
        self.grad_m2 += (jac.T @ jac) / info.sampler.n_chains

    def end(self, info: Info, metrics: Metrics):
        """Writes the Fisher Information Matrix into metrics.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The job metrics object.
        """
        self.grad_m2 /= info.iteration
        self.grad_m1 /= info.iteration
        metrics.fim = self.grad_m2 - torch.outer(self.grad_m1, self.grad_m1)


class ComputeCriteria(Job):
    r"""Job to compute AIC, BIC and log likelihood.

    These criteria allow for model selection.

    The AIC criterion is given by:

    .. math::
        \text{AIC} = -2 \log \mathcal{L}(x) + 2 k,

    where :math:`k` is the number of parameters.

    The BIC criterion is given by:

    .. math::
        \text{BIC} = -2 \log \mathcal{L}(x) + 2 \log n,

    where :math:`n` is the number of data measurements.

    Please note that this is a stochastic approximation.

    Attributes:
        loglik (float): The log likelihood.
    """

    loglik: float

    def __init__(self, info: Info):  # noqa: ARG002
        """Initializes the log likelihood to 0.

        Args:
            info (Info): The job information object.
        """
        self.loglik = 0.0

    def run(self, info: Info):
        """Updates the log likelihood by stochastic approximation.

        Args:
            info (Info): The job information object.
        """
        self.loglik += info.logliks.detach().sum().item() / info.sampler.n_chains

    def end(self, info: Info, metrics: Metrics):
        """Computes other criteria and write them into metrics.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The job metrics object.
        """
        metrics.loglik = self.loglik / info.iteration
        metrics.nloglik_pen = (
            info.data.size * info.model.pen(info.model.params_).item() - metrics.loglik
            if info.model.pen is not None
            else -metrics.loglik
        )

        d = info.model.params_.numel
        aic_pen = 2 * d
        bic_pen = d * torch.log(torch.tensor(info.data.effective_size)).item()

        metrics.aic = 2 * metrics.nloglik_pen + aic_pen
        metrics.bic = 2 * metrics.nloglik_pen + bic_pen


class ComputeEBEs(Job):
    r"""Job to compute the EBEs of b.

    Empirical Bayes Estimators are given by the posterior means of the random effects:

    .. math::
        \text{EBE} = \mathbb{E}_{b \sim p(\cdot \mid x, \theta)}(b).

    Please note that this is a stochastic approximation.

    Attributes:
        ebes (torch.Tensor): The Empirical Bayes Estimators.
    """

    ebes: torch.Tensor

    def __init__(self, info: Info):
        """Initializes the EBEs to 0.

        Args:
            info: The job information object.
        """
        self.ebes = torch.zeros(info.sampler.state.shape[1:])

    def run(self, info: Info):
        """Updates the EBEs by stochastic approximation.

        Args:
            info (Info): The job information object.
        """
        self.ebes += info.b.detach().mean(dim=0)

    def end(self, info: Info, metrics: Metrics):
        """Writes EBEs into metrics.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The job metrics object.
        """
        metrics.ebes = self.ebes / info.iteration
