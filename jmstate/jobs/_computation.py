import warnings
from collections.abc import Callable
from typing import Any

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import Info, Job, Metrics


class ComputeFIM(Job):
    r"""Job to compute the Fisher Information Matrix.

    Many methods exist for computing the Fisher Information Matrix in latent variable
    models. In particular, this class leverages the expected Fisher Information Matrix
    using the identity:

    .. math::
        \mathcal{I}_n(\theta) = \sum_{i=1}^n \mathbb{E}_{b \sim p(\cdot \mid x_i,
        \theta)} \left(\nabla \log \mathcal{L}(\theta ; x_i, b) \nabla \log
        \mathcal{L}(\theta ; x_i, b)^T \right)

    Please note that this is a stochastic approximation. By using the `bias=True`
    option which is enabled by default, you can substract a bias term giving the
    covariance matrix instead. This can be useful if the parameter estimate is quite
    far from the MLE.

    For more information, see ISSN 2824-7795.

    Attributes:
        bias (bool): Whether or not to substract the bias term. Defaults to True.
        jac (torch.Tensor): The first moment estimate of the jacobian.
        jac_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The Jacobian
            function. Useful because of batching.
    """

    bias: bool
    jac: torch.Tensor
    jac_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __new__(cls, bias: bool = True) -> Callable[[Info], Job]:
        """Creates the Fisher information Job.

        Args:
            bias (bool): Whether or not to substract the bias term. Defaults to True.
        """
        return super().__new__(cls, bias)

    def __init__(self, bias: bool, *, info: Info):  # type: ignore
        """Initializes the moment estimates to 0.

        Warns the user if the model has not yet been fitted through the `fit`
        attribute.

        Args:
            bias (bool): Whether or not to substract the bias term. Defaults to True.
            info (Info): The job information object.
        """
        if not info.model.fit_:
            warnings.warn(
                "Model should be fitted before computing Fisher Information Matrix",
                stacklevel=2,
            )

        d = info.model.params_.numel
        self.bias = bias
        self.jac = torch.zeros(info.data.size, d)

        def _jac_fn(params_flat_tensor: torch.Tensor, b: torch.Tensor):
            params = info.model.params_.from_flat_tensor(params_flat_tensor)
            return info.logpdfs_aux_fn(params, b)[0].mean(dim=0)

        self.jac_fn = torch.func.jacfwd(_jac_fn)  # type: ignore

    def run(self, info: Info):
        """Updates the moments estimates by stochastic approximation.

        Args:
            info (Info): The job information object.
        """
        self.jac += self.jac_fn(
            info.model.params_.as_flat_tensor, info.sampler.b
        ).detach()

    def end(self, info: Info, metrics: Metrics):
        """Writes the Fisher Information Matrix into metrics.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The job metrics object.
        """
        self.jac /= info.iteration
        metrics.fim = self.jac.T @ self.jac
        if self.bias:
            grad = self.jac.mean(dim=0)
            metrics.fim -= torch.outer(grad, grad) / info.data.size


class ComputeCriteria(Job):
    r"""Job to compute AIC, BIC and log likelihood.

    These criteria allow for model selection.

    The AIC criterion is given by:

    .. math::
        \text{AIC} = -2 \log \mathcal{L}(x) + 2 k,

    where :math:`k` is the number of parameters.

    The BIC criterion is given by:

    .. math::
        \text{BIC} = -2 \log \mathcal{L}(x) + \log \det \mathcal{I}_n,

    where :math:`\mathcal{I}_n` is the Fisher Information Matrix. If it has not been
    computed, it is estimated using the number of samples and approximating the penalty
    by :math:`k \log n`, where :math:`k` is the number of parameters.

    The log likelihood itself is computed the entropy identity:

    .. math::
        \log \mathcal{L}(x) = \mathbb{E}_{b \sim p(\cdot \mid x,
        \theta)} \left( \log \mathcal{L}(\theta ; x, b) \right) - \mathbb{E}_{b \sim
        p(\cdot \mid x, \theta)} \left( \log p(b \mid x, \theta) \right)

    The entropy itself is approximated using a Gaussian approximation of the
    posterior distribution.

    Please note that this is a stochastic approximation.

    Attributes:
        logpdf (float): The log complete likelihood.
        b (torch.Tensor): The first moment estimate of the latent variables.
        b2 (torch.Tensor): The second moment estimate of the latent variables.
    """

    logpdf: float
    b: torch.Tensor
    b2: torch.Tensor

    def __new__(cls) -> Callable[[Info], Job]:
        """Creates the criteria computation job."""
        return super().__new__(cls)

    def __init__(self, info: Info, **_kwargs: Any):  # type: ignore
        """Initializes the log likelihood to 0.

        Args:
            info (Info): The job information object.
        """
        if not info.model.fit_:
            warnings.warn(
                "Model should be fitted before computing criteria", stacklevel=2
            )

        self.logpdf = 0.0
        self.b = torch.zeros(info.data.size, info.model.params_.Q.dim)
        self.b2 = torch.zeros(
            info.data.size, info.model.params_.Q.dim, info.model.params_.Q.dim
        )

    def run(self, info: Info):
        """Updates the log likelihood by stochastic approximation.

        Args:
            info (Info): The job information object.
        """
        self.logpdf += info.sampler.logpdfs.sum().item() / info.sampler.n_chains
        self.b += info.sampler.b.mean(dim=0)
        self.b2 += (
            torch.einsum("ijk,ijl->jkl", info.sampler.b, info.sampler.b)
            / info.sampler.n_chains
        )

    def end(self, info: Info, metrics: Metrics):
        """Computes other criteria and write them into metrics.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The job metrics object.
        """
        # The LOG_TWO_PI factor cancels out with prior log likelihood
        self.b /= info.iteration
        self.b2 /= info.iteration
        covs = self.b2 - torch.einsum("ij,ik->ijk", self.b, self.b)
        entropy = 0.5 * (torch.logdet(covs) + info.model.params_.Q.dim).sum().item()

        metrics.loglik = self.logpdf / info.iteration + entropy
        metrics.nloglik_pen = (
            info.data.size * info.model.pen(info.model.params_).item() - metrics.loglik
            if info.model.pen is not None
            else -metrics.loglik
        )

        d = info.model.params_.numel
        aic_pen = 2 * d
        if hasattr(metrics, "fim"):
            bic_pen = torch.logdet(metrics.fim).item()
        else:
            warnings.warn("FIM not computed, using sample size instead", stacklevel=2)
            bic_pen = d * torch.log(torch.tensor(info.data.size)).item()

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

    def __new__(cls) -> Callable[[Info], Job]:
        """Creates the EBEs computation job."""
        return super().__new__(cls)

    def __init__(self, info: Info):  # type: ignore
        """Initializes the EBEs to 0.

        Args:
            info: The job information object.
        """
        self.ebes = torch.zeros(info.sampler.b.shape[1:])

    def run(self, info: Info):
        """Updates the EBEs by stochastic approximation.

        Args:
            info (Info): The job information object.
        """
        self.ebes += info.sampler.b.mean(dim=0)

    def end(self, info: Info, metrics: Metrics):
        """Writes EBEs into metrics.

        Args:
            info (Info): The job information object.
            metrics (Metrics): The job metrics object.
        """
        metrics.ebes = self.ebes / info.iteration
