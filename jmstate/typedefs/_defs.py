from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    NamedTuple,
    Protocol,
    Self,
    TypeAlias,
    runtime_checkable,
)

import torch
from pydantic import AfterValidator
from torch import nn
from xxhash import xxh3_64_intdigest

from ._validators import (
    is_col,
    is_ndim,
    is_non_neg,
    is_prob,
    is_strict_pos,
    is_valid_dtype,
)

if TYPE_CHECKING:
    from ..model._base import MultiStateJointModel
    from ..model._sampler import MetropolisHastingsSampler
    from ._data import ModelData
    from ._params import ModelParams


# Type Aliases
Num = int | float
Trajectory: TypeAlias = list[tuple[Num, Any]]


# Pydantic annotations
ValidDtype = Annotated[torch.dtype, AfterValidator(is_valid_dtype)]
Tensor0D = Annotated[torch.Tensor, AfterValidator(is_ndim(0))]
Tensor1D = Annotated[torch.Tensor, AfterValidator(is_ndim(1))]
Tensor2D = Annotated[torch.Tensor, AfterValidator(is_ndim(2))]
Tensor3D = Annotated[torch.Tensor, AfterValidator(is_ndim(3))]
Tensor4D = Annotated[torch.Tensor, AfterValidator(is_ndim(4))]
TensorCol = Annotated[Tensor2D, AfterValidator(is_col)]
Tensor1DPositive = Annotated[Tensor1D, AfterValidator(is_non_neg)]

IntNonNegative = Annotated[int, AfterValidator(is_non_neg)]
IntStrictlyPositive = Annotated[int, AfterValidator(is_strict_pos)]

NumNonNegative = Annotated[Num, AfterValidator(is_non_neg)]
NumStrictlyPositive = Annotated[Num, AfterValidator(is_strict_pos)]
NumProbability = Annotated[Num, AfterValidator(is_prob)]


# Protocols
@runtime_checkable
class IndividualEffectsFn(Protocol):
    """The individual effects function protocol.

    Calls the individual effects function.

    It must be able to yield 2D or 3D tensors given inputs of `gamma` (population
    parameters), `x` (covariates matrix), and `b` (random effects). Note `b` is
    either 2D or 3D.

    Args:
        gamma (torch.Tensor | None): The population parameters.
        x (Tensor2D | None): The fixed covariates matrix.
        b (Tensor2D | Tensor3D): The random effects.

    Returns:
        Tensor2D | Tensor3D: The individual parameters `psi`.

    Examples:
        >>> individual_effects_fn = lambda gamma, x, b: gamma + b
    """

    def __call__(
        self, gamma: torch.Tensor | None, x: Tensor2D | None, b: Tensor2D | Tensor3D
    ) -> Tensor2D | Tensor3D: ...


@runtime_checkable
class RegressionFn(Protocol):
    """The regression function protocol.

    It must be able to yield 3D and 4D tensors given 1D or 2D time inputs, as well
        as `psi` input of order 2 or 3. This is not very restrictive, but requires to be
        careful. The last dimension is the dimension of the response variable; second
        last is the repeated measurements; third last is individual based; possible
        fourth last is for parallelization of the MCMC sampler.

        It is identical to LinkFn.

    Args:
        t (Tensor1D | Tensor2D): The evaluation times.
        psi (Tensor2D | Tensor3D): The individual parameters.

    Returns:
        Tensor3D | Tensor4D: The response variable values.

    Examples:
        >>> def sigmoid(t: torch.Tensor, psi: torch.Tensor):
        >>>     scale, offset, slope = psi.chunk(3, dim=-1)
        >>>     # Fully broadcasted
        >>>     return (scale * torch.sigmoid((t - offset) / slope)).unsqueeze(-1)
        >>> regression_fn = sigmoid
    """

    def __call__(
        self, t: Tensor1D | Tensor2D, psi: Tensor2D | Tensor3D
    ) -> Tensor3D | Tensor4D: ...


@runtime_checkable
class LinkFn(Protocol):
    """The link function protocol.

    It must be able to yield 3D and 4D tensors given 1D or 2D time inputs, as well
        as `psi` input of order 2 or 3. This is not very restrictive, but requires to be
        careful. The last dimension is the dimension of the response variable; second
        last is the repeated measurements; third last is individual based; possible
        fourth last is for parallelization of the MCMC sampler.

        It is identical to RegressionFn.

    Args:
        t (Tensor1D | Tensor2D): The evaluation times.
        psi (Tensor2D | Tensor3D): The individual parameters.

    Returns:
        Tensor3D | Tensor4D: The response variable values.

    Examples:
        >>> def sigmoid(t: torch.Tensor, psi: torch.Tensor):
        >>>     scale, offset, slope = psi.chunk(3, dim=-1)
        >>>     # Fully broadcasted
        >>>     return (scale * torch.sigmoid((t - offset) / slope)).unsqueeze(-1)
        >>> link_fn = sigmoid
    """

    def __call__(
        self, t: Tensor1D | Tensor2D, psi: Tensor2D | Tensor3D
    ) -> Tensor3D | Tensor4D: ...


@runtime_checkable
class ClockMethod(Protocol):
    r"""The clock method protocol.

    This protocol is useful to differentiate between two natural mappings when dealing
    with base hazards:

    .. math::
        (t_0, t_1) \mapsto \begin{cases} t_1 - t_0 \text{(clock reset)}, \\ t_1
        \text{(clock forward)} \end{cases}.
    """

    def __call__(self, t0: TensorCol, t1: Tensor2D) -> Tensor2D: ...


# Named tuples
class MatRepr(NamedTuple):
    r"""A simple `NamedTuple` containing matrix representation information.

    Note three types of covariance matrices parametrization are provided: scalar
    matrix; diagonal matrix; full matrix. Defaults to the full matrix parametrization.
    This is achieved through a log Cholesky parametrization of the inverse covariance
    matrix. Formally, consider :math:`P = \Sigma^{-1}` the precision matrix and let
    :math:`L` be the Cholesky factor with positive diagonal elements, the log Cholseky
    is given by:

    .. math::
        \tilde{L}_{ij} = L_{ij}, \, i > j,

    and:

    .. math::
        \tilde{L}_{ii} = \log L_{ii}.

    This is very numerically stable and fast, as it doesn't require inverting the
    matrix when computing quadratic forms. The log determinant is then equal to:

    .. math::

        \log \det P = 2 \operatorname{Tr}(\tilde{L}).

    You can use these methods by creating the appropriate `MatRepr` with methods of
    `ball`, `diag` or `full`.

    Additionnally, if your data has mixed missing values, do not use `full` matrix
    parametrization for the residuals, as is this case the components must be
    independent.


    Attributes:
        flat (Tensor1D): The flat representation.
        dim (IntStrictlyPositive): The matrix dimension.
        method (str): The parametrization method, either `ball`, `diag` or `full`.
    """

    flat: Tensor1D
    dim: IntStrictlyPositive
    method: str


class BucketData(NamedTuple):
    """A simple `NamedTuple` containing transition information.

    Attributes:
        idxs (Tensor1D): The individual indices.
        t0 (TensorCol): A column vector of previous transition times.
        t1 (TensorCol): A column vecotr of next transition times.
    """

    idxs: Tensor1D
    t0: TensorCol
    t1: TensorCol


class AuxData(NamedTuple):
    """A simple internal `NamedTuple` for auxiliary data in sampling."""

    psi: Tensor3D
    logliks: Tensor2D


class HazardInfo(NamedTuple):
    """A simple internal `NamedTuple` required for hazard computation."""

    t0: TensorCol
    t1: Tensor2D
    x: Tensor2D | None
    psi: Tensor2D | Tensor3D
    alpha: Tensor1D
    beta: Tensor1D | None
    base_hazard_fn: BaseHazardFn
    link_fn: LinkFn


# SimpleNamespaces
class Info(SimpleNamespace):
    """A `SimpleNamespace` containing information used for the jobs.

    This may be used by the user as a custom bus.

    Attributes:
        data (ModelData): Learnable model data passed to `do` method.
        logpdfs_aux_fn (Callable[[ModelParams, Tensor3D], tuple[Tensor2D, AuxData]]):
            The log probability function with some aux containing individual effects as
            well as the log likelihoods. Used in optimization steps.
        iteration (int): The current iteration value. -1 at start and max at end.
        max_iterations (int): The maximum number of iterations allowed.
        model (MultiStateJointModel): The multistate joint model.
        sampler (MetropolisHastingsSampler): The current MCMC sampler.
        opt (torch.optim.Optimizer): The current optimizer (might not be set).
        b (Tensor3D): The random effects. The first dimension is equal to `n_chains`.
        logliks (Tensor2D): The log likelihoods.
        psi (Tensor3D): The individual effects. The first dimension is equal to
            `n_chains`.
    """

    data: ModelData
    logpdfs_aux_fn: Callable[[ModelParams, Tensor3D], tuple[Tensor2D, AuxData]]
    iteration: int
    max_iterations: int
    model: MultiStateJointModel
    sampler: MetropolisHastingsSampler
    opt: torch.optim.Optimizer


class Metrics(SimpleNamespace):
    """A `SimpleNamespace` containing metrics computed by the jobs.

    This may be used by the user as a custom bus.

    Attributes:
        fim (Tensor2D): The Fisher Information Matrix.
        ebes (Tensor2D): The Empirical Bayes Estimators of the rand effects.
        loglik (float): The log likelihood.
        nloglik_pen (float): The penalized negative log likelihood.
        aic (float): The AIC criterion.
        bic (float): The BIC criterion.
        pred_y (list[Tensor3D]): The predicted response variables for each drawing of
            random effects.
        pred_surv_logps (list[Tensor2D]): The predicted log survival probabilities for
            each drawing of random effects.
        pred_trajectories (list[Trajectory]): The predicted (sampled) trajectories for
            each drawing of random effects.
        params_history (list[ModelParams]): The parameters' evolution list.
        mcmc_diagnostics (list[dict[str, Any]]): The list of MCMC diagnostic dicts.
    """

    fim: Tensor2D
    ebes: Tensor2D
    loglik: float
    nloglik_pen: float
    aic: float
    bic: float
    pred_y: list[Tensor3D]
    pred_surv_logps: list[Tensor2D]
    pred_trajectories: list[Trajectory]
    params_history: list[ModelParams]
    mcmc_diagnostics: list[dict[str, Any]]


# Abstract classes
class BaseHazardFn(nn.Module, ABC):
    """The base hazard base class.

    This is not a protocol because caching is done, and therefore a key is required.
    Making this a `nn.Module`, one can check the value of the parameters and store their
    hashed values at the same time as the

    Note the base hazard function is in log scale, and expects a former transition time
    column vector `t0` as well as other times points at which the base hazard is to be
    computed. `t1` is a matrix with the same number of rows as `t0`.

    Implement a `forward` and do not forget the `super().__init__()` when declaring your
    own class.

    Pass the parameters you want to optimize in the `ModelParams.extra` attribute as a
    list. If you do not want them to be optimized, then by default they do not require
    gradients for the given implementations.
    """

    @property
    def key(self) -> tuple[int, ...]:
        """Returns a hash containing the class type and parameters if there are any.

        Returns:
            tuple[int, ...]: A key used in caching operations.
        """
        ident = id(self)
        parameters = list(self.parameters())
        if parameters == []:
            return (ident,)

        params_flat = nn.utils.parameters_to_vector(self.parameters())
        return (ident, xxh3_64_intdigest(params_flat.detach().numpy()))  # type: ignore

    @abstractmethod
    def forward(self, t0: TensorCol, t1: Tensor2D) -> Tensor2D: ...


class _Base_Job:
    def __new__(cls, *args: Any, **kwargs: Any) -> Callable[[Info], Self]:
        """Creates a partial in order to be initialized later.

        Returns:
            Callable[[Info], Self]: The object or a partial.
        """

        def _job_factory(info: Info):
            return cls._create_obj(*args, info=info, **kwargs)

        _job_factory.cls = cls  # type: ignore

        return _job_factory

    @classmethod
    def _create_obj(cls, *args: Any, **kwargs: Any) -> Self:
        """Creates a factory to wrap the creation.

        Returns:
            Self: The initialized job.
        """
        obj = super().__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj


class Job(_Base_Job):
    """This is the public base class for any Job.

    Please note the behaviour of this class is quite special and inherited from the
    private class `_Base_Job`. When `__new__` is called, the class will return a factory
    that is a `Callable[[Info], Job]`, and the `__init__` is not run until the main MCMC
    loop calls it.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Callable[[Info], Job]:
        """Returns the Job factory needed by the `do` method.

        Returns:
            Callable[[Info], Job]: The job factory.
        """
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the class.

        This must accept `info` as a keyword or **kwargs.
        """

    def run(self, *args: Any, **kwargs: Any) -> bool | None:
        """Run operations.

        This must accept `info` as a keyword or **kwargs.

        Returns:
            bool | None: None or False if not requiring to stop. True to require stop.
                When all non None returning jobs are returning True, then the main loop
                will be stopped.
        """

    def end(self, *args: Any, **kwargs: Any):
        """End operations.

        This must accept `info` and `metrics` as keywords or **kwargs.
        """


# Constants
LOG_TWO_PI: Final[Tensor0D] = torch.log(torch.tensor(2.0 * torch.pi))
