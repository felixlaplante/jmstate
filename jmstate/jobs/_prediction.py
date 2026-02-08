import copy
from typing import Any

import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._data import SampleData
from ..typedefs._defs import (
    Info,
    IntStrictlyPositive,
    Job,
    Metrics,
    Tensor2D,
    TensorCol,
    Trajectory,
)
from ..typedefs._params import ModelParams
from ..utils._checks import check_consistent_size, check_inf, check_nan
from ..utils._dtype import get_dtype


class PredictY(Job):
    r"""Job to predict longitudinal values.

    For every drawing of a random effect :math:`b`, this computes at the prediction
    times :math:`u` the values of the regression function given input data:

    .. math::
        h(u, b).

    The variable `u` is expected to be a matrix with the same number of rows as
    individuals, and the same number of columns as prediction times.

    Attributes:
        u (torch.Tensor): The matrix containing prediction times.
        pred_y (list[torch.Tensor]): The list of predicted longitudinal values.
    """

    u: torch.Tensor
    pred_y: list[torch.Tensor]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __new__(cls, u: Tensor2D):
        """Creates the predicting job.

        Args:
            u (Tensor2D): The matrix containing prediction times.
        """
        return super().__new__(cls, u)

    def __init__(self, u: Tensor2D, info: Info):  # type: ignore
        """Initializes the predicting job.

        Args:
            u (Tensor2D): The matrix containing prediction times.
            info (Info): The job information object.

        Raises:
            ValueError: If u contains inf values.
            ValueError: If u contains NaN values.
            ValueError: If u has incompatible shape.
        """
        self.u = u.to(get_dtype())
        self.pred_y = []

        check_inf(((self.u, "u"),))
        check_nan(((self.u, "u"),))
        check_consistent_size(((self.u, 0, "u"), (info.data.size, None, "data.size")))

    def run(self, info: Info):
        """Appends the predicted longitudinal values.

        Args:
            info (Info): The job information object.
        """
        y = info.model.model_design.regression_fn(self.u, info.sampler.aux.psi)
        self.pred_y += [y[i] for i in range(y.size(0))]

    def end(self, metrics: Metrics, **_kwargs: Any):
        """Writes the predicted longitudinal values.

        Args:
            metrics (Metrics): The job metrics object.
        """
        metrics.pred_y = self.pred_y


class PredictSurvLogps(Job):
    r"""Job to predict survival log probability values.

    For every drawing of a random effect :math:`b`, this computes at the prediction
    times :math:`u` the values of the log survival probabilities given input data and
    conditionally to survival up to time :math:`c`:

    .. math::
        \log \mathbb{P}(T^* \geq u \mid T^* > c) = -\int_c^u \lambda(t) \, dt.

    When multiple transitions are allowed, :math:`\lambda(t)` is a sum over all
        possible transitions, that is to say if an individual is in the state :math:`k`
        from time :math:`t_0`, this gives:

        .. math::
            -\int_c^u \sum_{k'} \lambda^{k' \mid k}(t \mid t_0) \, dt.

    Please note this makes use of the Chasles property in order to avoid the computation
    of two integrals and make computations more precise.

    The variable `u` is expected to be a matrix with the same number of rows as
    individuals, and the same number of columns as prediction times.

    Attributes:
        u (torch.Tensor): The matrix containing prediction times.
        pred_surv_logps (torch.Tensor): The predicted survival log probabilities.
    """

    u: torch.Tensor
    pred_surv_logps: list[torch.Tensor]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __new__(cls, u: Tensor2D):
        """Creates the predicting job.

        Args:
            u (Tensor2D): The matrix containing prediction times.
        """
        return super().__new__(cls, u)

    def __init__(self, u: Tensor2D, info: Info):  # type: ignore
        """Inits the predicting job.

        Args:
            u (Tensor2D): The matrix containing prediction times.
            info (Info): The job information object.

        Raises:
            ValueError: If u contains inf values.
            ValueError: If u contains NaN values.
            ValueError: If u has incompatible shape.
        """
        self.u = u.to(get_dtype())
        self.pred_surv_logps = []

        check_inf(((self.u, "u"),))
        check_nan(((self.u, "u"),))
        check_consistent_size(((self.u, 0, "u"), (info.data.size, None, "data.size")))

    def run(self, info: Info):
        """Computes and appends the survival log probabilities.

        Args:
            info (Info): The job information object.
        """
        sample_data = SampleData(
            info.data.x,
            info.data.trajectories,
            info.sampler.aux.psi,
            info.data.c,
            skip_validation=True,
        )
        surv_logps = info.model.compute_surv_logps(sample_data, self.u)

        self.pred_surv_logps += [surv_logps[i] for i in range(surv_logps.size(0))]

    def end(self, metrics: Metrics, **_kwargs: Any):
        """Writes the predicted log survival probabilities to metrics.

        Args:
            metrics (Metrics): The job metrics object.
        """
        metrics.pred_surv_logps = self.pred_surv_logps


class PredictTrajectories(Job):
    r"""Job to predict  trajectories.

    For every drawing of a random effect :math:`b`, this simulates the trajectories
    up to time `c_max` with a maximum length of `max_length` to avoid infinite loops.

    The variable `c_max` is expected to be a column vector with the same number of rows
    as individuals.

    Attributes:
        c_max (torch.Tensor): The maximum sampling (censoring) times.
        max_length (int): The max length of the trajectories.
        pred_trajectories (list[list[Trajectory]]): The predicted (sampled)
            trajectories.
    """

    c_max: torch.Tensor
    max_length: int
    pred_trajectories: list[list[Trajectory]]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __new__(
        cls,
        c_max: TensorCol,
        max_length: IntStrictlyPositive = 10,
    ):
        """Creates the sampling predicting job.

        Args:
            c_max (TensorCol): The maximum sampling (censoring) times.
            max_length (IntStrictlyPositive, optional): The max length of the
                trajectories. Defaults to 10.
        """
        return super().__new__(cls, c_max, max_length)

    def __init__(  # type: ignore
        self,
        c_max: TensorCol,
        max_length: IntStrictlyPositive = 10,
        *,
        info: Info,
    ):
        """Inits the sampling predicting job.

        Args:
            c_max (TensorCol): The maximum sampling (censoring) times.
            max_length (IntStrictlyPositive, optional): The max length of the
                trajectories. Defaults to 10.
            info (Info): The job information object.

        Raises:
            ValueError: If c_max contains inf values.
            ValueError: If c_max contains NaN values.
            ValueError: If c_max has incompatible shape.
        """
        self.c_max = c_max.to(get_dtype())
        self.max_length = max_length
        self.pred_trajectories = []

        check_inf(((self.c_max, "c_max"),))
        check_nan(((self.c_max, "c_max"),))
        check_consistent_size(
            ((self.c_max, 0, "c_max"), (info.data.size, None, "data.size"))
        )

    def run(self, info: Info):
        """Predicts a trajectory given the current random effects drawing.

        Args:
            info (Info): The job information object.
        """
        for i in range(info.sampler.aux.psi.size(0)):
            sample_data = SampleData(
                info.data.x,
                info.data.trajectories,
                info.sampler.aux.psi[i],
                info.data.c,
                skip_validation=True,
            )

            self.pred_trajectories.append(
                info.model.sample_trajectories(
                    sample_data, self.c_max, max_length=self.max_length
                )
            )

    def end(self, metrics: Metrics, **_kwargs: Any):
        """Writes the predicted trajectories.

        Args:
            metrics (Metrics): The job metrics object.
        """
        metrics.pred_trajectories = self.pred_trajectories


class SwitchParams(Job):
    """Job to simulate different parameter values.

    This is useful when using the double Monte Carlo scheme of prediction.

    You can use this in conjunction with the `sample_params` method exposed in the
    `MultiStateJointModel` class.

    Examples:
        >>> params_list = my_model.sample_params(100)
        >>> my_model.do(pred_data, jobs=[PredictY(), SwitchParams(params_list)])
    """

    param_list: list[ModelParams]
    n_iterations_per_param: int
    n_params: int
    init_params: ModelParams

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __new__(
        cls, param_list: list[ModelParams], n_iterations_per_param: IntStrictlyPositive
    ):
        """Creates the job to switch between multiple parameters.

        Args:
            param_list (list[ModelParams]): The list of parameters.
            n_iterations_per_param (IntStrictlyPositive): The number of iterations to
                execute before switching.
        """
        return super().__new__(cls, param_list, n_iterations_per_param)

    def __init__(  # type: ignore
        self,
        param_list: list[ModelParams],
        n_iterations_per_param: IntStrictlyPositive,
        info: Info,
    ):
        """Initializes the job to switch between multiple parameters.

        Args:
            param_list (list[ModelParams]): The list of parameters.
            n_iterations_per_param (IntStrictlyPositive): The number of iterations to
                execute before switching.
            info (Info): The job information object.
        """
        self.param_list = param_list
        self.n_iterations_per_param = n_iterations_per_param
        self.n_params = len(param_list)

        self.init_params = copy.deepcopy(info.model.params_)

    def run(self, info: Info):
        """Executes the switches once every `n_iterations_per_param`.

        Args:
            info (Info): The job information object.
        """
        if info.iteration % self.n_iterations_per_param == 0:
            info.model.params_ = self.param_list[
                (info.iteration // self.n_iterations_per_param) % self.n_params
            ]

    def end(self, info: Info, **_kwargs: Any):
        """Restore default parameters.

        Args:
            info (Info): The job information object.
        """
        info.model.params_ = self.init_params
