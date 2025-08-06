import copy

import torch
from beartype import beartype

from ..typedefs._data import SampleData
from ..typedefs._defs import (
    Info,
    Job,
    Metrics,
    Tensor2D,
    Tensor3D,
    TensorCol,
    Trajectory,
)
from ..typedefs._params import ModelParams
from ..utils._checks import check_consistent_size


class PredictY(Job):
    """Job to predict longitudinal values."""

    u: torch.Tensor
    pred_y: list[Tensor3D]

    @beartype
    def __init__(self, u: torch.Tensor):
        self.u = u.to(torch.float64)
        self.pred_y = []

    def init(self, info: Info):
        check_consistent_size((self.u,), (0,), info.data.size)

    def run(self, info: Info):
        y = info.model.model_design.regression_fn(self.u, info.psi.detach())
        self.pred_y += [y[i] for i in range(y.size(0))]

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        metrics.pred_y = self.pred_y


class PredictSurvLogps(Job):
    """Job to predict survival log probability values."""

    u: torch.Tensor
    pred_surv_logps: list[Tensor2D]

    @beartype
    def __init__(self, u: Tensor2D):
        self.u = u.to(torch.float64)
        self.pred_surv_logps = []

    def init(self, info: Info):
        pass

    def run(self, info: Info):
        sample_data = SampleData(
            info.data.x,
            info.data.trajectories,
            info.psi.detach(),
            info.data.c,
            skip_validation=True,
        )
        surv_logps = info.model.compute_surv_logps(sample_data, self.u)

        self.pred_surv_logps += [surv_logps[i] for i in range(surv_logps.size(0))]

    def end(self, info: Info, metrics: Metrics):
        metrics.pred_surv_logps = self.pred_surv_logps


class PredictTrajectories(Job):
    """Job to predict trajectories."""

    c_max: torch.Tensor
    max_length: int
    pred_trajectories: list[Trajectory]

    @beartype
    def __init__(
        self,
        c_max: TensorCol,
        max_length: int = 10,
    ):
        self.c_max = c_max.to(torch.float64)
        self.max_length = max_length
        self.pred_trajectories = []

    def init(self, info: Info):
        check_consistent_size((self.c_max,), (0,), info.data.size)

    def run(self, info: Info):
        for i in range(info.psi.size(0)):
            sample_data = SampleData(
                info.data.x,
                info.data.trajectories,
                info.psi.detach()[i],
                info.data.c,
                skip_validation=True,
            )

            trajectories = info.model.sample_trajectories(
                sample_data, self.c_max, max_length=self.max_length
            )

            self.pred_trajectories += trajectories

    def end(self, info: Info, metrics: Metrics):
        metrics.pred_trajectories = self.pred_trajectories


class SwitchParams(Job):
    """Job to simulate different parameter values."""

    param_list: list[ModelParams]
    n_iterations_per_param: int
    n_params: int
    init_params: ModelParams

    @beartype
    def __init__(self, param_list: list[ModelParams], n_iterations_per_param: int):
        self.param_list = param_list
        self.n_iterations_per_param = n_iterations_per_param

        if self.n_iterations_per_param < 1:
            raise ValueError(
                f"n_iterations_per_param must be greater than 0, got {n_iterations_per_param}"
            )

        self.n_params = len(param_list)

    def init(self, info: Info):
        self.init_params = copy.deepcopy(info.model.params_)

    def run(self, info: Info):
        if info.iteration % self.n_iterations_per_param == 0:
            info.model.params_ = self.param_list[
                (info.iteration // self.n_iterations_per_param) % self.n_params
            ]

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        info.model.params_ = self.init_params
