import copy

import torch

from ..typedefs._data import SampleData
from ..typedefs._defs import Info, Job, Metrics
from ..typedefs._params import ModelParams


class PredictY(Job):
    """Job to predict longitudinal values."""

    u: torch.Tensor

    def __init__(self, u: torch.Tensor) -> None:
        self.u = torch.as_tensor(u, dtype=torch.float32)

    def init(self, info: Info, metrics: Metrics) -> None:
        if self.u.ndim != 2 or self.u.shape[0] != info.data.size:
            raise ValueError(
                f"u has shape {self.u.shape}, expected ({info.data.size}, eval_points)"
            )

        metrics.pred_y = []

    def run(self, info: Info, metrics: Metrics) -> None:
        psi = info.model.model_design.individual_effects_fn(
            info.model.params_.gamma, info.data.x, info.b
        )
        y = info.model.model_design.regression_fn(self.u, psi)
        metrics.pred_y.append(y)

    def end(self, info: Info, metrics: Metrics) -> None:
        pass


class PredictSurvLogps(Job):
    """Job to predict survival log probability values."""

    u: torch.Tensor

    def __init__(self, u: torch.Tensor) -> None:
        self.u = torch.as_tensor(u, dtype=torch.float32)

    def init(self, info: Info, metrics: Metrics) -> None:
        if self.u.ndim != 2 or self.u.shape[0] != info.data.size:
            raise ValueError(
                f"u has shape {self.u.shape}, expected ({info.data.size}, eval_points)"
            )

        metrics.pred_surv_logps = []

    def run(self, info: Info, metrics: Metrics) -> None:
        psi = info.model.model_design.individual_effects_fn(
            info.model.params_.gamma, info.data.x, info.b
        )
        sample_data = SampleData(
            info.data.x, info.data.trajectories, psi, info.data.c, skip_validation=True
        )
        surv_logps = info.model.compute_surv_logps(sample_data, self.u)

        metrics.pred_surv_logps.append(surv_logps)

    def end(self, info: Info, metrics: Metrics) -> None:
        pass


class PredictTrajectories(Job):
    """Job to predict trajectories."""

    c_max: torch.Tensor
    n_trajectories_samples_per_b: int
    max_length: int

    def __init__(
        self,
        c_max: torch.Tensor,
        n_trajectories_samples_per_b: int,
        max_length: int = 10,
    ):
        self.c_max = torch.as_tensor(c_max, dtype=torch.float32)
        self.n_trajectories_samples_per_b = n_trajectories_samples_per_b
        self.max_length = max_length
        if self.n_trajectories_samples_per_b < 1:
            raise ValueError(
                f"n_trajectories_samples_per_b must be greater than 0, got {n_trajectories_samples_per_b}"
            )

    def init(self, info: Info, metrics: Metrics) -> None:
        if self.c_max.shape != (info.data.size,):
            raise ValueError(
                f"c_max has incorrect shape, got {self.c_max.shape}, expected {(info.data.size,)}"
            )

        self.x_rep = (
            info.data.x.repeat(self.n_trajectories_samples_per_b, 1)
            if info.data.x is not None
            else None
        )
        self.trajectories_rep = (
            info.data.trajectories * self.n_trajectories_samples_per_b
        )
        self.c_rep = info.data.c.repeat(self.n_trajectories_samples_per_b)
        self.c_max_rep = self.c_max.repeat(self.n_trajectories_samples_per_b)

        metrics.pred_trajectories = []

    def run(self, info: Info, metrics: Metrics) -> None:
        psi = info.model.model_design.individual_effects_fn(
            info.model.params_.gamma, info.data.x, info.b
        )
        psi_rep = psi.repeat(self.n_trajectories_samples_per_b, 1)

        sample_data = SampleData(
            self.x_rep, self.trajectories_rep, psi_rep, self.c_rep, skip_validation=True
        )
        trajectories = info.model.sample_trajectories(
            sample_data, self.c_max_rep, self.max_length
        )

        trajectory_chunks = [
            trajectories[i * info.data.size : (i + 1) * info.data.size]
            for i in range(self.n_trajectories_samples_per_b)
        ]

        metrics.pred_trajectories += trajectory_chunks

    def end(self, info: Info, metrics: Metrics) -> None:
        pass


class SwitchParams(Job):
    """Job to simulate different parameter values."""

    params_list: list[ModelParams]
    n_iterations_per_param: int
    n_params: int
    init_params: ModelParams

    def __init__(self, params_list: list[ModelParams], n_iterations_per_param: int):
        self.params_list = params_list
        self.n_iterations_per_param = n_iterations_per_param
        if self.n_iterations_per_param < 1:
            raise ValueError(
                f"n_iterations_per_param must be greater than 0, got {n_iterations_per_param}"
            )
        self.n_params = len(params_list)

    def init(self, info: Info, metrics: Metrics) -> None:
        self.init_params = copy.deepcopy(info.model.params_)

    def run(self, info: Info, metrics: Metrics) -> None:
        if info.iteration % self.n_iterations_per_param == 0:
            info.model.params_ = self.params_list[
                (info.iteration // self.n_iterations_per_param) % self.n_params
            ]

    def end(self, info: Info, metrics: Metrics) -> None:
        info.model.params_ = self.init_params
