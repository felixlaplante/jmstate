import copy

import torch
from beartype import beartype

from ..typedefs._data import SampleData
from ..typedefs._defs import Info, Job, Metrics, Tensor2D, Tensor3D, TensorCol
from ..typedefs._params import ModelParams
from ..utils._checks import check_consistent_size


class PredictY(Job):
    """Job to predict longitudinal values."""

    u: torch.Tensor
    pred_y: list[Tensor3D]

    @beartype
    def __init__(self, u: torch.Tensor):
        self.u = u.to(torch.float32)

    def init(self, info: Info, metrics: Metrics):  # noqa: ARG002
        check_consistent_size((self.u,), (0,), info.data.size)

        self.pred_y = []

    def run(self, info: Info, metrics: Metrics):  # noqa: ARG002
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
        self.u = u.to(torch.float32)

    def init(self, info: Info, metrics: Metrics):  # noqa: ARG002
        self.pred_surv_logps = []

    def run(self, info: Info, metrics: Metrics):  # noqa: ARG002
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
    n_trajectories_samples_per_b: int
    max_length: int

    @beartype
    def __init__(
        self,
        c_max: TensorCol,
        n_trajectories_samples_per_b: int,
        max_length: int = 10,
    ):
        self.c_max = c_max.to(torch.float32)
        self.n_trajectories_samples_per_b = n_trajectories_samples_per_b
        self.max_length = max_length
        if self.n_trajectories_samples_per_b < 1:
            raise ValueError(
                f"n_trajectories_samples_per_b must be greater than 0, got {n_trajectories_samples_per_b}"
            )

    def init(self, info: Info, metrics: Metrics):
        if info.sampler.n_chains != 1:
            raise ValueError(
                f"n_chains must be equal to 1 for trajectory prediction, got {info.sampler.n_chains}"
            )

        check_consistent_size((self.c_max,), (0,), info.data.size)

        self.x_rep = (
            None
            if info.data.x is None
            else info.data.x.repeat(self.n_trajectories_samples_per_b, 1)
        )
        self.trajectories_rep = (
            info.data.trajectories * self.n_trajectories_samples_per_b
        )
        self.c_rep = info.data.c.repeat(self.n_trajectories_samples_per_b, 1)
        self.c_max_rep = self.c_max.repeat(self.n_trajectories_samples_per_b, 1)

        metrics.pred_trajectories = []

    def run(self, info: Info, metrics: Metrics):
        psi_rep = (
            info.psi.detach().squeeze(0).repeat(self.n_trajectories_samples_per_b, 1)
        )

        sample_data = SampleData(
            self.x_rep, self.trajectories_rep, psi_rep, self.c_rep, skip_validation=True
        )
        trajectories = info.model.sample_trajectories(
            sample_data, self.c_max_rep, max_length=self.max_length
        )

        trajectory_chunks = [
            trajectories[i * info.data.size : (i + 1) * info.data.size]
            for i in range(self.n_trajectories_samples_per_b)
        ]

        metrics.pred_trajectories += trajectory_chunks

    def end(self, info: Info, metrics: Metrics):
        pass


class SwitchParams(Job):
    """Job to simulate different parameter values."""

    params_list: list[ModelParams]
    n_iterations_per_param: int
    n_params: int
    init_params: ModelParams

    @beartype
    def __init__(self, params_list: list[ModelParams], n_iterations_per_param: int):
        self.params_list = params_list
        self.n_iterations_per_param = n_iterations_per_param

        if self.n_iterations_per_param < 1:
            raise ValueError(
                f"n_iterations_per_param must be greater than 0, got {n_iterations_per_param}"
            )

        self.n_params = len(params_list)

    def init(self, info: Info, metrics: Metrics):  # noqa: ARG002
        self.init_params = copy.deepcopy(info.model.params_)

    def run(self, info: Info, metrics: Metrics):  # noqa: ARG002
        if info.iteration % self.n_iterations_per_param == 0:
            info.model.params_ = self.params_list[
                (info.iteration // self.n_iterations_per_param) % self.n_params
            ]

    def end(self, info: Info, metrics: Metrics):  # noqa: ARG002
        info.model.params_ = self.init_params
