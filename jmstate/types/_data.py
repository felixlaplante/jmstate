from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import torch
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils._param_validation import validate_params  # type: ignore
from sklearn.utils.validation import (  # type: ignore
    assert_all_finite,  # type: ignore
    check_consistent_length,  # type: ignore
)

from ..utils._checks import check_trajectories
from ..utils._surv import build_all_buckets
from ._defs import (
    IndividualEffectsFn,
    LinkFn,
    LogBaseHazardFn,
    RegressionFn,
    Trajectory,
)

if TYPE_CHECKING:
    from ..model._fit import FitMixin
    from ..model._predict import PredictMixin


# Dataclasses
@dataclass
class ModelDesign(BaseEstimator):
    """Class containing model design.

    For all functions, please use broadcasting as much as possible. It is almost always
    possible to broadcast parameters to vectorize efficiently the operations. If you
    copy, beware of a heavy performance hit. If unable, please use vmap.

    Also, note that the function passed to the MCMC sampler will be built using the
    `torch.no_grad()` decorator. If needs be, use `torch.enable_grad()` if one of the
    model design functions always require gradient computation regardless of setting.

    Ensure all functions all well defined on a closed interval and are differentiable
    almost everywhere.

    Attributes:
        individual_effects_fn (IndividualEffectsFn): The individual effects function. It
            must be able to yield 2D or 3D tensors given inputs of `gamma` (population
            parameters), `x` (covariates matrix), and `b` (random effects). Note `b` is
            either 2D or 3D.
        regression_fn (RegressionFn): The regression function. It
            must be able to yield 3D and 4D tensors given 1D or 2D time inputs, as well
            as `psi` input of order 2 or 3. This is not very restrictive, but requires
            to be careful. The last dimension is the dimension of the response variable;
            second last is the repeated measurements; third last is individual based;
            possible fourth last is for parallelization of the MCMC sampler.
        surv_map (Mapping[tuple[Any, Any], tuple[LogBaseHazardFn, LinkFn]]): A mapping
            of transition keys that can be typed however you want. The tuple contains a
            log base hazard function, as well as a link function that shares the same
            requirements as `regression_fn`. Log base hazard function is expected to be
            pure if caching is enabled, otherwise it will lead to false computations.

    Examples:
        >>> def sigmoid(t: torch.Tensor, psi: torch.Tensor):
        ...     scale, offset, slope = psi.chunk(3, dim=-1)
        ...     # Fully broadcasted
        ...     return (scale * torch.sigmoid((t - offset) / slope)).unsqueeze(-1)
        >>> individual_effects_fn = lambda gamma, x, b: gamma + b
        >>> regression_fn = sigmoid
        >>> surv_map = {("alive", "dead"): (Exponential(1.2), sigmoid)}
    """

    individual_effects_fn: IndividualEffectsFn
    regression_fn: RegressionFn
    surv_map: Mapping[
        tuple[Any, Any],
        tuple[LogBaseHazardFn, LinkFn],
    ]


@dataclass
class ModelData(BaseEstimator):
    r"""Dataclass containing learnable multistate joint model data.

    Note `y` is expected to be a 3D tensor of dimension :math:`(n, m, d)` if there are
    :math:`n` individual, with a maximum number of :math:`m` measurements in
    :math:`\mathbb{R}^d`. Padding is done with NaNs. The attribute `t` is expected to
    be either a 2D tensor of dimension :math:`(n, m)` if there are :math:`n` individual,
    or a 1D tensor of dimension :math:`(m,)` if the times are shared by all individual.
    Padding is not mandatory, but can be done with NaNs. `t` must not contain NaN values
    where `y` is not NaN. If the `r` attribute of `ModelParameters` is set to `full` and
    `y` is is :math:`\mathbb{R}^d` with :math:`d > 1`, then `r.covariance_type` must not
    be set to `full`, as it will lead to incorrect likelihood computations.

    This data can be completed using the `prepare` method, which makes the data usable
    for manual likelihood computations and MCMC. This usage is only encourage for people
    aware of the codebase. It will be run automatically by fitting and predicton methods
    without user input.

    Raises:
        ValueError: If some trajectory is empty.
        ValueError: If some trajectory is not sorted.
        ValueError: If some trajectory is not compatible with the censoring times.
        ValueError: If any of the inputs contain inf or NaN values except `y`.
        ValueError: If the size is not consistent between inputs.

    Attributes:
        x (torch.Tensor | None): The fixed covariates.
        t (torch.Tensor): The measurement times. Either a 1D tensor if the times are
            shared by all individual, or a matrix of individual times. Use padding with
            NaNs when necessary.
        y (torch.Tensor): The measurements. A 3D tensor of dimension :math:`(n, m, d)`
            if there are :math:`n` individual, with a maximum number of :math:`m`
            measurements in :math:`\mathbb{R}^d`. Use padding with NaNs when
            necessary.
        trajectories (list[Trajectory]): The list of the individual trajectories.
            A `Trajectory` is a list of tuples containing time and state.
        c (torch.Tensor): The censoring times as a column vector. They must not
            be less than the trajectory maximum times.
        valid_mask (torch.Tensor): The mask of the valid measurements.
        n_valid (torch.Tensor): The number of valid measurements.
        valid_t (torch.Tensor): The valid times.
        valid_y (torch.Tensor): The valid measurements.
        buckets (dict[tuple[Any, Any], tuple[torch.Tensor, ...]]): The buckets for the
            trajectories.
    """

    x: torch.Tensor | None
    t: torch.Tensor
    y: torch.Tensor
    trajectories: list[Trajectory]
    c: torch.Tensor
    valid_mask: torch.Tensor = field(init=False)
    n_valid: torch.Tensor = field(init=False)
    valid_t: torch.Tensor = field(init=False)
    valid_y: torch.Tensor = field(init=False)
    buckets: dict[tuple[Any, Any], tuple[torch.Tensor, ...]] = field(init=False)

    def __len__(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)

    def __post_init__(self):
        """Runs the post init conversions.

        Raises:
            ValueError: If some trajectory is empty.
            ValueError: If some trajectory is not sorted.
            ValueError: If some trajectory is not compatible with the censoring times.
            ValueError: If any of the inputs contain inf or NaN values except `t` and
                `y`.
            ValueError: If the size is not consistent between inputs.
        """
        validate_params(
            {
                "x": [torch.Tensor],
                "t": [torch.Tensor],
                "y": [torch.Tensor],
                "trajectories": [list],
                "c": [torch.Tensor],
                "skip_validation": [bool],
            },
            prefer_skip_nested_validation=True,
        )

        check_trajectories(self.trajectories, self.c)

        assert_all_finite(self.x, input_name="x")
        assert_all_finite(self.t, input_name="t", allow_nan=True)
        assert_all_finite(self.y, input_name="y", allow_nan=True)
        assert_all_finite(self.c, input_name="c")

        check_consistent_length(self.x, self.y, self.c, self.trajectories)
        check_consistent_length(self.t.transpose(0, -1), self.y.transpose(0, -2))

        # Check NaNs between t and y
        if ((~self.y.isnan()).any(dim=-1) & self.t.isnan()).any():
            raise ValueError("NaN time values on non NaN y values")

    def prepare(self, model: FitMixin | PredictMixin) -> Self:
        """Sets the representation for likelihood computations according to model.

        Args:
            model (FitMixin | PredictMixin): The multistate joint model.

        Returns:
            Self: The prepared data.
        """
        from ..model._fit import FitMixin  # noqa: PLC0415
        from ..model._predict import PredictMixin  # noqa: PLC0415

        validate_params(
            {
                "model": [FitMixin, PredictMixin],
            },
            prefer_skip_nested_validation=True,
        )
        check_consistent_length(model.model_parameters.r.cov, self.y.transpose(0, -1))

        self.valid_mask = ~self.y.isnan()
        self.n_valid = self.valid_mask.sum(dim=-2).to(torch.get_default_dtype())
        self.valid_t = self.t.nan_to_num(self.t.nanmean().item())
        self.valid_y = self.y.nan_to_num()
        self.buckets = build_all_buckets(
            self.trajectories, self.c, tuple(model.model_design.surv_map.keys())
        )

        return self


@dataclass
class ModelDataUnchecked(ModelData):
    """Unchecked model data class."""

    def __post_init__(self):
        pass


@dataclass
class SampleData(BaseEstimator):
    """Dataclass for data used in sampling.

    This assumes exact knowledge of the individual parameters `psi`. It is exposed
    in the `compute_surv_logps` and `sample_trajectories` methods of `HazardMixin`,
    which itself is used in the `MultiStateJointModel` class. This is used for data
    simulation, and internally, for the prediction of quantities linked to the
    survival function or trajectories. The `t_trunc` attribute is optional, and if not
    provided, it is set to the maximum time of observation of the individuals. It
    corresponds to the truncation time or conditionning time.

    Raises:
        ValueError: If some trajectory is empty.
        ValueError: If some trajectory is not sorted.
        ValueError: If some trajectory is not compatible with the censoring times.
        ValueError: If any of the inputs contain inf or NaN values.
        ValueError: If the size is not consistent between inputs.

    Attributes:
        x (torch.Tensor | None): The fixed covariates.
        trajectories (list[Trajectory]): The list of the individual trajectories.
            A `Trajectory` is a list of tuples containing time and state.
        psi (torch.Tensor): The individual parameters. Define it as a matrix with
            the same number of rows as there are `len(trajectories)`. Only use a 3D
            tensor if you fully understand the codebase and the mechanisms. Trajectory
            sampling may only be used with matrices.
        c (torch.Tensor | None, optional): The censoring times as a column vector. They
            must not be less than the trajectory maximum times. This corresponds to
            the last times of observation of the individuals or prediction current
            times.
    """

    x: torch.Tensor | None
    trajectories: list[Trajectory]
    psi: torch.Tensor
    t_trunc: torch.Tensor | None = None

    def __len__(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)

    def __post_init__(self):
        """Runs the post init conversions and checks.

        Raises:
            ValueError: If some trajectory is empty.
            ValueError: If some trajectory is not sorted.
            ValueError: If some trajectory is not compatible with the truncation times.
            ValueError: If any of the inputs contain inf or NaN values.
            ValueError: If the size is not consistent between inputs.
        """
        validate_params(
            {
                "trajectories": [list],
                "psi": [torch.Tensor],
                "t_trunc": [torch.Tensor],
            },
            prefer_skip_nested_validation=True,
        )

        check_trajectories(self.trajectories, self.t_trunc)

        assert_all_finite(self.x, input_name="x")
        assert_all_finite(self.psi, input_name="psi")
        assert_all_finite(self.t_trunc, input_name="t_trunc")

        check_consistent_length(
            self.x, self.psi.transpose(0, -2), self.t_trunc, self.trajectories
        )


@dataclass
class SampleDataUnchecked(SampleData):
    """Unchecked sample data class."""

    def __post_init__(self):
        pass
