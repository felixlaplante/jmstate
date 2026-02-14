from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from sklearn.utils._param_validation import validate_params  # type: ignore
from sklearn.utils.validation import (  # type: ignore
    assert_all_finite,  # type: ignore
    check_consistent_length,  # type: ignore
)

from ..utils._checks import check_trajectories
from ..utils._surv import build_all_buckets
from ._defs import BaseHazardFn, IndividualEffectsFn, LinkFn, RegressionFn, Trajectory
from ._params import ModelParams


# Dataclasses
@dataclass
class ModelDesign:
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
        surv (Mapping[tuple[Any, Any], tuple[BaseHazardFn, LinkFn]]): A mapping of
            transition keys that can be typed however you want. The tuple contains a
            base hazard function in log scale, as well as a link function that shares
            the same requirements as `regression_fn`. Base hazard function is expected
            to be pure if caching is enabled, otherwise it will lead to false
            computations.

    Examples:
        >>> def sigmoid(t: torch.Tensor, psi: torch.Tensor):
        >>>     scale, offset, slope = psi.chunk(3, dim=-1)
        >>>     # Fully broadcasted
        >>>     return (scale * torch.sigmoid((t - offset) / slope)).unsqueeze(-1)
        >>> individual_effects_fn = lambda gamma, x, b: gamma + b
        >>> regression_fn = sigmoid
        >>> surv = {("alive", "dead"): (Exponential(1.2), sigmoid)}
    """

    individual_effects_fn: IndividualEffectsFn
    regression_fn: RegressionFn
    surv: Mapping[
        tuple[Any, Any],
        tuple[BaseHazardFn, LinkFn],
    ]


@dataclass
class ModelData:
    r"""Dataclass containing learnable multistate joint model data.

    Note `y` is expected to be a 3D tensor of dimension :math:`(n, m, d)` if there are
    :math:`n` individual, with a maximum number of :math:`m` measurements in
    :math:`\mathbb{R}^d`. Its values should not be all NaNs.

    Raises:
        ValueError: If the trajectories are not sorted by time.
        ValueError: If the censoring time is lower than the maximum transition time.
        ValueError: If any of the inputs contain inf values.
        ValueError: If any of the inputs contain NaN values except `y`.
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
    """

    x: torch.Tensor | None
    t: torch.Tensor
    y: torch.Tensor
    trajectories: list[Trajectory]
    c: torch.Tensor

    def __len__(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)

    def __post_init__(self):
        """Runs the post init conversions.

        Raises:
            ValueError: If the trajectories are not sorted by time.
            ValueError: If the censoring time is lower than the maximum transition time.
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


class CompleteModelData(ModelData):
    """Complete model data class.

    Use this to compute the log pdfs by hand if necessary. It is possible to simply
    call the `MultiStateJointModel` class as it inherits from `nn.Module`.

    Attributes:
        valid_mask (torch.Tensor): The mask of the valid measurements.
        n_valid (torch.Tensor): The number of valid measurements.
        valid_t (torch.Tensor): The valid times.
        valid_y (torch.Tensor): The valid measurements.
        buckets (dict[tuple[Any, Any], tuple[torch.Tensor, ...]]): The buckets for the
            trajectories.
    """

    valid_mask: torch.Tensor = field(init=False)
    n_valid: torch.Tensor = field(init=False)
    valid_t: torch.Tensor = field(init=False)
    valid_y: torch.Tensor = field(init=False)
    buckets: dict[tuple[Any, Any], tuple[torch.Tensor, ...]] = field(init=False)

    @validate_params(
        {
            "model_design": [ModelDesign],
            "params": [ModelParams],
        },
        prefer_skip_nested_validation=True,
    )
    def prepare(self, model_design: ModelDesign, params: ModelParams):
        """Sets the missing representation.

        Args:
            model_design (ModelDesign): The design of the model.
            params (ModelParams): The model parameters.
        """
        check_consistent_length(params.r.cov, self.y.transpose(0, -1))

        self.valid_mask = ~self.y.isnan()
        self.n_valid = self.valid_mask.sum(dim=-2).to(torch.get_default_dtype())
        self.valid_t = self.t.nan_to_num(self.t.nanmean().item())
        self.valid_y = self.y.nan_to_num()
        self.buckets = build_all_buckets(
            self.trajectories, self.c, tuple(model_design.surv.keys())
        )


@dataclass
class SampleData:
    """Dataclass for data used in sampling.

    Raises:
        ValueError: If the trajectories are not sorted by time.
        ValueError: If the censoring time is lower than the maximum transition time.
        ValueError: If any of the inputs contain inf values.
        ValueError: If any of the inputs contain NaN values.
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
    c: torch.Tensor | None = None

    def __len__(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)

    def __post_init__(self):
        """Runs the post init conversions and checks.

        Raises:
            ValueError: If the trajectories are not sorted by time.
            ValueError: If the censoring time is lower than the maximum transition time.
            ValueError: If any of the inputs contain inf values.
            ValueError: If any of the inputs contain NaN values.
            ValueError: If the size is not consistent between inputs.
        """
        validate_params(
            {
                "trajectories": [list],
                "psi": [torch.Tensor],
                "c": [torch.Tensor],
            },
            prefer_skip_nested_validation=True,
        )

        check_trajectories(self.trajectories, self.c)

        assert_all_finite(self.x, input_name="x")
        assert_all_finite(self.psi, input_name="psi")
        assert_all_finite(self.c, input_name="c")

        check_consistent_length(
            self.x, self.psi.transpose(0, -2), self.c, self.trajectories
        )
