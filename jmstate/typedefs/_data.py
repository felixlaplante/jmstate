import warnings
from collections.abc import Mapping
from dataclasses import field
from functools import cached_property
from typing import Any

import torch
from pydantic import ConfigDict, dataclasses

from ..utils._checks import (
    check_consistent_size,
    check_inf,
    check_nan,
    check_trajectory_c,
    check_trajectory_empty,
    check_trajectory_sorting,
)
from ..utils._dtype import get_dtype
from ..utils._surv import build_all_buckets
from ._defs import (
    BaseHazardFn,
    IndividualEffectsFn,
    LinkFn,
    RegressionFn,
    Tensor1D,
    Tensor2D,
    Tensor3D,
    TensorCol,
    Trajectory,
)
from ._params import ModelParams


# Dataclasses
@dataclasses.dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
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


@dataclasses.dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
class ModelData:
    r"""Dataclass containing learnable multistate joint model data.

    Not `y` is expected to be a 3D tensor of dimension :math:`(n, m, d)` if there are
    :math:`n` individual, with a maximum number of :math:`m` measurements in
    :math:`\mathbb{R}^d`. Its values should not be all NaNs.

    Bypass checks by activating the `skip_validation` flag.

    Raises:
        ValueError: If the trajectories are not sorted by time.
        ValueError: If the censoring time is lower than the maximum transition time.
        ValueError: If any of the inputs contain inf values.
        ValueError: If any of the inputs contain NaN values except `y`.
        ValueError: If the size is not consistent between inputs.

    Attributes:
        x (Tensor2D | None): The fixed covariates.
        t (Tensor1D | Tensor2D): The measurement times. Either a 1D tensor if the
            times are shared by all individual, or a matrix of individual times.
            Use padding with NaNs when necessary.
        y (Tensor3D): The measurements. A 3D tensor of dimension :math:`(n, m, d)`
            if there are :math:`n` individual, with a maximum number of :math:`m`
            measurements in :math:`\mathbb{R}^d`. Use padding with NaNs when
            necessary.
        trajectories (list[Trajectory]): The list of the individual trajectories.
            A `Trajectory` is a list of tuples containing time and state.
        c (TensorCol): The censoring times as a column vector. They must not
            be less than the trajectory maximum times.
        skip_validation (bool): Whether to skip validatoin or not. Defaults to False.
    """

    x: Tensor2D | None
    t: Tensor1D | Tensor2D
    y: Tensor3D
    trajectories: list[Trajectory]
    c: TensorCol
    skip_validation: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Runs the post init conversions.

        Raises:
            ValueError: If the trajectories are not sorted by time.
            ValueError: If the censoring time is lower than the maximum transition time.
            ValueError: If any of the inputs contain inf values.
            ValueError: If any of the inputs contain NaN values.
            ValueError: If the size is not consistent between inputs.
        """
        if self.skip_validation:
            return

        dtype = get_dtype()

        object.__setattr__(self, "x", None if self.x is None else self.x.to(dtype))
        object.__setattr__(self, "t", self.t.to(dtype))
        object.__setattr__(self, "y", self.y.to(dtype))
        object.__setattr__(self, "c", self.c.to(dtype))

        check_trajectory_empty(self.trajectories)
        check_trajectory_sorting(self.trajectories)
        check_trajectory_c(self.trajectories, self.c)

        last_times = torch.tensor(
            [trajectory[-1][0] for trajectory in self.trajectories], dtype=dtype
        )

        check_inf(
            (
                (self.x, "x"),
                (self.t, "t"),
                (self.y, "y"),
                (self.c, "c"),
                (last_times, "trajectories"),
            )
        )
        check_nan(((self.x, "x"), (self.c, "c")))
        check_consistent_size(
            (
                (self.x, 0, "x"),
                (self.y, 0, "y"),
                (self.c, 0, "c"),
                (self.size, None, "trajectories"),
            )
        )
        check_consistent_size(((self.t, -1, "t"), (self.y, -2, "y")))

        # Check NaNs between t and y
        if ((~self.y.isnan()).any(dim=-1) & self.t.isnan()).any():
            raise ValueError("NaN time values on non NaN y values")

    @cached_property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)

    def effective_size(self, indep_residuals: bool) -> int:
        """Gets the effective size of the dataset, used for BIC.

        Args:
            indep_residuals (bool): Whether or not to assume the independence of the
                residuals.

        Returns:
            int: The effective size.
        """
        n_long = (
            int((~torch.isnan(self.y)).sum())
            if indep_residuals
            else int((~torch.isnan(self.y)).any(dim=-1).sum())
        )

        return n_long + sum(len(trajectory) for trajectory in self.trajectories)


class CompleteModelData(ModelData):
    """Complete model data class."""

    valid_mask: Tensor3D = field(init=False)
    n_valid: Tensor2D = field(init=False)
    valid_t: Tensor1D | Tensor2D = field(init=False)
    valid_y: Tensor2D | Tensor3D = field(init=False)
    buckets: dict[tuple[Any, Any], tuple[torch.Tensor, ...]] = field(init=False)

    def prepare(self, model_design: ModelDesign, params: ModelParams):
        """Sets the missing representation.

        Raises:
            ValueError: If y and R are not compatible in shape.

        Args:
            model_design (ModelDesign): The design of the model.
            params (ModelParams): The model parameters.
        """
        check_consistent_size(((params.R_repr.dim, None, "R"), (self.y, -1, "y")))

        nan_mask = self.y.isnan()
        valid_mask = ~nan_mask
        self.valid_mask = valid_mask.to(get_dtype())
        self.n_valid = self.valid_mask.sum(dim=-2)
        self.valid_t = self.t.nan_to_num(self.t.nanmean().item())
        self.valid_y = self.y.nan_to_num()
        self.buckets = build_all_buckets(
            self.trajectories, self.c, tuple(model_design.surv.keys())
        )

        if (
            params.R_repr.method == "full"
            and (valid_mask.any(dim=-1) & nan_mask.any(dim=-1)).any()
        ):
            warnings.warn(
                (
                    "R method should not be full when having mixed NaNs as incorrect "
                    "likelihood will be computed"
                ),
                stacklevel=2,
            )


@dataclasses.dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
class SampleData:
    """Dataclass for data used in sampling.

    Bypass checks by activating the `skip_validation` flag.

    Raises:
        ValueError: If the trajectories are not sorted by time.
        ValueError: If the censoring time is lower than the maximum transition time.
        ValueError: If any of the inputs contain inf values.
        ValueError: If any of the inputs contain NaN values.
        ValueError: If the size is not consistent between inputs.

    Attributes:
        x (Tensor2D | None): The fixed covariates.
        trajectories (list[Trajectory]): The list of the individual trajectories.
            A `Trajectory` is a list of tuples containing time and state.
        psi (Tensor2D | Tensor3D): The individual parameters. Define it as a matrix with
            the same number of rows as there are `len(trajectories)`. Only use a 3D
            tensor if you fully understand the codebase and the mechanisms. Trajectory
            sampling may only be used with matrices.
        c (TensorCol | None, optional): The censoring times as a column vector. They
            must not be less than the trajectory maximum times. This corresponds to
            the last times of observation of the individuals or prediction current
            times.
        skip_validation (bool): Whether to skip validatoin or not. Defaults to False.
    """

    x: Tensor2D | None
    trajectories: list[Trajectory]
    psi: Tensor2D | Tensor3D
    c: TensorCol | None = None
    skip_validation: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Runs the post init conversions and checks."""
        if self.skip_validation:
            return

        dtype = get_dtype()

        object.__setattr__(self, "x", None if self.x is None else self.x.to(dtype))
        object.__setattr__(self, "psi", self.psi.to(dtype))
        object.__setattr__(self, "c", None if self.c is None else self.c.to(dtype))

        check_trajectory_empty(self.trajectories)
        check_trajectory_sorting(self.trajectories)
        check_trajectory_c(self.trajectories, self.c)

        last_times = torch.tensor(
            [trajectory[-1][0] for trajectory in self.trajectories], dtype=dtype
        )

        check_inf(
            (
                (self.x, "x"),
                (self.psi, "psi"),
                (self.c, "c"),
                (last_times, "trajectories"),
            )
        )
        check_nan(((self.x, "x"), (self.psi, "psi"), (self.c, "c")))
        check_consistent_size(
            (
                (self.x, -2, "x"),
                (self.psi, -2, "psi"),
                (self.c, -2, "c"),
                (self.size, None, "trajectories"),
            )
        )

    @cached_property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)
