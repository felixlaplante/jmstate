import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import torch
from pydantic import ConfigDict, dataclasses

from ..utils._checks import (
    check_consistent_size,
    check_inf,
    check_trajectory_c,
    check_trajectory_empty,
    check_trajectory_sorting,
)
from ..utils._surv import build_traj_repr
from ._defs import (
    HazardFns,
    IndividualEffectsFn,
    RegressionFn,
    Tensor1D,
    Tensor2D,
    Tensor3D,
    TensorCol,
    Trajectory,
    TrajRepr,
)
from ._params import ModelParams


# Dataclasses
@dataclasses.dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelDesign:
    """Class containing model design. Base hazard function is expected to be pure."""

    individual_effects_fn: IndividualEffectsFn
    regression_fn: RegressionFn
    surv: dict[
        tuple[Any, Any],
        HazardFns,
    ]


@dataclasses.dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelData:
    """Dataclass containing learnable multistate joint model data.

    Raises:
        ValueError: If the tensors contain inf.
        ValueError: If the size is not consistent between tensors.
        ValueError: If the trajectories are not sorted by time.
        ValueError: If the censoring time is lower than the maximum transition time.
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
            ValueError: If the tensors contain inf.
            ValueError: If the size is not consistent between tensors.
            ValueError: If the trajectories are not sorted by time.
            ValueError: If the censoring time is lower than the maximum transition time.
        """
        if self.skip_validation:
            return

        self.x = None if self.x is None else self.x.to(torch.get_default_dtype())
        self.t = self.t.to(torch.get_default_dtype())
        self.y = self.y.to(torch.get_default_dtype())
        self.c = self.c.to(torch.get_default_dtype())

        check_inf(((self.x, "x"), (self.t, "t"), (self.y, "y"), (self.c, "c")))
        check_consistent_size(
            (
                (self.x, 0, "x"),
                (self.y, 0, "y"),
                (self.c, 0, "c"),
                (self.size, None, "trajectories"),
            )
        )
        check_consistent_size(((self.t, -1, "t"), (self.y, -2, "y")))
        check_trajectory_empty(self.trajectories)
        check_trajectory_sorting(self.trajectories)
        check_trajectory_c(self.trajectories, self.c)

        # Check NaNs
        if ((~self.y.isnan()).any(dim=-1) & self.t.isnan()).any():
            raise ValueError("NaN time values on non NaN y values")

    @cached_property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)

    @cached_property
    def effective_size(self) -> int:
        """Gets the effective size of the dataset, used for BIC.

        Returns:
            int: The effective size.
        """
        return int((~torch.isnan(self.y)).any(dim=-1).sum()) + sum(
            len(trajectory) for trajectory in self.trajectories
        )


@dataclass
class CompleteModelData(ModelData):
    valid_mask: Tensor3D = field(init=False)
    n_valid: Tensor2D = field(init=False)
    valid_t: Tensor1D | Tensor2D = field(init=False)
    valid_y: Tensor2D | Tensor3D = field(init=False)
    buckets: dict[tuple[Any, Any], TrajRepr] = field(init=False)

    def init(self, model_design: ModelDesign, params: ModelParams):
        """Sets the missing representation.

        Raises:
            ValueError: If y and R are not compatible in shape.

        Args:
            model_design (ModelDesign): The design of the model.
            params (ModelParams): The model parameters.
        """
        check_consistent_size(((params.R_repr.dim, None, "R"), (self.y, -1, "y")))

        nan_mask = torch.isnan(self.y)
        valid_mask = ~nan_mask
        self.valid_mask = valid_mask.to(torch.get_default_dtype())
        self.n_valid = self.valid_mask.sum(dim=-2)
        self.valid_t = torch.nan_to_num(self.t)
        self.valid_y = torch.nan_to_num(self.y)
        self.buckets = build_traj_repr(self.trajectories, self.c, model_design.surv)

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


@dataclasses.dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class SampleData:
    """Dataclass for data used in sampling.

    Raises:
        ValueError: If the tensors contain inf.
        ValueError: If the size is not consistent between tensors.
        ValueError: If the trajectories are not sorted by time.
        ValueError: If the censoring time is lower than the maximum transition time.
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

        self.x = None if self.x is None else self.x.to(torch.get_default_dtype())
        self.psi = self.psi.to(torch.get_default_dtype())
        self.c = None if self.c is None else self.c.to(torch.get_default_dtype())

        check_inf(((self.x, "x"), (self.psi, "psi"), (self.c, "c")))
        check_consistent_size(
            (
                (self.x, -2, "x"),
                (self.psi, -2, "psi"),
                (self.c, -2, "c"),
                (self.size, None, "trajectories"),
            )
        )
        check_trajectory_empty(self.trajectories)
        check_trajectory_sorting(self.trajectories)
        check_trajectory_c(self.trajectories, self.c)

    @cached_property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)
