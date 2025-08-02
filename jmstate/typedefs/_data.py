from dataclasses import dataclass, field

import torch
from beartype import beartype

from ..utils._checks import (
    check_consistent_size,
    check_inf,
    check_trajectory_c,
    check_trajectory_sorting,
)
from ..utils._surv import build_vec_rep
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
    VecRep,
)


# Dataclasses
@beartype
@dataclass
class ModelDesign:
    """Class containing all multistate joint model design."""

    individual_effects_fn: IndividualEffectsFn
    regression_fn: RegressionFn
    surv: dict[
        tuple[int, int],
        tuple[
            BaseHazardFn,
            LinkFn,
        ],
    ]


@beartype
@dataclass
class ModelData:
    """Dataclass containing learnable multistate joint model data.

    Raises:
            ValueError: If the shape of t is not broadcastable with y.
            ValueError: If t contains torch.nan where y is not.
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
            ValueError: If the shape of t is not broadcastable with y.
            ValueError: If t contains torch.nan where y is not.
        """
        if self.skip_validation:
            return

        self.x = None if self.x is None else self.x.to(torch.float32)
        self.t = self.t.to(torch.float32)
        self.y = self.y.to(torch.float32)
        self.c = self.c.to(torch.float32)

        # Check NaNs
        if (
            self.t.shape == self.y.shape[:2]
            and (~self.y.isnan().all(dim=2) & self.t.isnan()).any()
        ):
            raise ValueError("Invalid time values on non NaN y values")

        check_inf((self.x, self.t, self.y, self.c))
        check_consistent_size((self.x, self.y, self.c), (0, 0, 0), self.size)
        check_trajectory_sorting(self.trajectories)
        check_trajectory_c(self.trajectories, self.c)

    @property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)


@beartype
@dataclass
class CompleteModelData(ModelData):
    valid_mask: Tensor3D = field(init=False)
    n_valid: Tensor2D = field(init=False)
    valid_t: Tensor1D | Tensor2D = field(init=False)
    valid_y: Tensor2D | Tensor3D = field(init=False)
    buckets: dict[tuple[int, int], VecRep] = field(init=False)
    n_chains: int = field(init=False)

    def init(self, model_design: ModelDesign, n_chains: int):
        self.valid_mask = (~torch.isnan(self.y)).to(torch.float32)
        self.n_valid = self.valid_mask.sum(dim=1)
        self.valid_t = torch.nan_to_num(self.t)
        self.valid_y = torch.nan_to_num(self.y)
        self.buckets = build_vec_rep(self.trajectories, self.c, model_design.surv)
        self.n_chains = n_chains


@beartype
@dataclass
class SampleData:
    """""Dataclass for data used in sampling.""" ""

    x: Tensor2D | None
    trajectories: list[Trajectory]
    psi: Tensor2D | Tensor3D
    c: TensorCol | None = None
    skip_validation: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Runs the post init conversions and checks."""
        if self.skip_validation:
            return

        self.x = None if self.x is None else self.x.to(torch.float32)
        self.c = None if self.c is None else self.c.to(torch.float32)
        self.psi = self.psi.to(torch.float32)

        check_inf((self.c, self.x, self.psi))
        check_consistent_size((self.x, self.psi, self.c), (0, -2, 0), self.size)
        check_trajectory_sorting(self.trajectories)
        check_trajectory_c(self.trajectories, self.c)

    @property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)
