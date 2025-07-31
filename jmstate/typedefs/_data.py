import itertools
from dataclasses import dataclass, field

import torch

from ..utils._surv import build_vec_rep
from ._defs import ModelDesign, Trajectory


@dataclass
class ModelData:
    """Dataclass containing learnable multistate joint model data.

    Raises:
            ValueError: If any tensor contains inf values.
            ValueError: If c is not 1D.
            ValueError: If x is not 2D.
            ValueError: If y is not 3D.
            ValueError: If the number of individuals is inconsistent.
            ValueError: If the shape of t is not broadcastable with y.
            ValueError: If t contains torch.nan where y is not.
            ValueError: If the trajectories are not sorted by time.
    """

    x: torch.Tensor | None
    t: torch.Tensor
    y: torch.Tensor
    trajectories: list[Trajectory]
    c: torch.Tensor
    skip_validation: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Runs the post init conversions.

        Raises:
            ValueError: If any tensor contains inf values.
            ValueError: If c is not 1D.
            ValueError: If x is not 2D.
            ValueError: If y is not 3D.
            ValueError: If the number of individuals is inconsistent.
            ValueError: If the shape of t is not broadcastable with y.
            ValueError: If t contains torch.nan where y is not.
            ValueError: If the trajectories are not sorted by time.
        """
        self.x = (
            torch.as_tensor(self.x, dtype=torch.float32) if self.x is not None else None
        )
        self.t = torch.as_tensor(self.t, dtype=torch.float32)
        self.y = torch.as_tensor(self.y, dtype=torch.float32)
        self.c = torch.as_tensor(self.c, dtype=torch.float32)

        if self.skip_validation:
            return

        for name, tensor in [
            ("c", self.c),
            ("x", self.x),
            ("y", self.y),
            ("t", self.t),
        ]:
            if tensor is not None and tensor.isinf().any():
                raise ValueError(f"{name} cannot contain inf values")

        # Check dimensions
        if self.c.ndim != 1:
            raise ValueError(f"c must be 1D, got {self.c.ndim}D")
        if self.x is not None and self.x.ndim != 2:
            raise ValueError(f"x must be None or 2D, got {self.x.ndim}D")
        if self.y.ndim != 3:
            raise ValueError(f"y must be 3D, got {self.y.ndim}D")

        # Check consistent size
        n = self.size
        if not (
            (self.x is None or self.x.shape[0] == n)
            and self.y.shape[0] == self.c.numel() == n
        ):
            raise ValueError("Inconsistent number of individuals")

        # Check time compatibility
        if self.t.shape not in ((self.y.shape[1],), self.y.shape[:2]):
            raise ValueError(f"Invalid t shape: {self.t.shape}")

        # Check for NaNs in t where y is valid
        if (
            self.t.shape == self.y.shape[:2]
            and (~self.y.isnan().all(dim=2) & self.t.isnan()).any()
        ):
            raise ValueError("t cannot be NaN where y is valid")

        # Check trajectory sorting
        if any(
            any(t0 > t1 for t0, t1 in itertools.pairwise(t for t, _ in trajectory))
            for trajectory in self.trajectories
        ):
            raise ValueError("Trajectories must be sorted by time")

        # Check if c is at least equal to the greatest part of each trajectory
        if any(
            trajectory[-1][0] > c for trajectory, c in zip(self.trajectories, self.c)
        ):
            raise ValueError("Last trajectory time must not be greater than c")

    @property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)


@dataclass
class CompleteModelData(ModelData):
    valid_mask: torch.Tensor = field(init=False)
    n_valid: torch.Tensor = field(init=False)
    valid_t: torch.Tensor = field(init=False)
    valid_y: torch.Tensor = field(init=False)
    buckets: dict[tuple[int, int], tuple[torch.Tensor, ...]] = field(init=False)

    def prepare(self, model_design: ModelDesign):
        self.valid_mask = (~torch.isnan(self.y)).to(torch.float32)
        self.n_valid = self.valid_mask.sum(dim=1)
        self.valid_t = torch.nan_to_num(self.t)
        self.valid_y = torch.nan_to_num(self.y)
        self.buckets = build_vec_rep(self.trajectories, self.c, model_design.surv)


@dataclass
class SampleData:
    """Dataclass for data used in sampling.

    Raises:
        ValueError: If any tensor contains inf values.
        ValueError: If c is not 1D or None.
        ValueError: If x is not 2D.
        ValueError: If psi is not 2D.
        ValueError: If the number of individuals is inconsistent.
        ValueError: If the trajectories are not sorted by time.
        ValueError: If the last trajectory time is greater than c.
    """

    x: torch.Tensor | None
    trajectories: list[Trajectory]
    psi: torch.Tensor
    c: torch.Tensor | None = None
    skip_validation: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Runs the post init conversions and checks."""
        self.x = (
            torch.as_tensor(self.x, dtype=torch.float32) if self.x is not None else None
        )
        self.c = (
            torch.as_tensor(self.c, dtype=torch.float32) if self.c is not None else None
        )
        self.psi = torch.as_tensor(self.psi, dtype=torch.float32)

        if self.skip_validation:
            return

        for name, tensor in [("c", self.c), ("x", self.x), ("psi", self.psi)]:
            if tensor is not None and tensor.isinf().any():
                raise ValueError(f"{name} cannot contain inf values")

        # Check dimensions
        if self.c is not None and self.c.ndim != 1:
            raise ValueError(f"c must be 1D, got {self.c.ndim}D")
        if self.x is not None and self.x.ndim != 2:
            raise ValueError(f"x must be None or 2D, got {self.x.ndim}D")
        if self.psi.ndim != 2:
            raise ValueError(f"psi must be 2D, got {self.psi.ndim}D")

        # Check consistent size
        n = self.size
        if not (
            (self.x is None or self.x.shape[0] == n)
            and self.psi.shape[0] == n
            and (self.c is None or self.c.numel() == n)
        ):
            raise ValueError("Inconsistent number of individuals")

        # Check trajectory sorting
        if any(
            any(t0 > t1 for t0, t1 in itertools.pairwise(t for t, _ in trajectory))
            for trajectory in self.trajectories
        ):
            raise ValueError("Trajectories must be sorted by time")

        # Check if c is at least equal to the greatest part of each trajectory
        if self.c is not None and any(
            trajectory[-1][0] > c for trajectory, c in zip(self.trajectories, self.c)
        ):
            raise ValueError("Last trajectory time must not be greater than c")

    @property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)
