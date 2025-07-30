import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isqrt
from typing import TYPE_CHECKING, Any

import torch

from ..utils._linalg import cov_from_flat
from ..utils._surv import build_vec_rep
from ._defs import BaseHazardFn, IndividualEffectsFn, LinkFn, RegressionFn, Trajectory

if TYPE_CHECKING:
    from ..model._base import MetropolisHastingsSampler, MultiStateJointModel


@dataclass
class ModelDesign:
    """Class containing all multistate joint model design.

    Raises:
        TypeError: If f is not callable.
        TypeError: If h is not callable.
        TypeError: If any of the base hazard functions is not callable.
        TypeError: If any of the link functions is not callable.
        ValueError: If the keys of alpha_dims and surv do not match.
    """

    individual_effects_fn: IndividualEffectsFn
    regression_fn: RegressionFn
    surv: dict[
        tuple[int, int],
        tuple[
            BaseHazardFn,
            LinkFn,
        ],
    ]

    def __post_init__(self):
        """Runs the checks themselves.

        Raises:
            TypeError: If f is not callable.
            TypeError: If h is not callable.
            TypeError: If any of the base hazard functions is not callable.
            TypeError: If any of the link functions is not callable.
            ValueError: If the keys of alpha_dims and surv do not match.
        """
        if not callable(self.individual_effects_fn):
            raise TypeError("individual_effects_fn must be callable")
        if not callable(self.regression_fn):
            raise TypeError("regression_fn must be callable")

        for key, (base_hazard_fn, link_fn) in self.surv.items():
            if not callable(base_hazard_fn):
                raise TypeError(f"Base hazard function for key {key} must be callable")
            if not callable(link_fn):
                raise TypeError(f"Link function for key {key} must be callable")


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
    extra_: dict[Any, Any] = field(default_factory=dict[str, Any], repr=False)

    def __post_init__(self):
        """Runs the post init conversions and checks."""
        self.x = (
            torch.as_tensor(self.x, dtype=torch.float32) if self.x is not None else None
        )
        self.t = torch.as_tensor(self.t, dtype=torch.float32)
        self.y = torch.as_tensor(self.y, dtype=torch.float32)
        self.c = torch.as_tensor(self.c, dtype=torch.float32)

        self._check()

    def _check(self):
        """Validate tensor dimensions and consistency.

        Raises:
            ValueError: If any tensor contains inf values.
            ValueError: If c is not 1D.
            ValueError: If x is not 2D or None.
            ValueError: If y is not 3D.
            ValueError: If the number of individuals is inconsistent.
            ValueError: If the shape of t is not broadcastable with y.
            ValueError: If t contains torch.nan where y is not.
            ValueError: If the trajectories are not sorted by time.
        """
        # Check for inf tensors
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

    def prepare(self, model_design: ModelDesign) -> None:
        """Add derived quantities.

        Args:
            model_design (ModelDesign): The current model design.

        Raises:
            TypeError: If self.params_.betas is None and x is not None or the other way.
        """
        # Add derived quantities
        self.extra_["valid_mask"] = (~torch.isnan(self.y)).to(torch.float32)
        self.extra_["n_valid"] = self.extra_["valid_mask"].sum(dim=1)
        self.extra_["valid_t"] = torch.nan_to_num(self.t)
        self.extra_["valid_y"] = torch.nan_to_num(self.y)
        self.extra_["buckets"] = build_vec_rep(
            self.trajectories, self.c, model_design.surv
        )

    def clear_extra(self) -> None:
        """Clears extra information."""
        self.extra_ = {}

    @property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)


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

    def __post_init__(self):
        """Runs the post init conversions and checks."""
        self.x = (
            torch.as_tensor(self.x, dtype=torch.float32) if self.x is not None else None
        )
        self.c = (
            torch.as_tensor(self.c, dtype=torch.float32) if self.c is not None else None
        )
        self.psi = torch.as_tensor(self.psi, dtype=torch.float32)

        self._check()

    def _check(self):
        """Validate tensor dimensions and consistency.

        Raises:
            ValueError: If any tensor contains inf values.
            ValueError: If c is not 1D or None.
            ValueError: If x is not 2D.
            ValueError: If psi is not 2D.
            ValueError: If the number of individuals is inconsistent.
            ValueError: If the trajectories are not sorted by time.
            ValueError: If the last trajectory time is greater than c
        """
        # Check for inf tensors
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


@dataclass
class ModelParams:
    """Dataclass containing model parameters.

    Raises:
        ValueError: If any of the main tensors contains inf.
        ValueError: If any of the main tensors is not 1D.
        ValueError: If any of the alpha tensors contains inf.
        ValueError: If any of the alpha tensors is not 1D.
        ValueError: If any of the beta tensors contains inf.
        ValueError: If any of the beta tensors is not 1D.
        ValueError: If the name matrix is not "Q" nor "R".
        ValueError: If the number of elements is not triangular and method is "full".
        ValueError: If the number of elements is not one and the method is "ball".
        ValueError: If the name matrix is not "Q" nor "R".
        ValueError: If the name matrix is not "Q" nor "R".
    """

    gamma: torch.Tensor | None
    Q_repr: tuple[torch.Tensor, str]
    R_repr: tuple[torch.Tensor, str]
    alphas: dict[tuple[int, int], torch.Tensor]
    betas: dict[tuple[int, int], torch.Tensor] | None
    Q_dim_: int = field(init=False, repr=False)
    R_dim_: int = field(init=False, repr=False)

    def __post_init__(self):
        """Convert and init to float32 the parameters."""
        Q_flat, Q_method = self.Q_repr
        R_flat, R_method = self.R_repr

        Q_flat = torch.as_tensor(Q_flat, dtype=torch.float32)
        R_flat = torch.as_tensor(R_flat, dtype=torch.float32)

        # Update representation tuples
        self.Q_repr = (Q_flat, Q_method)
        self.R_repr = (R_flat, R_method)

        self.gamma = (
            torch.as_tensor(self.gamma, dtype=torch.float32)
            if self.gamma is not None
            else None
        )

        for key, alpha in self.alphas.items():
            self.alphas[key] = torch.as_tensor(alpha, dtype=torch.float32)

        if self.betas is not None:
            for key, beta in self.betas.items():
                self.betas[key] = torch.as_tensor(beta, dtype=torch.float32)

        self._check()
        self._set_dims("Q")
        self._set_dims("R")

    def _check(self):
        """Validate all tensors are 1D and don't contain inf.

        Raises:
            ValueError: If any of the main tensors contains inf.
            ValueError: If any of the main tensors is not 1D.
            ValueError: If any of the alpha tensors contains inf.
            ValueError: If any of the alpha tensors is not 1D.
            ValueError: If any of the beta tensors contains inf.
            ValueError: If any of the beta tensors is not 1D.
        """
        # Check main tensors
        for name, tensor in [
            ("gamma", self.gamma),
            ("Q_flat_", self.Q_repr[0]),
            ("R_flat_", self.R_repr[0]),
        ]:
            if tensor is None:
                continue
            if tensor.isinf().any():
                raise ValueError(f"{name} contains inf")
            if tensor.ndim != 1:
                raise ValueError(f"{name} must be 1D")

        # Check dictionary tensors
        for key, alpha in self.alphas.items():
            if alpha.isinf().any():
                raise ValueError(f"alpha {key} contains inf")
            if alpha.ndim != 1:
                raise ValueError(f"alpha {key} must be 1D")

        if self.betas is not None:
            for key, beta in self.betas.items():
                if beta.isinf().any():
                    raise ValueError(f"beta {key} contains inf")
                if beta.ndim != 1:
                    raise ValueError(f"beta {key} must be 1D")

    def _set_dims(self, matrix: str) -> None:
        """Sets dimensions for matrix.

        Args:
            matrix (str): Either "Q" or "R".

        Raises:
            ValueError: If the name matrix is not "Q" nor "R".
            ValueError: If the number of elements is not triangular with method "full".
            ValueError: If the number of elements is not one and the method is "ball".
        """
        if matrix not in ("Q", "R"):
            raise ValueError(f"matrix should be either Q or R, got {matrix}")

        flat, method = getattr(self, matrix + "_repr")

        match method:
            case "full":
                n = (isqrt(1 + 8 * flat.numel()) - 1) // 2
                if (n * (n + 1)) // 2 != flat.numel():
                    raise ValueError(
                        f"{flat.numel()} is not a triangular number for matrix {matrix}"
                    )
                setattr(self, matrix + "_dim_", n)
            case "diag":
                n = flat.numel()
                setattr(self, matrix + "_dim_", n)
            case "ball":
                if flat.numel() != 1:
                    f"Excepected 1 element for flat, got {flat.numel()}"
                setattr(self, matrix + "_dim_", 1)
            case _:
                raise ValueError(f"Got method {method} unknown for matrix {matrix}")

    @property
    def as_list(self) -> list[torch.Tensor]:
        """Get a list of all the parameters for optimization.

        Returns:
            list[torch.Tensor]: The list of the parameters.
        """
        iterables: Iterable[torch.Tensor | list[torch.Tensor]] = (
            [self.gamma] if self.gamma is not None else [],
            [self.Q_repr[0], self.R_repr[0]],
            list(self.alphas.values()),
            list(self.betas.values()) if self.betas is not None else [],
        )

        return list(itertools.chain.from_iterable(iterables))

    @property
    def as_flat_tensor(self) -> torch.Tensor:
        """Get the flattened parameters.

        Returns:
            torch.Tensor: The flattened parameters.
        """
        return torch.cat(self.as_list)

    @property
    def numel(self) -> int:
        """Return the number of parameters.

        Returns:
            int: The number of the parameters.
        """
        return sum(p.numel() for p in self.as_list)

    def require_grad(self, req: bool):
        """Enable gradient computation on all parameters.

        Args:
            req (bool): Wether to require or not.
        """
        # Enable or diasable gradients
        for tensor in self.as_list:
            tensor.requires_grad_(req)

    def get_cov(self, matrix: str) -> torch.Tensor:
        """Get covariance from parameter.

        Args:
            matrix (str): Either "Q" or "R".

        Raises:
            ValueError: If the matrix is not in ("Q", "R")

        Returns:
            torch.Tensor: The covariance matrix.
        """
        if matrix not in ("Q", "R"):
            raise ValueError(f"matrix must be either Q or R, got {matrix}")

        # Get flat then covariance
        flat, method = getattr(self, matrix + "_repr")
        n = getattr(self, matrix + "_dim_")

        return cov_from_flat(flat, n, method)


@dataclass
class BaseInfo:
    data: "ModelData"
    iteration: int
    n_iterations: int
    model: "MultiStateJointModel"
    sampler: "MetropolisHastingsSampler"
    optimizer: torch.optim.Optimizer | None


@dataclass
class AllInfo(BaseInfo):
    b: torch.Tensor
    logpdfs: torch.Tensor
    logliks: torch.Tensor


# Abstract
class Job(ABC):
    @abstractmethod
    def init(self, info: BaseInfo, metrics: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def run(self, info: AllInfo, metrics: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def end(self, info: BaseInfo, metrics: dict[str, Any]) -> None:
        pass
