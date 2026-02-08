import torch
from rich.console import Console
from rich.tree import Tree

from ..typedefs._defs import Any, Trajectory
from ..utils._surv import build_buckets


# Utils
def rich_str(obj: Any) -> str:
    """Get the rich string representation of an object.

    Args:
        obj (Any): The object to get the string representation of.

    Returns:
        str: The string representation.
    """
    console = Console()
    return console._render_buffer(console.render(obj))[:-1]  # type: ignore


# Descriptors
def add_x(x: torch.Tensor | None, tree: Tree):
    """Add the fixed covariates to the tree.

    Args:
        x (torch.Tensor | None): The fixed covariates.
        tree (Tree): The tree to add to.
    """
    if x is not None:
        tree.add(f"x: {x.size(0)} individual(s) x {x.size(1)} covariate(s)")


def add_t(t: torch.Tensor, tree: Tree):
    """Add the measurement times.

    Args:
        t (torch.Tensor): The measurement times.
        tree (Tree): The tree to add to.
    """
    if t.ndim == 1:
        tree.add(f"t: {t.size(0)} shared measurement(s)")
    else:
        tree.add(f"t: {t.size(0)} individual(s) x {t.size(1)} measurement(s)")


def add_y(y: torch.Tensor, tree: Tree):
    """Add the measurements.

    Args:
        y (torch.Tensor): The measurements.
        tree (Tree): The tree to add to.
    """
    tree.add(
        f"y: {y.size(0)} individual(s) x {y.size(1)} measurement(s) x {y.size(2)} "
        "dimension(s)"
    )


def add_psi(psi: torch.Tensor, tree: Tree):
    """Add the individual parameters.

    Args:
        psi (torch.Tensor): The individual parameters.
        tree (Tree): The tree to add to.
    """
    if psi.ndim == 2:  # noqa: PLR2004
        tree.add(f"psi: {psi.size(0)} individual(s) x {psi.size(1)} parameter(s)")
    else:
        tree.add(
            f"psi: {psi.size(0)} sample(s) x {psi.size(1)} individual(s) x "
            f"{psi.size(2)} parameter(s)"
        )


def add_trajectories(trajectories: list[Trajectory], tree: Tree):
    """Add the trajectories.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        tree (Tree): The tree to add to.
    """
    buckets = build_buckets(trajectories)

    node = tree.add(
        f"trajectories: {len(trajectories)} individual(s) with "
        f"{sum(len(t) - 1 for t in trajectories)} observed transitions"
    )
    for k, v in buckets.items():
        node.add(f"{k[0]} --> {k[1]}: {v.idxs.numel()}")


def add_c(c: torch.Tensor | None, tree: Tree):
    """Add the censoring times.

    Args:
        c (torch.Tensor | None): The censoring times.
        tree (Tree): The tree to add to.
    """
    if c is not None:
        tree.add(f"c: {c.size(0)} censoring time(s)")
