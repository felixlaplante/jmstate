"""Multi-state joint modeling package.

This package provides tools for multi-state joint modeling with PyTorch.
"""

from . import functions, jobs, typedefs, utils  # type: ignore # noqa: F401
from .model import MultiStateJointModel

__author__ = "Félix Laplante"
__email__ = "felixlaplante0@gmail.com"
__license__ = "MIT"

__all__ = ["MultiStateJointModel"]
