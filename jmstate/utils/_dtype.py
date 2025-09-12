import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import ValidDtype


class _DTypeManager:
    """Manages the default dtype for the jmstate package."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        """Initialize the manager with a default dtype.

        Args:
            dtype (torch.dtype): The default dtype.
        """
        self._dtype = dtype

    def set_dtype(self, dtype: torch.dtype) -> None:
        """Set the default dtype.

        Args:
            dtype (torch.dtype): The dtype to set.
        """
        self._dtype = dtype

    def get_dtype(self) -> torch.dtype:
        """Get the default dtype."""
        return self._dtype

    def __call__(self) -> torch.dtype:
        """Allow the manager to be called directly to get the dtype."""
        return self._dtype


_dtype_manager = _DTypeManager()


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def set_dtype(dtype: ValidDtype):
    """Set the default dtype. It must be either `torch.float32` or `torch.float64`.

    It defaults to `torch.float32`.

    Args:
        dtype (ValidDtype): The dtype to set.
    """
    _dtype_manager.set_dtype(dtype)


def get_dtype() -> torch.dtype:
    """Get the default dtype. It is either `torch.float32` or `torch.float64`.

    It defaults to `torch.float32`.

    Returns:
        torch.dtype: The default dtype.
    """
    return _dtype_manager.get_dtype()
