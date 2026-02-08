import torch
from pydantic import ConfigDict, validate_call

from ..typedefs._defs import ValidDtype

# Default dtype
_dtype = torch.float32


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def set_dtype(dtype: ValidDtype):
    """Set the default dtype. It must be either `torch.float32` or `torch.float64`.

    It defaults to `torch.float32`.

    Args:
        dtype (ValidDtype): The dtype to set.
    """
    global _dtype  # noqa: PLW0603
    _dtype = dtype


def get_dtype() -> torch.dtype:
    """Get the default dtype. It is either `torch.float32` or `torch.float64`.

    It defaults to `torch.float32`.

    Returns:
        torch.dtype: The default dtype.
    """
    return _dtype
