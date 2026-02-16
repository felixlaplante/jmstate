from collections.abc import Iterable

import torch
from sklearn.utils._param_validation import validate_params  # type: ignore


@validate_params(
    {"parameters": [Iterable]},
    prefer_skip_nested_validation=True,
)
def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flatten an iterable of parameters into a single vector.

    Exact copy of ``torch.nn.utils.parameters_to_vector`` for convenience.

    Args:
        parameters (Iterable[Tensor]): an iterable of Tensors that are the parameters of
            a model.

    Returns:
        The parameters represented by a single vector.
    """
    vec: list[torch.Tensor] = []
    for param in parameters:
        vec.append(param.view(-1))
    return torch.cat(vec)


@validate_params(
    {"vec": [torch.Tensor], "parameters": [Iterable]},
    prefer_skip_nested_validation=True,
)
def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]):
    """Copy slices of a vector into an iterable of parameters.

    Rewrite the ``torch.nn.utils.vector_to_parameters`` function to allow for inplace
    copy of the parameters, mandatory for shared parameters.

    Args:
        vec (Tensor): a single vector representing the parameters of a model.
        parameters (Iterable[Tensor]): an iterable of Tensors that are the parameters of
            a model.
    """
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param.data.copy_(vec[pointer : (pointer := pointer + num_param)].view_as(param))
