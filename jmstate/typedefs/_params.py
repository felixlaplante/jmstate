from collections.abc import Iterator
from numbers import Integral
from typing import Any, Self, cast

import torch
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils._param_validation import (  # type: ignore
    Interval,  # type: ignore
    StrOptions,  # type: ignore
    validate_params,  # type: ignore
)
from sklearn.utils.validation import assert_all_finite  # type: ignore
from torch import nn

from ..utils._checks import check_matrix_dim
from ..utils._linalg import flat_from_log_cholesky, log_cholesky_from_flat


class UniqueParams(nn.Module):
    """`nn.Module` that has unique parameters."""

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return an iterator over the unique parameters.

        Args:
            recurse (bool, optional): Whether to recurse into submodules. Defaults to
                True.

        Returns:
            Iterator[nn.Parameter]: An iterator over the unique parameters.
        """
        seen: set[int] = set()
        for param in super().parameters(recurse):
            if (ptr := param.data_ptr()) not in seen:
                seen.add(ptr)
                yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Return an iterator over the unique parameters.

        Args:
            prefix (str, optional): The prefix to prepend to the parameter names.
                Defaults to "".
            recurse (bool, optional): Whether to recurse into submodules. Defaults to
                True.
            remove_duplicate (bool, optional): Whether to remove duplicate parameters.
                Defaults to True.

        Returns:
            Iterator[nn.Parameter]: An iterator over the unique parameters.
        """
        seen: set[int] = set()
        for name, param in super().named_parameters(prefix, recurse, remove_duplicate):
            if (ptr := param.data_ptr()) not in seen:
                seen.add(ptr)
                yield name, param


class CovParam(BaseEstimator, nn.Module):
    r"""`nn.Module` containing covariance parameters.

    Note three types of covariance matrices parametrization are provided: scalar
    matrix; diagonal matrix; full matrix. Defaults to the full matrix parametrization.
    This is achieved through a log Cholesky parametrization of the inverse covariance
    matrix. Formally, consider :math:`P = \Sigma^{-1}` the precision matrix and let
    :math:`L` be the Cholesky factor with positive diagonal elements, the log Cholseky
    is given by:

    .. math::
        \tilde{L}_{ij} = L_{ij}, \, i > j,

    and:

    .. math::
        \tilde{L}_{ii} = \log L_{ii}.

    This is very numerically stable and fast, as it doesn't require inverting the
    matrix when computing quadratic forms. The log determinant is then equal to:

    .. math::

        \log \det P = 2 \operatorname{Tr}(\tilde{L}).

    You can use these methods by creating the appropriate `MatRepr` with methods of
    `ball`, `diag` or `full`.

    Additionnally, if your data has mixed missing values, do not use `full` matrix
    parametrization for the residuals, as is this case the components must be
    independent.

    Attributes:
        flat (torch.Tensor): The flat representation of the covariance matrix.
        dim (int): The dimension of the covariance matrix.
        method (str): The method used to parametrize the covariance matrix.
    """

    @classmethod
    @validate_params(
        {
            "V": [torch.Tensor],
            "method": [StrOptions({"full", "diag", "ball"})],
        },
        prefer_skip_nested_validation=True,
    )
    def from_cov(cls, V: torch.Tensor, method: str = "full") -> Self:
        r"""Gets representation from covariance matrix according to choice of method.

        Args:
            V (torch.Tensor): The square covariance matrix parameter.
            method (str, optional): The method, `full`, `diag` or `ball`. Defaults to
                `full`.

        Returns:
            Self: The usable representation.
        """
        L = cast(torch.Tensor, torch.linalg.cholesky(V.inverse()))  # type: ignore
        L.diagonal().log_()
        return cls(flat_from_log_cholesky(L, method), L.size(0), method)

    @validate_params(
        {
            "flat": [torch.Tensor],
            "dim": [Interval(Integral, 1, None, closed="left")],
            "method": [StrOptions({"full", "diag", "ball"})],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, flat: torch.Tensor, dim: int, method: str):
        """Initializes the `CovParam` object.

        Args:
            flat (torch.Tensor): The flat representation of the covariance matrix.
            dim (int): The dimension of the covariance matrix.
            method (str): The method used to parametrize the covariance matrix.

        Raises:
            ValueError: If the representation is invalid.
        """
        super().__init__()

        self.flat = nn.Parameter(flat)
        self.dim = dim
        self.method = method

        check_matrix_dim(self.flat, self.dim, self.method)

    @property
    def cov(self) -> torch.Tensor:
        """Gets the covariance matrix.

        Returns:
            torch.Tensor: The covariance matrix.
        """
        L = log_cholesky_from_flat(self.flat, self.dim, self.method)
        L.diagonal().exp_()
        return torch.cholesky_inverse(L)

    @property
    def _inv_cholesky_and_log_eigvals(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets Cholesky factor as well as log eigvals.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Precision matrix and log eigvals.
        """
        L = log_cholesky_from_flat(self.flat, self.dim, self.method)
        log_eigvals = 2 * L.diag()
        L.diagonal().exp_()

        return L, log_eigvals


class ModelParams(BaseEstimator, UniqueParams):
    r"""`nn.Module` containing model parameters.

    Shared parameters are possible by assigning the exact same object to multiple fields
    so that the data pointer is the same. `self.parameters()` will not return duplicate
    parameters and is safe to use.

    Note three types of covariance matrices parametrization are provided: scalar
    matrix; diagonal matrix; full matrix. Defaults to the full matrix parametrization.
    This is achieved through a log Cholesky parametrization of the inverse covariance
    matrix. Formally, consider :math:`P = \Sigma^{-1}` the precision matrix and let
    :math:`L` be the Cholesky factor with positive diagonal elements, the log Cholseky
    is given by:

    .. math::
        \tilde{L}_{ij} = L_{ij}, \, i > j,

    and:

    .. math::
        \tilde{L}_{ii} = \log L_{ii}.

    This is very numerically stable and fast, as it doesn't require inverting the
    matrix when computing quadratic forms. The log determinant is then equal to:

    .. math::

        \log \det P = 2 \operatorname{Tr}(\tilde{L}).

    You can use these methods by creating the appropriate `MatRepr` with methods of
    `ball`, `diag` or `full`.

    Additionnally, if your data has mixed missing values, do not use `full` matrix
    parametrization for the residuals, as is this case the components must be
    independent.

    Attributes:
        gamma (torch.Tensor | None): The population level parameters.
        q (CovParam): The random effects precision matrix representation.
        r (CovParam): The residuals precision matrix representation.
        alphas (nn.ParameterDict): The link linear parameters.
        betas (nn.ParameterDict | None): The covariates parameters.
        extra (nn.ParameterDict | None): A dictionary of parameters that is
            passed in addition to other mandatory parameters.

    Examples:
        >>> # To create a model with simple parameters
        >>> init_params = ModelParams(
        ...     torch.zeros_like(gamma),
        ...     CovParam.from_cov(torch.eye(Q.size(0)), method="diag"),
        ...     CovParam.from_cov(torch.eye(R.size(0)), method="ball"),
        ...     {k: torch.zeros_like(v) for k, v in alphas.items()},
        ...     {k: torch.zeros_like(v) for k, v in betas.items()},
        ... )
        >>> # To create a model with shared parameters
        >>> alpha_shared = torch.zeros_like(list(alphas.values())[0])
        >>> init_params_shared = ModelParams(
        ...     torch.zeros_like(gamma),
        ...     CovParam.from_cov(torch.eye(Q.size(0)), method="diag"),
        ...     CovParam.from_cov(torch.eye(R.size(0)), method="ball"),
        ...     {k: alpha_shared for k, v in alphas.items()},
        ...     {k: torch.zeros_like(v) for k, v in betas.items()},
        ... )
    """

    gamma: torch.Tensor | None
    q: CovParam
    r: CovParam
    alphas: nn.ParameterDict
    betas: nn.ParameterDict | None
    extra: nn.ParameterDict | None

    @validate_params(
        {
            "gamma": [torch.Tensor, None],
            "q": [CovParam],
            "r": [CovParam],
            "alphas": [dict],
            "betas": [dict],
            "extra": [dict, None],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        gamma: torch.Tensor | None,
        q: CovParam,
        r: CovParam,
        alphas: dict[tuple[Any, Any], torch.Tensor],
        betas: dict[tuple[Any, Any], torch.Tensor] | None,
        *,
        extra: dict[str, torch.Tensor] | None = None,
    ):
        """Initializes the `ModelParams` object.

        Args:
            gamma (torch.Tensor | None): The population level parameters.
            q (CovParam): The random effects precision matrix representation.
            r (CovParam): The residuals precision matrix representation.
            alphas (dict[tuple[Any, Any], torch.Tensor]): The link linear parameters.
            betas (dict[tuple[Any, Any], torch.Tensor] | None): The covariates
                parameters.
            extra (dict[str, torch.Tensor] | None, optional): A dictionary of parameters
                that is passed in addition to other mandatory parameters. Defaults to
                None.

        Raises:
            ValueError: If any of the tensors contains inf or NaN values.
        """
        super().__init__()

        self.gamma = None if gamma is None else nn.Parameter(gamma)
        self.q = q
        self.r = r
        self.alphas = nn.ParameterDict({str(key): val for key, val in alphas.items()})
        self.betas = (
            None
            if betas is None
            else nn.ParameterDict({str(key): val for key, val in betas.items()})
        )
        self.extra = None if extra is None else nn.ParameterDict(extra)

        for key, val in self.named_parameters():
            assert_all_finite(val.detach(), input_name=key)

    def numel(self) -> int:
        """Return the number of unique parameters.

        Returns:
            int: The number of the (unique) parameters.
        """
        return sum(p.numel() for p in self.parameters())
