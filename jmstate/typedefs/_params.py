from collections.abc import Callable
from dataclasses import field, replace
from functools import cached_property
from itertools import chain
from typing import Any, Self, cast

import torch
from numpy import array2string
from pydantic import ConfigDict, dataclasses, validate_call
from rich.tree import Tree

from ..utils._checks import check_inf, check_matrix_dim, check_nan
from ..utils._linalg import flat_from_log_cholesky, log_cholesky_from_flat
from ..visualization._print import rich_str
from ._defs import Tensor1D


@dataclasses.dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
class CovParams:
    r"""Dataclass containing covariance parameters.

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
        flat (Tensor1D): The flat representation of the covariance matrix.
        dim (int): The dimension of the covariance matrix.
        method (str): The method used to parametrize the covariance matrix.
        skip_validation (bool): Whether to skip validation.
    """

    flat: Tensor1D
    dim: int
    method: str
    skip_validation: bool = field(default=False, repr=False)

    @classmethod
    def from_cov(cls, V: torch.Tensor, method: str = "full") -> Self:
        r"""Gets representation from covariance matrix according to choice of method.

        Note three types of covariance matrices parametrization are provided: scalar
        matrix; diagonal matrix; full matrix. Defaults to the full matrix
        parametrization. This is achieved through a log Cholesky parametrization of the
        inverse covariance matrix. Formally, consider :math:`P = \Sigma^{-1}` the
        precision matrix and let :math:`L` be the Cholesky factor with positive diagonal
        elements, the log Cholseky is given by:

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

        Args:
            V (Tensor2D): The square covariance matrix parameter.
            method (str, optional): The method, full, diag or ball. Defaults to "full".

        Returns:
            Self: The usable representation.
        """
        L = cast(torch.Tensor, torch.linalg.cholesky(V.inverse()))  # type: ignore
        L.diagonal().log_()
        return cls(flat_from_log_cholesky(L, method), L.size(0), method)

    def __str__(self) -> str:
        """Return a string representation of the model parameters.

        Returns:
            str: The string representation.
        """
        tree = Tree("CovParams")
        tree.add(
            f"flat: {array2string(self.flat.numpy(), precision=3, suppress_small=True)}"
        )
        tree.add(f"dim: {self.dim}")
        tree.add(f"method: {self.method}")

        return rich_str(tree)

    def __post_init__(self):
        """Checks if the representation is valid.

        Bypass checks by activating the `skip_validation` flag.

        Raises:
            ValueError: If the representation is invalid.
        """
        if self.skip_validation:
            return

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


@dataclasses.dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
class ModelParams:
    r"""Dataclass containing model parameters.

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

    Bypass checks by activating the `skip_validation` flag.

    Attributes:
        gamma (Tensor1D | None): The population level parameters.
        Q (CovParams): The random effects precision matrix representation.
        R (CovParams): The residuals precision matrix representation.
        alphas (dict[tuple[Any, Any], Tensor1D]): The link linear parameters.
        betas (dict[tuple[Any, Any], Tensor1D] | None): The covariates parameters.
        extra (list[torch.Tensor] | None): A list of parameters that is passed in
            addition to other mandatory parameters.
        skip_validation (bool): A boolean value to skip validation.
    """

    gamma: Tensor1D | None
    Q: CovParams
    R: CovParams
    alphas: dict[tuple[Any, Any], Tensor1D]
    betas: dict[tuple[Any, Any], Tensor1D] | None
    extra: list[torch.Tensor] | None = field(default=None, repr=False)
    skip_validation: bool = field(default=False, repr=False)

    def __str__(self) -> str:
        """Return a string representation of the model parameters.

        Returns:
            str: The string representation.
        """

        def _to_str(t: torch.Tensor) -> str:
            return array2string(t.numpy(), precision=3, suppress_small=True)

        tree = Tree("ModelParams")
        if self.gamma is not None:
            tree.add(f"gamma: {_to_str(self.gamma)}")
        tree.add(f"Q: {self.Q}")
        tree.add(f"R: {self.R}")
        alphas = tree.add("alphas:")
        for k, v in self.alphas.items():
            alphas.add(f"{k[0]} --> {k[1]}: {_to_str(v)}")
        if self.betas is not None:
            betas = tree.add("betas:")
            for k, v in self.betas.items():
                betas.add(f"{k[0]} --> {k[1]}: {_to_str(v)}")

        return rich_str(tree)

    def __post_init__(self):
        """Validate and put to dtype all tensors.

        Raises:
            ValueError: If any of the tensors contains inf values.
            ValueError: If any of the tensors contains NaN values.
        """
        if self.skip_validation:
            return

        def _sort_dict(
            dct: dict[tuple[Any, Any], torch.Tensor],
        ) -> dict[tuple[Any, Any], torch.Tensor]:
            return dict(sorted(dct.items(), key=lambda kv: str(kv[0])))

        # For reproducibility purposes and order
        object.__setattr__(self, "alphas", _sort_dict(self.alphas))
        if self.betas is not None:
            object.__setattr__(self, "betas", _sort_dict(self.betas))

        dtype = torch.get_default_dtype()

        for t in self.as_list:
            t.data = t.to(dtype)
        if self.extra is not None:
            for t in self.extra:
                t.data = t.to(dtype)

        for key, val in self.as_dict.items():
            check_inf(((val, key),))
            check_nan(((val, key),))

    def _map_fn_params(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Self:
        """Map operation and get new parameters.

        Args:
            fn (Callable[[torch.Tensor], torch.Tensor]): The operation.

        Returns:
            Self: The new parameters (it might be a reshape).
        """

        def _map_fn(dict: dict[tuple[Any, Any], torch.Tensor]):
            return {key: fn(val) for key, val in dict.items()}

        return self.__class__(
            None if self.gamma is None else fn(self.gamma),
            replace(self.Q, flat=fn(self.Q.flat), skip_validation=True),
            replace(self.R, flat=fn(self.R.flat), skip_validation=True),
            _map_fn(self.alphas),
            None if self.betas is None else _map_fn(self.betas),
            extra=self.extra,
            skip_validation=True,
        )

    @cached_property
    def as_list(self) -> list[torch.Tensor]:
        """Gets a list of all the unique parameters.

        Returns:
            list[torch.Tensor]: The list of the (unique) parameters.
        """
        seen: set[torch.Tensor] = set()
        candidates = [self.gamma, self.Q.flat, self.R.flat, self.alphas, self.betas]
        _is_new = lambda x: not (x in seen or seen.add(x))  # noqa: E731  # type: ignore

        def _items(
            v: torch.Tensor | dict[tuple[Any, Any] | None, torch.Tensor],
        ) -> list[torch.Tensor]:
            if v is None:
                return []
            if isinstance(v, torch.Tensor):
                return [v] if _is_new(v) else []
            return [t for t in v.values() if _is_new(t)]

        return list(chain.from_iterable(_items(v) for v in candidates))

    @cached_property
    def as_dict(self) -> dict[str, torch.Tensor]:
        """Gets a dict of all the unique parameters names and values.

        Returns:
            dict[str, torch.Tensor]]: The dict of
                the (unique) parameters names and values.
        """
        seen: set[torch.Tensor] = set()
        candidates = {
            "gamma": self.gamma,
            "Q": self.Q.flat,
            "R": self.R.flat,
            "alphas": self.alphas,
            "betas": self.betas,
        }
        _is_new = lambda x: not (x in seen or seen.add(x))  # noqa: E731  # type: ignore

        def _items(
            k: str, v: torch.Tensor | dict[tuple[Any, Any] | None, torch.Tensor]
        ) -> list[tuple[str, torch.Tensor]]:
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                return [(k, v)] if _is_new(v) else []
            return [(f"{k}[{sk}]", sv) for sk, sv in v.items() if _is_new(sv)]

        return dict(chain.from_iterable(_items(k, v) for k, v in candidates.items()))

    @property
    def as_flat_tensor(self) -> Tensor1D:
        """Get the flattened unique parameters.

        Returns:
            torch.Tensor: The flattened (unique) parameters.
        """
        return torch.cat([p.reshape(-1) for p in self.as_list])

    @cached_property
    def numel(self) -> int:
        """Return the number of unique parameters.

        Returns:
            int: The number of the (unique) parameters.
        """
        return sum(p.numel() for p in self.as_list)

    def requires_grad_(self, req: bool):
        """Enable or disable gradient computation on non extra parameters.

        Args:
            req (bool): Wether to require or not.
        """
        for t in self.as_list:
            t.requires_grad_(req)

    def extra_requires_grad_(self, req: bool):
        """Enable or disable gradient computation on extra parameters.

        Args:
            req (bool): Wether to require or not.
        """
        if self.extra is None:
            return
        for t in self.extra:
            t.requires_grad_(req)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def from_flat_tensor(self, flat: Tensor1D) -> Self:
        """Gets a ModelParams object based on the flat representation.

        This uses the current object as the reference.

        Args:
            flat (torch.Tensor): The flat representation.

        Returns:
            Self The constructed ModelParams.
        """
        seen: dict[torch.Tensor, torch.Tensor] = {}
        i = 0

        def _next(ref: torch.Tensor):
            nonlocal seen, i
            if ref in seen:
                return seen[ref]

            n = ref.numel()
            seen[ref] = flat[i : i + n].reshape(ref.shape)
            i += n

            return seen[ref]

        return cast(Self, self._map_fn_params(_next))

    def detach(self) -> Self:
        """Returns a detached reshape of the parameters.

        Returns:
            Self The detached reshape.
        """
        return cast(Self, self._map_fn_params(torch.detach))

    def clone(self) -> Self:
        """Returns a clone of the parameters.

        Returns:
            Self The clone.
        """
        seen: dict[torch.Tensor, torch.Tensor] = {}

        def _next(t: torch.Tensor):
            nonlocal seen
            if t not in seen:
                seen[t] = t.clone()

            return seen[t]

        return cast(Self, self._map_fn_params(_next))
