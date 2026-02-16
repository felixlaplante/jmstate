from math import ceil
from numbers import Integral
from typing import Any, cast

import torch
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from sklearn.utils.validation import (  # type: ignore
    assert_all_finite,  # type: ignore
    check_consistent_length,  # type: ignore
    check_is_fitted,  # type: ignore
)
from tqdm import trange

from ..types._data import (
    ModelData,
    ModelDataUnchecked,
    ModelDesign,
    SampleData,
    SampleDataUnchecked,
)
from ..types._defs import Trajectory
from ..types._parameters import ModelParameters
from ..utils._cache import Cache
from ..utils._convert_parameters import parameters_to_vector, vector_to_parameters
from ._hazard import HazardMixin
from ._sampler import MCMCMixin


class PredictMixin(HazardMixin, MCMCMixin):
    """Mixin class for prediction."""

    design: ModelDesign
    params: ModelParameters
    n_chains: int
    n_warmup: int
    n_subsample: int
    verbose: bool | int
    fim_: torch.Tensor | None
    _cache: Cache

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the prediction mixin."""
        super().__init__(*args, **kwargs)

    def _sample_params(self, sample_size: int) -> torch.Tensor:
        """Sample model parameters based on asymptotic behavior of the MLE.

        This uses Bernstein-von Mises theorem to approximate the posterior distribution
        of the parameters as a multivariate normal distribution with mean equal to the
        MLE and covariance matrix equal to the inverse of the Fisher Information Matrix.

        Args:
            sample_size (int): The desired sample size.

        Raises:
            ValueError: If the model is not fitted.

        Returns:
            torch.Tensor: A tensor of sampled model parameters as vectors.
        """
        check_is_fitted(self, "fim_")

        dist = torch.distributions.MultivariateNormal(
            parameters_to_vector(self.params.parameters()).detach(),
            cast(torch.Tensor, self.fim_).inverse(),
        )
        return dist.sample((sample_size,))

    @torch.no_grad()  # type: ignore
    @validate_params(
        {
            "data": [ModelData],
            "u": [torch.Tensor],
            "n_samples": [Interval(Integral, 1, None, closed="left")],
            "double_monte_carlo": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def predict_y(
        self,
        data: ModelData,
        u: torch.Tensor,
        *,
        n_samples: int = 1000,
        double_monte_carlo: bool = False,
    ) -> torch.Tensor:
        r"""Function to predict longitudinal values.

        For every drawing of a random effect :math:`b`, this computes at the prediction
        times :math:`u` the values of the regression function given input data:

        .. math::
            h(u, b).

        The variable `u` is expected to be a matrix with the same number of rows as
        individuals, and the same number of columns as prediction times.

        If `double_monte_carlo` is True, then the prediction is computed using double
        Monte Carlo Rizopoulos-style (2011).

        Args:
            data (ModelData): The data to predict.
            u (torch.Tensor): The matrix containing prediction times.
            n_samples (int, optional): The number of samples to draw from the
                posterior. Defaults to 1000.
            double_monte_carlo (bool, optional): Whether to use double Monte Carlo.
                Defaults to False.

        Raises:
            ValueError: If `double_monte_carlo` is True and the model is not fitted.
            ValueError: If `u` contains inf or NaN values.
            ValueError: If `u` has incompatible shape.

        Returns:
            torch.Tensor: The predicted longitudinal values.
        """
        assert_all_finite(u, input_name="u")
        check_consistent_length(u, data)

        # Load and complete data
        data = ModelDataUnchecked(
            data.x, data.t, data.y, data.trajectories, data.c
        ).prepare(self)

        # Initialize variables
        y_pred: list[torch.Tensor] = []
        n_iter = ceil(n_samples / self.n_chains)

        if double_monte_carlo:
            init_params = parameters_to_vector(self.params.parameters())
            sampled_params = self._sample_params(n_iter)

        # Initialize MCMC
        sampler = self._init_mcmc(data)
        sampler.run(self.n_warmup)

        for i in trange(
            n_iter, desc="Predicting longitudinal values", disable=not self.verbose
        ):
            if double_monte_carlo:
                vector_to_parameters(sampled_params[i], self.params.parameters())  # type: ignore

            y = self.design.regression_fn(u, sampler.psi)
            y_pred.extend(y[i] for i in range(y.size(0)))
            sampler.run(self.n_subsample)

        # Restore parameters
        if double_monte_carlo:
            vector_to_parameters(init_params, self.params.parameters())  # type: ignore
        self._cache.clear()
        return torch.stack(y_pred[:n_samples])

    @torch.no_grad()  # type: ignore
    @validate_params(
        {
            "data": [ModelData],
            "u": [torch.Tensor],
            "n_samples": [Interval(Integral, 1, None, closed="left")],
            "double_monte_carlo": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def predict_surv_logps(
        self,
        data: ModelData,
        u: torch.Tensor,
        *,
        n_samples: int = 1000,
        double_monte_carlo: bool = False,
    ) -> torch.Tensor:
        r"""Function to predict survival log probability values.

        For every drawing of a random effect :math:`b`, this computes at the prediction
        times :math:`u` the values of the log survival probabilities given input data
        and conditionally to survival up to time :math:`c`:

        .. math::
            \log \mathbb{P}(T^* \geq u \mid T^* > c) = -\int_c^u \lambda(t) \, dt.

        When multiple transitions are allowed, :math:`\lambda(t)` is a sum over all
            possible transitions, that is to say if an individual is in the state
            :math:`k` from time :math:`t_0`, this gives:

            .. math::
                -\int_c^u \sum_{k'} \lambda^{k' \mid k}(t \mid t_0) \, dt.

        Please note this makes use of the Chasles property in order to avoid the
        computation of two integrals and make computations more precise.

        The variable `u` is expected to be a matrix with the same number of rows as
        individuals, and the same number of columns as prediction times.

        If `double_monte_carlo` is True, then the prediction is computed using double
        Monte Carlo Rizopoulos-style (2011).

        Args:
            data (ModelData): The data to predict.
            u (torch.Tensor): The matrix containing prediction times.
            n_samples (int, optional): The number of samples to draw from the
                posterior. Defaults to 1000.
            double_monte_carlo (bool, optional): Whether to use double Monte Carlo.
                Defaults to False.

        Raises:
            ValueError: If `double_monte_carlo` is True and the model is not fitted.
            ValueError: If `u` contains inf or NaN values.
            ValueError: If `u` has incompatible shape.

        Returns:
            torch.Tensor: The predicted survival log probabilities.
        """
        assert_all_finite(u, input_name="u")
        check_consistent_length(u, data)

        # Load and complete data
        data = ModelDataUnchecked(
            data.x, data.t, data.y, data.trajectories, data.c
        ).prepare(self)

        # Initialize variables
        surv_logps_pred: list[torch.Tensor] = []
        n_iter = ceil(n_samples / self.n_chains)

        if double_monte_carlo:
            init_params = parameters_to_vector(self.params.parameters())
            sampled_params = self._sample_params(n_iter)

        # Initialize MCMC
        sampler = self._init_mcmc(data)
        sampler.run(self.n_warmup)

        for i in trange(
            n_iter,
            desc="Predicting survival log probabilities",
            disable=not self.verbose,
        ):
            if double_monte_carlo:
                vector_to_parameters(sampled_params[i], self.params.parameters())  # type: ignore

            sample_data = SampleData(data.x, data.trajectories, sampler.psi, data.c)
            surv_logps = self.compute_surv_logps(sample_data, u)
            surv_logps_pred.extend(surv_logps[i] for i in range(surv_logps.size(0)))
            sampler.run(self.n_subsample)

        if double_monte_carlo:
            vector_to_parameters(init_params, self.params.parameters())  # type: ignore
        self._cache.clear()
        return torch.stack(surv_logps_pred[:n_samples])

    @torch.no_grad()  # type: ignore
    @validate_params(
        {
            "data": [ModelData],
            "c": [torch.Tensor],
            "max_length": [Interval(Integral, 1, None, closed="left")],
            "n_samples": [Interval(Integral, 1, None, closed="left")],
            "double_monte_carlo": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def predict_trajectories(
        self,
        data: ModelData,
        c: torch.Tensor,
        *,
        max_length: int = 10,
        n_samples: int = 1000,
        double_monte_carlo: bool = False,
    ) -> list[list[Trajectory]]:
        r"""Function to predict trajectories.

        For every drawing of a random effect :math:`b`, this simulates the trajectories
        up to a censoring time `c` with a maximum length of `max_length` to avoid
        infinite loops. This uses a variant of Gillepsie's algorithm.

        The variable `c` is expected to be a column vector with the same number of
        rows as individuals.

        If `double_monte_carlo` is True, then the prediction is computed using double
        Monte Carlo Rizopoulos-style (2011).

        Args:
            data (ModelData): The data to predict.
            c (torch.Tensor): The sampling censoring times.
            max_length (int, optional): The maximum length of the trajectories.
                Defaults to 10.
            n_samples (int, optional): The number of samples to draw from the
                posterior. Defaults to 1000.
            double_monte_carlo (bool, optional): Whether to use double Monte Carlo.
                Defaults to False.

        Raises:
            ValueError: If `double_monte_carlo` is True and the model is not fitted.
            ValueError: If `c` contains inf or NaN values.
            ValueError: If `c` has incompatible shape.

        Returns:
            list[list[Trajectory]]: The predicted trajectories.
        """
        assert_all_finite(c, input_name="c")
        check_consistent_length(c, data)

        # Load and complete data
        data = ModelDataUnchecked(
            data.x, data.t, data.y, data.trajectories, data.c
        ).prepare(self)

        # Initialize variables
        trajectories_pred: list[list[Trajectory]] = []
        n_iter = ceil(n_samples / self.n_chains)

        if double_monte_carlo:
            init_params = parameters_to_vector(self.params.parameters())
            sampled_params = self._sample_params(n_iter)

        # Initialize MCMC
        sampler = self._init_mcmc(data)
        sampler.run(self.n_warmup)

        for i in trange(
            n_iter,
            desc="Predicting trajectories",
            disable=not self.verbose,
        ):
            if double_monte_carlo:
                vector_to_parameters(sampled_params[i], self.params.parameters())  # type: ignore

            # Sample trajectories, not possible to vectorize fully
            for j in range(sampler.psi.size(0)):
                sample_data = SampleDataUnchecked(
                    data.x, data.trajectories, sampler.psi[j], data.c
                )
                trajectories_pred.append(
                    self.sample_trajectories(sample_data, c, max_length=max_length)
                )
            sampler.run(self.n_subsample)  # type: ignore

        if double_monte_carlo:
            vector_to_parameters(init_params, self.params.parameters())  # type: ignore
        self._cache.clear()
        return trajectories_pred[:n_samples]
