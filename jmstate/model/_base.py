import copy
import warnings
from typing import Any, Callable, cast

import torch

from ..utils._defs import LOGTWOPI, TWO, Trajectory
from ..utils._structures import ModelData, ModelDesign, ModelParams, SampleData
from ._hazard import HazardMixin
from ._longitudinal import LongitudinalMixin
from ._sampler import MetropolisHastingsSampler


class MultiStateJointModel(LongitudinalMixin, HazardMixin):
    """A class of the nonlinear multistate joint model.

    It feature possibility
    to simulate data, fit based on stochastic gradient with any torch.optim
    optimizer of choice.

    Args:
        LongitudinalMixin (_type_): The longitudinal part of the model.
        HazardMixin (_type_): The hazard part of the model.
    """

    model_design: ModelDesign
    params_: ModelParams
    optimizer: type[torch.optim.Optimizer]
    optimizer_params: dict[str, float] | None
    n_quad: int
    n_bissect: int
    enable_likelihood_cache: bool
    enable_predict_cache: bool
    cache_limit: int | None
    fit_: bool

    def __init__(
        self,
        model_design: ModelDesign,
        init_params: ModelParams,
        *,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_params: dict[str, float] | None = None,
        pen: Callable[[ModelParams], torch.Tensor] | None = None,
        n_quad: int = 32,
        n_bissect: int = 32,
        enable_likelihood_cache: bool = True,
        enable_predict_cache: bool = True,
        cache_limit: int | None = 256,
    ):
        """Initializes the joint model based on the user defined design.

        Args:
            model_design (ModelDesign): Model design containing modeling information.
            init_params (ModelParams): Initial values for the parameters.
            optimizer (type[torch.optim.Optimizer], optional): The optimizer constructor. Defaults to torch.optim.Adam.
            optimizer_params (dict[str, Any] | None, optional): Optimizer parameter dict. Defaults to None.
            pen (Callable[[ModelParams], torch.Tensor] | None, optional): The penalization function. Defaults to None.
            n_quad (int, optional): The used numnber of points for Gauss-Legendre quadrature. Defaults to 32.
            n_bissect (int, optional): The number of bissection steps used in transition sampling. Defaults to 32.
            enable_likelihood_cache (bool, optional): Enables cache during fit, and MCMC loops and likelihood computations in general. Defaults to True.
            enable_predict_cache (bool, optional): Enables cache during predicting steps. Defaults to True.
            cache_limit (int | None, optional): The max length of cache. Defaults to 256.

        Raises:
            TypeError: If pen is not None and is not callable.
        """
        # Store model components
        self.model_design = model_design
        self.params_ = copy.deepcopy(init_params)

        # Set up optimizer
        self.optimizer = optimizer
        self.optimizer_params = (
            optimizer_params if optimizer_params is not None else {"lr": 0.01}
        )

        # Store penalization
        if pen is not None and not callable(pen):
            raise TypeError("pen must be callable or None")
        self.pen: Callable[[ModelParams], torch.Tensor] = lambda params: (
            torch.tensor(0.0, dtype=torch.float32) if pen is None else pen(params)
        )

        # Info of the Mixin Classes
        super().__init__(
            n_quad=n_quad,
            n_bissect=n_bissect,
            enable_likelihood_cache=enable_likelihood_cache,
            enable_predict_cache=enable_predict_cache,
            cache_limit=cache_limit,
        )

        # Initialize attributes that will be set later
        self.data: ModelData | None = None
        self.metrics_: dict[str, Any] = {}
        self.fit_ = False

    def _logliks(
        self, b: torch.Tensor, data: ModelData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the total log likelihood up to a constant.

        Args:
            b (torch.Tensor): The individual random effects.
            data (ModelData): Dataset on which the likeihood is computed.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The computed total log logpdf/likelihood.
        """

        def _prior_logliks(b: torch.Tensor) -> torch.Tensor:
            Q_inv_cholesky, Q_log_eigvals = self.params_.get_cholesky_and_log_eigvals(
                "Q"
            )

            Q_quad_form = (b @ Q_inv_cholesky).pow(2).sum(dim=1)
            Q_log_det = (Q_log_eigvals - LOGTWOPI).sum()

            return 0.5 * (Q_log_det - Q_quad_form)

        # Transform random effects to individual-specific parameters
        psi = self.model_design.f(self.params_.gamma, b)

        # Compute individual likelihood components
        long_logliks = super()._long_logliks(psi, data)
        hazard_logliks = super()._hazard_logliks(psi, data)
        prior_logliks = _prior_logliks(b)

        # Sum all likelihood components
        logliks = long_logliks + hazard_logliks
        logpdfs = logliks + prior_logliks

        return logpdfs, logliks

    def _setup_mcmc(
        self,
        data: ModelData,
        init_step_size: float,
        adapt_rate: float,
        target_accept_rate: float,
    ) -> MetropolisHastingsSampler:
        """Setup the MCMC kernel and hyperparameters.

        Args:
            data (ModelData): The dataset on which the likelihood is to be computed.
            init_step_size (float, optional): Kernel standard error in Metropolis Hastings.
            adapt_rate (float, optional): Adaptation rate for the step_size.
            target_accept_rate (float, optional): Mean acceptance target.

        Returns:
            MetropolisHastingsSampler: The intialized Markov kernel.
        """
        # Initialize random effects
        init_b = torch.zeros((data.size, self.params_.Q_dim_), dtype=torch.float32)

        return MetropolisHastingsSampler(
            lambda b: self._logliks(b, data),
            init_b,
            init_step_size,
            adapt_rate,
            target_accept_rate,
        )

    def fit(
        self,
        data: ModelData,
        *,
        n_iter: int = 2000,
        batch_size: int = 1,
        callbacks: list[Callable[[dict[str, Any]], None]] | None,
        init_step_size: float = 0.1,
        adapt_rate: float = 0.01,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
        verbose: bool = True,
    ) -> None:
        """Fits the MultiStateJointModel.

        Args:
            data (ModelData): The dataset to learn from.
            n_iter (int, optional): Number of iterations for optimization. Defaults to 2000.
            batch_size (int, optional): Batch size used in fitting. Defaults to 1.
            callbacks (list[Callable[[dict[str, Any], dict[str, Any]], None]] | None, optional): A list of callbacks. Defaults to None.
            init_step_size (float, optional): Kernel standard error in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the step_size. Defaults to 0.01.
            accept_target (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
            verbose (bool, optional): Wheter or not to show progress. Defaults to True.

        Raises:
            ValueError: If batch_size is not greater than 0.
            TypeError: If some callback is not callable.
        """
        # Verify batch_size
        if batch_size < 1:
            raise ValueError(f"batch_size must be greater than 0, got {batch_size}")

        # Load and complete data
        self.data = data
        x_rep = data.x.repeat(batch_size, 1) if data.x is not None else None
        t_rep = data.t if data.t.ndim == 1 else data.t.repeat(batch_size, 1)
        y_rep = data.y.repeat(batch_size, 1, 1)
        trajectories_rep = data.trajectories * batch_size
        c_rep = data.c.repeat(batch_size)

        data_rep = ModelData(x_rep, t_rep, y_rep, trajectories_rep, c_rep)

        # Prepare data
        data_rep.prepare(self.model_design)

        # Set up optimizer
        self.params_.require_grad(True)
        optimizer = self.optimizer(params=self.params_.as_list, **self.optimizer_params)  # type: ignore

        # Set up MCMC
        sampler = self._setup_mcmc(data_rep, init_step_size, adapt_rate, accept_target)
        sampler.warmup(init_warmup)

        def _fit(iter: int):
            (b, logpdfs), logliks = sampler.step()

            # Optimization step: Update parameters
            penalty = self.pen(self.params_)
            loglik = logliks.sum() / batch_size
            nloglik_pen = -loglik + penalty

            loss = -logpdfs.sum() / batch_size + penalty
            optimizer.zero_grad() # type: ignore
            loss.backward()  # type: ignore
            optimizer.step() # type: ignore

            info = {
                "data": data,
                "iter": iter,
                "n_iter": n_iter,
                "start": iter == 0,
                "end": (iter + 1) == n_iter,
                "model": self,
                "params": self.params_,
                "sampler": sampler,
                "b": b,
                "logpdfs": logpdfs,
                "loglik": loglik,
                "nloglik_pen": nloglik_pen,
            }

            if callbacks is None:
                return

            for callback in callbacks:
                if not callable(callable):
                    raise TypeError("callback is not callable")
                callback(info)

        # Main fitting loop
        sampler.loop(
            n_iter, cont_warmup, _fit, desc="Fitting joint model", verbose=verbose
        )

        params_flat = torch.cat([p.detach().flatten() for p in self.params_.as_list])
        if torch.isnan(params_flat).any() or torch.isinf(params_flat).any():
            warnings.warn("Error infering model parameters", stacklevel=2)

        # Set fit_ to True
        self.params_.require_grad(False)
        self.fit_ = True
        data_rep.clear_extra()
        self.clear_cache()

    def post_fit(
        self,
        new_data: ModelData | None = None,
        *,
        n_iter: int = 1000,
        callbacks: list[Callable[[dict[str, Any], dict[str, Any]], None]] | None = None,
        init_step_size: float = 0.1,
        adapt_rate: float = 0.01,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Fits the MultiStateJointModel.

        Args:
            new_data (ModelData): The dataset to learn from. Defaults to None.
            n_iter (int, optional): Number of iterations for optimization. Defaults to 1000.
            callbacks (list[Callable[[dict[str, Any], dict[str, Any]], None]] | None, optional): A list of callbacks. Defaults to None.
            init_step_size (float, optional): Kernel standard error in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the step_size. Defaults to 0.01.
            accept_target (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
            verbose (bool, optional): Wheter or not to show progress. Defaults to True.

        Raises:
            ValueError: If new_data is be None when model is not fit
            TypeError: If some callback is not callable.
        """
        # Prepare data
        if new_data is None and not self.fit_:
            raise ValueError("new_data must not be None when model is not fit")

        data = cast(ModelData, self.data if new_data is None else new_data)
        data.prepare(self.model_design)

        # Set up gradient if needed
        self.params_.require_grad(True)

        # Set up MCMC
        sampler = self._setup_mcmc(data, init_step_size, adapt_rate, accept_target)
        sampler.warmup(init_warmup)

        # Initialize metrics
        metrics: dict[str, Any] = {}

        def _post_fit(iter: int):
            (b, logpdfs), logliks = sampler.step()

            # Compute stats
            penalty = self.pen(self.params_)
            loglik = logliks.sum()
            nloglik_pen = -loglik + penalty
            loss = -logpdfs.sum() + penalty

            for p in self.params_.as_list:
                if p.grad is not None:
                    p.grad.zero_()
            loss.backward()  # type: ignore

            info = {
                "data": data,
                "iter": iter,
                "n_iter": n_iter,
                "start": iter == 0,
                "end": (iter + 1) == n_iter,
                "model": self,
                "params": self.params_,
                "sampler": sampler,
                "b": b,
                "logpdfs": logpdfs,
                "loglik": loglik,
                "nloglik_pen": nloglik_pen,
            }

            if callbacks is None:
                return

            for callback in callbacks:
                if not callable(callable):
                    raise TypeError("callback is not callable")
                callback(info, metrics)

        # Main post fitting loop
        sampler.loop(
            n_iter, cont_warmup, _post_fit, desc="Running post fit", verbose=verbose
        )

        # Set up metrics
        if new_data is None:
            self.metrics_ = metrics

        # Restore default
        self.params_.require_grad(False)
        data.clear_extra()
        self.clear_cache()

        return metrics

    def get_stderror(self) -> ModelParams:
        """Returns the standard error of the parameters.

        They can be used to draw confidence intervals.

        Raises:
            ValueError: If the Fisher Information Matrix could not be computed.

        Returns:
            ModelParams: The standard error in the same format as the parameters.
        """
        # Check if fim is defined
        if self.metrics_.get("fim") is None:
            raise ValueError(
                "Fisher Information Matrix must be previously computed. CIs may not be computed."
            )

        # Get parameter vector
        params_list = self.params_.as_list
        params_flat = torch.cat([p.detach().flatten() for p in params_list])

        # Compute standard errors
        try:
            fim_inv = torch.linalg.pinv(self.metrics_["fim"])  # type: ignore
            flat_se = torch.sqrt(fim_inv.diag())  # type: ignore

        except Exception as e:
            warnings.warn(
                f"Error inverting Fisher Information Matrix: {e}", stacklevel=2
            )
            flat_se = torch.full_like(params_flat, torch.nan)

        # Organize by parameter structure
        i = 0

        def _next(ref: torch.Tensor) -> torch.Tensor:
            nonlocal i
            n = ref.numel()
            result = flat_se[i : i + n]
            i += n
            return result

        gamma = _next(self.params_.gamma) if self.params_.gamma is not None else None

        Q_flat = _next(self.params_.Q_repr[0])
        Q_method = self.params_.Q_repr[1]

        R_flat = _next(self.params_.R_repr[0])
        R_method = self.params_.R_repr[1]

        alphas = {key: _next(val) for key, val in self.params_.alphas.items()}

        betas = (
            {key: _next(val) for key, val in self.params_.betas.items()}
            if self.params_.betas is not None
            else None
        )
        return ModelParams(gamma, (Q_flat, Q_method), (R_flat, R_method), alphas, betas)

    def predict_y(
        self,
        pred_data: ModelData,
        t: torch.Tensor,
        *,
        n_iter_b: int,
        step_size: float = 0.1,
        adapt_rate: float = 0.01,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
        verbose: bool = True,
    ) -> list[torch.Tensor]:
        """Predicts the longitudinal values (y) for new individuals.

        Args:
            pred_data (ModelData): Prediction data.
            u (torch.Tensor): The evaluation times of the probabilities.
            n_iter_b (int): Number of iterations for random effects sampling.
            step_size (float, optional): Kernel step. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate. Defaults to 0.01.
            accept_target (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
            max_length (int, optional): Maximum iterations or sampling. Defaults to 100.
            verbose (bool, optional): Wheter or not to show progress. Defaults to True.

        Raises:
            ValueError: If u is of incorrect shape.

        Returns:
            list[torch.Tensor]: A list for each b of survival probabilities.
        """
        # Convert and check if c_max matches the right shape
        t = torch.as_tensor(t, dtype=torch.float32)
        if t.ndim != TWO or t.shape[0] != pred_data.size:
            raise ValueError(
                "t has shape {u.shape}, expected {(sample_data.size, eval_points)}"
            )

        # Load and complete prediction data
        pred_data.prepare(self.model_design)

        # Set up MCMC for prediction
        sampler = self._setup_mcmc(pred_data, step_size, adapt_rate, accept_target)

        # Warmup MCMC
        sampler.warmup(init_warmup)

        # Generate predicted probabilites
        predicted_y: list[torch.Tensor] = []

        def _predict_longitudinal(_iter: int):
            (b, _), _ = sampler.step()

            # Transform to individual-specific parameters
            psi = self.model_design.f(self.params_.gamma, b)

            y = self.model_design.h(t, pred_data.x, psi)

            predicted_y.append(y)

        sampler.loop(
            n_iter_b,
            cont_warmup,
            _predict_longitudinal,
            desc="Predicting longitudinal expected values",
            verbose=verbose,
        )

        pred_data.clear_extra()
        self.clear_cache()
        return predicted_y

    def predict_surv_log_probs(
        self,
        pred_data: ModelData,
        u: torch.Tensor,
        *,
        n_iter_b: int,
        step_size: float = 0.1,
        adapt_rate: float = 0.01,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
        verbose: bool = True,
    ) -> list[torch.Tensor]:
        """Predicts the survival (event free) probabilities for new individuals.

        Args:
            pred_data (ModelData): Prediction data.
            u (torch.Tensor): The evaluation times of the probabilities.
            n_iter_b (int): Number of iterations for random effects sampling.
            step_size (float, optional): Kernel step in Metropolis. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate. Defaults to 0.01.
            accept_target (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
            max_length (int, optional): Maximum iterations or sampling. Defaults to 100.
            verbose (bool, optional): Wheter or not to show progress. Defaults to True.

        Raises:
            ValueError: If u is of incorrect shape.

        Returns:
            list[torch.Tensor]: A list for each b of survival probabilities.
        """
        # Convert and check if c_max matches the right shape
        u = torch.as_tensor(u, dtype=torch.float32)
        if u.ndim != TWO or u.shape[0] != pred_data.size:
            raise ValueError(
                "u has shape {u.shape}, expected {(sample_data.size, eval_points)}"
            )

        # Load and complete prediction data
        pred_data.prepare(self.model_design)

        # Set up MCMC for prediction
        sampler = self._setup_mcmc(pred_data, step_size, adapt_rate, accept_target)

        # Warmup MCMC
        sampler.warmup(init_warmup)

        # Generate predicted probabilites
        predicted_log_probs: list[torch.Tensor] = []

        def _predict_surv_log_probs(_iter: int):
            (b, _), _ = sampler.step()

            # Transform to individual-specific parameters
            psi = self.model_design.f(self.params_.gamma, b)

            sample_data = SampleData(
                pred_data.x, pred_data.trajectories, psi, pred_data.c
            )

            log_probs = self.compute_surv_log_probs(sample_data, u)

            predicted_log_probs.append(log_probs)

        sampler.loop(
            n_iter_b,
            cont_warmup,
            _predict_surv_log_probs,
            desc="Predicting survival log probabilities",
            verbose=verbose,
        )

        pred_data.clear_extra()
        self.clear_cache()
        return predicted_log_probs

    def predict_trajectories(
        self,
        pred_data: ModelData,
        c_max: torch.Tensor,
        *,
        n_iter_b: int,
        n_iter_T: int,
        step_size: float = 0.1,
        adapt_rate: float = 0.01,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
        max_length: int = 100,
        verbose: bool = True,
    ) -> list[list[list[Trajectory]]]:
        """Predict survival trajectories for new individuals.

        Args:
            pred_data (ModelData): Prediction data.
            c_max (torch.Tensor): Maximum prediction times.
            n_iter_b (int): Number of iterations for random effects sampling.
            n_iter_T (int): Number of trajectory samples per random effects sample.
            step_size (float, optional): Kernel step in Metropolis. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate. Defaults to 0.01.
            accept_target (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
            max_length (int, optional): Maximum iterations or sampling. Defaults to 100.
            verbose (bool, optional): Wheter or not to show progress. Defaults to True.

        Raises:
            ValueError: If n_iter_T is not greater than 0.
            RuntimeError: If the prediction fails.

        Returns:
            list[list[list[Trajectory]]]: A list of lists of trajectories. First list is for a b sample, then multiples iid drawings of the trajectories.
        """
        # Convert and check if c_max matches the right shape
        c_max = torch.as_tensor(c_max, dtype=torch.float32)
        if c_max.shape != (pred_data.size,):
            raise ValueError(
                "c_max has incorrect shape, got {c_max.shape}, expected {(sample_data.size,)}"
            )

        # Load and complete prediction data

        # Set up MCMC for prediction
        sampler = self._setup_mcmc(pred_data, step_size, adapt_rate, accept_target)

        # Warmup MCMC
        sampler.warmup(init_warmup)

        # Check n_iter_T
        if n_iter_T < 1:
            raise ValueError(f"n_iter_T must be greater than 0, got {n_iter_T}")

        # Prepare replicate data for trajectory sampling
        x_rep = pred_data.x.repeat(n_iter_T, 1) if pred_data.x is not None else None
        trajectories_rep = pred_data.trajectories * n_iter_T
        c_rep = pred_data.c.repeat(n_iter_T)
        c_max_rep = c_max.repeat(n_iter_T)

        # Generate predictions
        predicted_trajectories: list[list[list[Trajectory]]] = []

        def _predict_trajectories(_iter: int):
            (b, _), _ = sampler.step()

            # Transform to individual-specific parameters
            psi = self.model_design.f(self.params_.gamma, b)
            # Replicate for multiple trajectory samples
            psi_rep = psi.repeat(n_iter_T, 1)

            sample_data = SampleData(x_rep, trajectories_rep, psi_rep, c_rep)
            # Sample trajectories
            trajectories = self.sample_trajectories(sample_data, c_max_rep, max_length)

            # Organize by trajectory iteration
            trajectory_chunks = [
                trajectories[i * pred_data.size : (i + 1) * pred_data.size]
                for i in range(n_iter_T)
            ]

            predicted_trajectories.append(trajectory_chunks)

        sampler.loop(
            n_iter_b,
            cont_warmup,
            _predict_trajectories,
            desc="Predicting trajectories",
            verbose=verbose,
        )

        pred_data.clear_extra()
        self.clear_cache()
        return predicted_trajectories
