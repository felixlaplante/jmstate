from typing import Any, Callable, SupportsFloat

import torch
from tqdm import trange

from ..typedefs._defs import Tensor0D, Tensor1D, Tensor2D, Tensor3D


class MetropolisHastingsSampler:
    """A robust Metropolis-Hastings sampler with adaptive step size."""

    logpdf_aux: Callable[[Tensor3D], tuple[Tensor2D, tuple[torch.Tensor, ...]]]
    init_state: Tensor3D
    n_chains: int
    adapt_rate: float
    target_accept_rate: float
    current_state: Tensor3D
    current_logpdf: Tensor2D
    current_aux: tuple[torch.Tensor, ...]
    step_sizes: Tensor1D
    n_samples: Tensor0D
    n_accepted: Tensor1D

    def __init__(
        self,
        logpdf_aux_fn: Callable[[Tensor3D], tuple[Tensor2D, tuple[torch.Tensor, ...]]],
        init_state: Tensor3D,
        n_chains: int,
        init_step_size: SupportsFloat,
        adapt_rate: SupportsFloat,
        target_accept_rate: SupportsFloat,
    ):
        """Initialize the Metropolis-Hastings sampler kernel.

        Args:
            logpdf_aux_fn (Callable[[Tensor3D], tuple[Tensor2D, tuple[torch.Tensor, ...]]]): logpdf function.
            init_state (Tensor3D): Starting state for the chain.
            n_chains (int): The number of parallel chains to spawn.
            init_step_size (SupportsFloat): Kernel step in Metropolis Hastings.
            adapt_rate (SupportsFloat): Adaptation rate for the step_size.
            target_accept_rate (SupportsFloat): Mean acceptance target.

        Raises:
            RuntimeError: If the initial log prob fails to be computed.
        """
        self.logpdf_aux_fn = logpdf_aux_fn
        self.n_chains = n_chains
        self.adapt_rate = float(adapt_rate)
        self.target_accept_rate = float(target_accept_rate)

        # Initialize state
        self.current_state = init_state

        # Compute initial log logpdf
        self.current_logpdf, self.current_aux = self.logpdf_aux_fn(self.current_state)

        # Steps initialization
        self.step_sizes = torch.full(
            (init_state.size(-2),), float(init_step_size), dtype=torch.float32
        )

        # Statistics tracking
        self.n_samples = torch.tensor(0, dtype=torch.int64)
        self.n_accepted = torch.zeros((init_state.shape[-2],), dtype=torch.float32)

        self._check()

    def _check(self):
        """Check if every input is valid.

        Raises:
            TypeError: If n_chains is not strictly positive.
            ValueError: If init_step_size is not strictly positive.
            ValueError: If adapt_rate is not positive.
            ValueError: If target_accept_rate is not in (0, 1).
        """
        if self.n_chains <= 0:
            raise ValueError(f"n_chains must be strictly positive, got {self.n_chains}")
        if self.step_sizes[0] <= 0:
            raise ValueError(
                f"init_step_size must be strictly positive, got {self.step_sizes[0]}"
            )
        if self.adapt_rate < 0:
            raise ValueError(f"adapt_rate must be positive, got {self.adapt_rate}")
        if not 0 < self.target_accept_rate < 1:
            raise ValueError(
                f"target_accept_rate must be in (0, 1), got {self.target_accept_rate}"
            )

    @torch.no_grad()  # type: ignore
    def step(self) -> tuple[Tensor2D, tuple[torch.Tensor, ...]]:
        """Performs a single kernel step.

        Returns:
            tuple[Tensor2D, tuple[torch.Tensor, ...]]: Current state and aux.
        """
        # Generate proposal noise
        noise = torch.randn_like(self.current_state, dtype=torch.float32)

        # Get the proposal
        proposed_state = self.current_state + noise * self.step_sizes.view(1, -1, 1)
        proposed_logpdf, proposed_aux = self.logpdf_aux_fn(proposed_state)
        logpdf_diff = proposed_logpdf - self.current_logpdf

        # Vectorized acceptance decision
        log_uniform = torch.log(torch.rand_like(logpdf_diff))
        accept_mask = log_uniform < logpdf_diff

        self.current_state[accept_mask] = proposed_state[accept_mask]
        self.current_logpdf[accept_mask] = proposed_logpdf[accept_mask]
        for i, _ in enumerate(self.current_aux):
            self.current_aux[i][accept_mask] = proposed_aux[i][accept_mask]

        # Update statistics
        self.n_samples += 1
        self.n_accepted += accept_mask.to(torch.float32).mean(dim=0)

        self._adapt_step_sizes(accept_mask)

        return self.current_state, self.current_aux

    def _adapt_step_sizes(self, accept_mask: Tensor1D):
        adaptation = (
            accept_mask.to(torch.float32).mean(dim=0) - self.target_accept_rate
        ) * self.adapt_rate
        self.step_sizes *= torch.exp(adaptation)

    def warmup(self, warmup: int):
        """Warmups the MCMC.

        Args:
            warmup (int): The number of warmup steps.

        Raises:
            ValueError: If the warmup steps is not positive.
        """
        if warmup < 0:
            raise ValueError("Warmup must be a non-negative integer")

        for _ in range(warmup):
            self.step()

    def loop(
        self,
        n_iterations: int,
        cont_warmup: int,
        job: Callable[[], bool | None],
        desc: str,
        verbose: bool,
    ) -> None:
        """Loops while subsampling.

        Args:
            n_iterations (int): The number of iterations to do.
            cont_warmup (int): The sublamping MCMC number.
            job (Callable[[], bool | None]): The function to execute.
            desc (str): The description during the loop.
            verbose (bool): Wheter or not to show progress.

        Raises:
            ValueError: If n_iter is not greater or equal to one.
            RuntimeError: If an iteration fails.
        """
        for _ in trange(n_iterations, desc=desc, disable=not verbose):
            self.warmup(cont_warmup)
            if job():
                break

    @property
    def acceptance_rates(self) -> Tensor1D:
        """Gets the acceptance_rate.

        Returns:
            torch.Tensor: The means of the acceptance_rates accross iterations.
        """
        return self.n_accepted / torch.clamp(self.n_samples, min=1.0)

    @property
    def mean_acceptance_rate(self) -> float:
        """Gets the acceptance_rate mean across all individuals.

        Returns:
            torch.Tensor: The means accross iterations and individuals.
        """
        return self.acceptance_rates.mean().item()

    @property
    def mean_step_size(self) -> float:
        """Gets the mean step size.

        Returns:
            float: The mean step size.
        """
        return self.step_sizes.mean().item()

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Gets the summary of the MCMC diagnostics.

        Returns:
            dict[str, Any]: The dict of the diagnostics.
        """
        return {
            "acceptance_rates": self.acceptance_rates.clone(),
            "mean_acceptance_rate": self.mean_acceptance_rate,
            "step_sizes": self.step_sizes.clone(),
            "mean_step_size": self.mean_step_size,
        }
