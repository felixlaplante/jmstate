from typing import Any, Callable

import torch
from tqdm import trange

from ..typedefs._defs import (
    IntPositive,
    NumPositive,
    NumProbability,
    Tensor0D,
    Tensor1D,
    Tensor2D,
    Tensor3D,
)


class MetropolisHastingsSampler:
    """A robust Metropolis-Hastings sampler with adaptive step size."""

    logpdf_aux: Callable[[Tensor3D], tuple[Tensor2D, tuple[torch.Tensor, ...]]]
    init_state: Tensor3D
    n_chains: IntPositive
    adapt_rate: NumPositive
    target_accept_rate: NumProbability
    state: Tensor3D
    logpdf: Tensor2D
    aux: tuple[torch.Tensor, ...]
    step_sizes: Tensor1D
    n_samples: Tensor0D
    n_accepted: Tensor1D

    def __init__(
        self,
        logpdf_aux_fn: Callable[[Tensor3D], tuple[Tensor2D, tuple[torch.Tensor, ...]]],
        init_state: Tensor3D,
        n_chains: IntPositive,
        init_step_size: NumPositive,
        adapt_rate: NumPositive,
        target_accept_rate: NumProbability,
    ):
        """Initialize the Metropolis-Hastings sampler kernel.

        Args:
            logpdf_aux_fn (Callable[[Tensor3D], tuple[Tensor2D, tuple[torch.Tensor, ...]]]): logpdf function.
            init_state (Tensor3D): Starting state for the chain.
            n_chains (IntPositive): The number of parallel chains to spawn.
            init_step_size (NumPositive): Kernel step in Metropolis Hastings.
            adapt_rate (NumPositive): Adaptation rate for the step_size.
            target_accept_rate (NumProbability): Mean acceptance target.
        """
        self.logpdf_aux_fn = logpdf_aux_fn
        self.n_chains = n_chains
        self.adapt_rate = adapt_rate
        self.target_accept_rate = target_accept_rate

        # Initialize state
        self.init_state = init_state.clone()
        self.state = init_state

        # Compute initial log logpdf
        self.logpdf, self.aux = self.logpdf_aux_fn(self.state)

        # Proposal noise initialization
        self._noise = torch.empty_like(self.state)

        # Steps initialization
        self.step_sizes = torch.full((init_state.size(-2),), init_step_size)

        # Statistics tracking
        self.n_samples = torch.tensor(0, dtype=torch.int64)
        self.n_accepted = torch.zeros(init_state.shape[-2])

    @torch.no_grad()  # type: ignore
    def step(self) -> tuple[Tensor2D, tuple[torch.Tensor, ...]]:
        """Performs a single kernel step.

        Returns:
            tuple[Tensor2D, tuple[torch.Tensor, ...]]: Current state and aux.
        """
        # Generate proposal noise
        self._noise.uniform_(-1.0, 1.0)

        # Get the proposal
        proposed_state = self.state + self._noise * self.step_sizes.view(1, -1, 1)
        proposed_logpdf, proposed_aux = self.logpdf_aux_fn(proposed_state)
        logpdf_diff = proposed_logpdf - self.logpdf

        # Vectorized acceptance decision
        log_uniform = torch.log(torch.rand_like(logpdf_diff))
        accept_mask = log_uniform < logpdf_diff

        self.state[accept_mask] = proposed_state[accept_mask]
        self.logpdf[accept_mask] = proposed_logpdf[accept_mask]
        for i, _ in enumerate(self.aux):
            self.aux[i][accept_mask] = proposed_aux[i][accept_mask]

        # Update statistics
        self.n_samples += 1
        self.n_accepted += accept_mask.to(torch.get_default_dtype()).mean(dim=0)

        self._adapt_step_sizes(accept_mask)

        return self.state, self.aux

    def _adapt_step_sizes(self, accept_mask: Tensor1D):
        adaptation = (
            accept_mask.to(torch.get_default_dtype()).mean(dim=0)
            - self.target_accept_rate
        ) * self.adapt_rate
        self.step_sizes *= torch.exp(adaptation)

    def run(self, n_steps: int) -> tuple[Tensor3D, tuple[torch.Tensor, ...]]:
        """Runs the MCMC for n_iterations.

        Args:
            n_steps (int): The number of steps.
        """
        state = self.init_state
        aux = tuple(torch.zeros_like(t, device="meta") for t in self.aux)

        for _ in range(n_steps):
            state, aux = self.step()

        return state, aux

    def loop(
        self,
        max_iterations: int,
        n_steps: int,
        job: Callable[[Tensor3D, tuple[torch.Tensor, ...]], bool],
        desc: str,
        verbose: bool,
    ) -> None:
        """Loops while subsampling.

        Args:
            max_iterations (int): The number of iterations to do.
            n_steps (int): The sublamping MCMC number.
            job (Callable[[], bool | None]): The function to execute.
            desc (str): The description during the loop.
            verbose (bool): Wheter or not to show progress.
        """
        for _ in trange(max_iterations, desc=desc, disable=not verbose):
            state, aux = self.run(n_steps)
            if job(state, aux):
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
