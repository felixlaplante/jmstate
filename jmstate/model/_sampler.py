from collections.abc import Callable
from typing import Any

import torch
from tqdm import trange


class MetropolisHastingsSampler:
    """A robust Metropolis-Hastings sampler with adaptive step size."""

    logpdfs_aux_fn: Callable[
        [torch.Tensor], tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
    ]
    init_state: torch.Tensor
    n_chains: int
    adapt_rate: int | float
    target_accept_rate: int | float
    state: torch.Tensor
    logpdfs: torch.Tensor
    aux: tuple[torch.Tensor, torch.Tensor]
    step_sizes: torch.Tensor
    n_samples: torch.Tensor
    n_accepted: torch.Tensor

    def __init__(
        self,
        logpdfs_aux_fn: Callable[
            [torch.Tensor], tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        ],
        init_state: torch.Tensor,
        n_chains: int,
        init_step_size: int | float,
        adapt_rate: int | float,
        target_accept_rate: int | float,
    ):
        """Initializes the Metropolis-Hastings sampler kernel.

        Args:
            logpdfs_aux_fn (Callable[[torch.Tensor], tuple[torch.Tensor,
                tuple[torch.Tensor, torch.Tensor]]]): logp dfs function.
            init_state (torch.Tensor): Starting state for the chain.
            n_chains (int): The number of parallel chains to spawn.
            init_step_size (int | float): Kernel step in Metropolis Hastings.
            adapt_rate (int | float): Adaptation rate for the step_size.
            target_accept_rate (int | float): Mean acceptance target.
        """
        self.logpdfs_aux_fn = logpdfs_aux_fn
        self.n_chains = n_chains
        self.adapt_rate = adapt_rate
        self.target_accept_rate = target_accept_rate

        # Initialize state
        self.init_state = init_state
        self.state = init_state.clone()

        # Compute initial log logpdfs
        self.logpdfs, self.aux = self.logpdfs_aux_fn(self.state)

        # Proposal noise initialization
        self._noise = torch.empty_like(self.state)

        # Steps initialization
        self.step_sizes = torch.full((1, self.state.size(-2)), init_step_size)

        # Statistics tracking
        self.n_samples = torch.tensor(0, dtype=torch.int64)
        self.n_accepted = torch.zeros(self.state.size(-2))

    def step(self) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Performs a single kernel step.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, ...]]: Current state and aux.
        """
        # Generate proposal noise
        self._noise.normal_()

        # Get the proposal
        proposed_state = self.state + self._noise * self.step_sizes.unsqueeze(-1)
        proposed_logpdfs, proposed_aux = self.logpdfs_aux_fn(proposed_state)
        logpdf_diff = proposed_logpdfs - self.logpdfs

        # Vectorized acceptance decision
        log_uniform = torch.log(torch.rand_like(logpdf_diff))
        accept_mask = log_uniform < logpdf_diff

        torch.where(
            accept_mask.unsqueeze(-1), proposed_state, self.state, out=self.state
        )
        torch.where(accept_mask, proposed_logpdfs, self.logpdfs, out=self.logpdfs)

        (psi, logliks), (proposed_psi, proposed_logliks) = self.aux, proposed_aux
        torch.where(accept_mask.unsqueeze(-1), proposed_psi, psi, out=psi)
        torch.where(accept_mask, proposed_logliks, logliks, out=logliks)

        # Update statistics
        self.n_samples += 1
        mean_accept_mask = accept_mask.to(torch.get_default_dtype()).mean(dim=0)
        self.n_accepted += mean_accept_mask

        # Update step sizes
        self._adapt_step_sizes(mean_accept_mask)

        return self.state, self.aux

    def _adapt_step_sizes(self, mean_accept_mask: torch.Tensor):
        """Adapts the step sizes based on the mean acceptances.

        Args:
            mean_accept_mask (torch.Tensor): The mean acceptances.
        """
        adaptation = (mean_accept_mask - self.target_accept_rate) * self.adapt_rate
        self.step_sizes *= torch.exp(adaptation)

    def run(self, n_steps: int) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
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
        job: Callable[[torch.Tensor, tuple[torch.Tensor, ...]], bool],
        desc: str,
        verbose: bool,
    ) -> None:
        """Loops while sub-sampling.

        Args:
            max_iterations (int): The number of iterations to do.
            n_steps (int): The sublamping MCMC number.
            job (Callable[[], bool | None]): The function to execute.
            desc (str): The description during the loop.
            verbose (bool): Whether or not to show progress.
        """
        for _ in trange(max_iterations, desc=desc, disable=not verbose):
            state, aux = self.run(n_steps)
            if job(state, aux):
                break

    @property
    def acceptance_rates(self) -> torch.Tensor:
        """Gets the acceptance_rate.

        Returns:
            torch.Tensor: The means of the acceptance_rates across iterations.
        """
        return self.n_accepted / torch.clamp(self.n_samples, min=1.0)

    @property
    def mean_acceptance_rate(self) -> float:
        """Gets the acceptance_rate mean across all individuals.

        Returns:
            torch.Tensor: The means across iterations and individuals.
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
