from collections.abc import Callable
from typing import Any, cast

import torch

from ..typedefs._defs import AuxData
from ..utils._dtype import get_dtype


class MetropolisHastingsSampler:
    """A robust Metropolis-Hastings sampler with adaptive step size."""

    logpdfs_aux_fn: Callable[[torch.Tensor], tuple[torch.Tensor, AuxData]]
    n_chains: int
    adapt_rate: int | float
    target_accept_rate: int | float
    b: torch.Tensor
    logpdfs: torch.Tensor
    aux: AuxData
    step_sizes: torch.Tensor
    n_samples: torch.Tensor
    n_accepted: torch.Tensor

    def __init__(
        self,
        logpdfs_aux_fn: Callable[[torch.Tensor], tuple[torch.Tensor, AuxData]],
        init_b: torch.Tensor,
        n_chains: int,
        init_step_size: int | float,
        adapt_rate: int | float,
        target_accept_rate: int | float,
    ):
        """Initializes the Metropolis-Hastings sampler kernel.

        Args:
            logpdfs_aux_fn (Callable[[torch.Tensor], tuple[torch.Tensor, AuxData]]):
                The log pdfs function with auxiliary data.
            init_b (torch.Tensor): Starting b for the chain.
            n_chains (int): The number of parallel chains to spawn.
            init_step_size (int | float): Kernel step in Metropolis Hastings.
            adapt_rate (int | float): Adaptation rate for the step_size.
            target_accept_rate (int | float): Mean acceptance target.
        """
        dtype = get_dtype()

        self.logpdfs_aux_fn = cast(
            Callable[[torch.Tensor], tuple[torch.Tensor, AuxData]],
            torch.no_grad()(logpdfs_aux_fn),
        )
        self.n_chains = n_chains
        self.adapt_rate = adapt_rate
        self.target_accept_rate = target_accept_rate

        # Initialize b
        self.b = init_b.clone()

        # Compute initial log logpdfs
        self.logpdfs, self.aux = self.logpdfs_aux_fn(self.b)

        # Proposal noise initialization
        self._noise = torch.empty_like(self.b)
        self.step_sizes = torch.full((1, self.b.size(-2)), init_step_size, dtype=dtype)

        # Statistics tracking
        self.n_samples = torch.tensor(0, dtype=torch.int64)
        self.n_accepted = torch.zeros(self.b.size(-2), dtype=dtype)

    def step(self):
        """Performs a single kernel step."""
        # Generate proposal noise
        self._noise.normal_()

        # Get the proposal
        proposed_state = self.b + self._noise * self.step_sizes.unsqueeze(-1)
        proposed_logpdfs, proposed_aux = self.logpdfs_aux_fn(proposed_state)
        logpdf_diff = proposed_logpdfs - self.logpdfs

        # Vectorized acceptance decision
        log_uniform = torch.log(torch.rand_like(logpdf_diff))
        accept_mask = log_uniform < logpdf_diff

        torch.where(accept_mask.unsqueeze(-1), proposed_state, self.b, out=self.b)
        torch.where(accept_mask, proposed_logpdfs, self.logpdfs, out=self.logpdfs)

        torch.where(
            accept_mask.unsqueeze(-1), proposed_aux.psi, self.aux.psi, out=self.aux.psi
        )
        torch.where(
            accept_mask, proposed_aux.logliks, self.aux.logliks, out=self.aux.logliks
        )

        # Update statistics
        self.n_samples += 1
        mean_accept_mask = accept_mask.to(get_dtype()).mean(dim=0)
        self.n_accepted += mean_accept_mask

        # Update step sizes
        adaptation = (mean_accept_mask - self.target_accept_rate) * self.adapt_rate
        self.step_sizes *= torch.exp(adaptation)

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
