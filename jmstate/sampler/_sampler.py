import warnings
from typing import Callable

import torch
from tqdm import tqdm


class MetropolisHastingsSampler:
    """A robust Metropolis-Hastings sampler with adaptive step size."""

    def __init__(
        self,
        log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
        init_state: torch.Tensor,
        init_step_size: float,
        adapt_rate: float,
        target_accept_rate: float,
    ):
        """Initialize the Metropolis-Hastings sampler kernel.

        Args:
            log_prob_fn (Callable[[torch.Tensor], torch.Tensor]): Function that computes log probability.
            init_state (torch.Tensor): Starting state for the chain.
            init_step_size (float, optional): Kernel standard error in Metropolis Hastings.
            adapt_rate (float, optional): Adaptation rate for the step_size.
            target_accept_rate (float, optional): Mean acceptance target.

        Raises:
            RuntimeError: If the initial log prob fails to be computed.
        """
        self.log_prob_fn = log_prob_fn
        self.adapt_rate = adapt_rate
        self.target_accept_rate = target_accept_rate

        # Initialize state
        self.current_state = init_state.clone().detach()

        # Compute initial log probability
        try:
            self.current_log_prob = self.log_prob_fn(self.current_state)
        except Exception as e:
            raise RuntimeError(f"Failed to compute initial log probability: {e}")

        # Steps initialization
        self.step_sizes = torch.full(
            (self.current_state.shape[0],), init_step_size, dtype=torch.float32
        )

        # Statistics tracking
        self.n_samples = torch.tensor(0.0, dtype=torch.float32)
        self.n_accepted = torch.zeros(
            (self.current_state.shape[0],), dtype=torch.float32
        )

        self._check()

    def _check(self):
        """Check if every input is valid.

        Raises:
            TypeError: If the function is not callable.
            ValueError: If init_step_size is not strictly positive.
            ValueError: If adapt_rate is not strictly positive.
            ValueError: If target_accept_rate is not in (0, 1).
        """
        if not callable(self.log_prob_fn):
            raise TypeError("log_prob_fn must be callable")

        if self.step_sizes[0] <= 0:
            raise ValueError("step_size must be strictly positive")

        if self.adapt_rate <= 0:
            raise ValueError("adapt_rate must be strictly positive")

        if not 0 < self.target_accept_rate < 1:
            raise ValueError("target_accept_rate must be between 0 and 1")

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a single kernel step.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing current_state and current_log_prob.
        """
        # Detach current state to avoid gradient accumulation
        self.current_state = self.current_state.detach()
        self.current_log_prob = self.current_log_prob.detach()

        # Generate proposal isotropic noise
        noise = torch.randn_like(self.current_state, dtype=torch.float32)

        # Get the proposal
        proposed_state = self.current_state + noise * self.step_sizes.view(-1, 1)

        # Compute proposal log probability
        try:
            proposed_log_prob = self.log_prob_fn(proposed_state)
        except Exception as e:
            warnings.warn(f"Failed to compute proposal log probability: {e}")
            return self.current_state, self.current_log_prob

        # Compute acceptance probability
        log_prob_diff = proposed_log_prob - self.current_log_prob

        # Vectorized acceptance decision
        log_uniform = torch.log(torch.clamp(torch.rand_like(log_prob_diff), min=1e-8))
        accept_mask = log_uniform < log_prob_diff

        # Update accepted states
        self.current_state[accept_mask] = proposed_state[accept_mask]
        self.current_log_prob[accept_mask] = proposed_log_prob[accept_mask]

        # Update statistics
        self.n_samples += 1
        self.n_accepted += accept_mask.float()

        # Adapt step sizes
        self._adapt_step_sizes(accept_mask)

        return self.current_state, self.current_log_prob

    def _adapt_step_sizes(self, accept_mask: torch.Tensor):
        adaptation = (accept_mask.float() - self.target_accept_rate) * self.adapt_rate
        self.step_sizes *= torch.exp(adaptation)

    def warmup(self, warmup: int) -> None:
        """Warmups the MCMC.

        Args:
            warmup (int): The number of warmup steps.

        Raises:
            ValueError: If the warmup steps is not positive.
        """
        if warmup < 0:
            raise ValueError("Warmup must be a non-negative integer")

        with torch.no_grad():
            for _ in range(warmup):
                self.step()

    def loop(
        self,
        n_iter: int,
        cont_warmup: int,
        job: Callable[[int], None],
        desc: str,
        verbose: bool,
    ) -> None:
        """Loops while subsampling.

        Args:
            n_iter (int): The number of iterations to do.
            cont_warmup (int): The sublamping MCMC number.
            job (Callable[[], None]): The function to execute.
            desc (str): The description during the loop.
            verbose (bool): Wheter or not to show progress.

        Raises:
            ValueError: If n_iter is not greater or equal to one.
        """
        if n_iter < 1:
            raise ValueError(f"n_iter must be at least one, got {n_iter}")

        for iter in tqdm(range(n_iter), desc=desc, disable=not verbose):
            try:
                self.warmup(cont_warmup)
                job(iter)
            except Exception as e:
                warnings.warn(f"Error in iteration {iter}: {e}")
                continue

    @property
    def acceptance_rates(self) -> torch.Tensor:
        """Gets the acceptance_rate mean.

        Returns:
            torch.Tensor: The means of the acceptance_rates accross iterations.
        """
        return self.n_accepted / torch.clamp(self.n_samples, min=1.0)

    @property
    def mean_step_size(self) -> float:
        """Gets the mean step size.

        Returns:
            float: The mean step size.
        """
        return self.step_sizes.mean().item()
