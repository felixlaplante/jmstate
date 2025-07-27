import warnings
from typing import Any, Callable

import torch
from tqdm import tqdm


class MetropolisHastingsSampler:
    """A robust Metropolis-Hastings sampler with adaptive step size."""

    pdf_fn: Callable[[torch.Tensor], tuple[torch.Tensor, Any]]
    init_state: torch.Tensor
    init_step_size: float
    adapt_rate: float
    target_accept_rate: float
    current_state: torch.Tensor
    current_pdf: torch.Tensor
    step_sizes: torch.Tensor
    n_samples: torch.Tensor
    n_accepted: torch.Tensor

    def __init__(
        self,
        pdf_fn: Callable[[torch.Tensor], tuple[torch.Tensor, Any]],
        init_state: torch.Tensor,
        init_step_size: float,
        adapt_rate: float,
        target_accept_rate: float,
    ):
        """Initialize the Metropolis-Hastings sampler kernel.

        Args:
            pdf_fn (Callable[[torch.Tensor], tuple[torch.Tensor, Any]]): pdf with aux.
            init_state (torch.Tensor): Starting state for the chain.
            init_step_size (float, optional): Kernel step in Metropolis Hastings.
            adapt_rate (float, optional): Adaptation rate for the step_size.
            target_accept_rate (float, optional): Mean acceptance target.

        Raises:
            RuntimeError: If the initial log prob fails to be computed.
        """
        self.pdf_fn = pdf_fn
        self.adapt_rate = adapt_rate
        self.target_accept_rate = target_accept_rate

        # Initialize state
        self.current_state = init_state.clone().detach()

        # Compute initial log pdf
        try:
            self.current_pdf, _ = self.pdf_fn(self.current_state)
        except Exception as e:
            raise RuntimeError(f"Failed to compute initial log pdf: {e}") from e

        # Steps initialization
        self.step_sizes = torch.full(
            (self.current_state.shape[0],), init_step_size, dtype=torch.float32
        )

        # Statistics tracking
        self.n_samples = torch.tensor(0, dtype=torch.int64)
        self.n_accepted = torch.zeros((self.current_state.shape[0],), dtype=torch.int64)

        self._check()

    def _check(self):
        """Check if every input is valid.

        Raises:
            TypeError: If the function is not callable.
            ValueError: If init_step_size is not strictly positive.
            ValueError: If adapt_rate is not strictly positive.
            ValueError: If target_accept_rate is not in (0, 1).
        """
        if not callable(self.pdf_fn):
            raise TypeError("pdf_fn must be callable")

        if self.step_sizes[0] <= 0:
            raise ValueError("step_size must be strictly positive")

        if self.adapt_rate <= 0:
            raise ValueError("adapt_rate must be strictly positive")

        if not 0 < self.target_accept_rate < 1:
            raise ValueError("target_accept_rate must be between 0 and 1")

    def step(self) -> tuple[tuple[torch.Tensor, torch.Tensor], Any]:
        """Performs a single kernel step.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing state and pdf.
        """
        # Detach current state to avoid gradient accumulation
        self.current_state = self.current_state.detach()
        self.current_pdf = self.current_pdf.detach()

        # Generate proposal isotropic noise
        noise = torch.randn_like(self.current_state, dtype=torch.float32)

        # Get the proposal
        proposed_state = self.current_state + noise * self.step_sizes.view(-1, 1)

        # Compute proposal log pdf
        proposed_pdf, aux = self.pdf_fn(proposed_state)

        # Compute acceptance pdf
        pdf_diff = proposed_pdf - self.current_pdf

        # Vectorized acceptance decision
        log_uniform = torch.log(torch.clamp(torch.rand_like(pdf_diff), min=1e-8))
        accept_mask = log_uniform < pdf_diff

        # Update accepted states
        self.current_state[accept_mask] = proposed_state[accept_mask]
        self.current_pdf[accept_mask] = proposed_pdf[accept_mask]

        # Update statistics
        self.n_samples += 1
        self.n_accepted += accept_mask

        # Adapt step sizes
        self._adapt_step_sizes(accept_mask)

        return (self.current_state, self.current_pdf), aux

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
                warnings.warn(f"Error in iteration {iter}: {e}", stacklevel=2)
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
