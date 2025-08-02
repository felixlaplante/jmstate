from typing import Any, Callable

import torch
from tqdm import tqdm

from ..typedefs._defs import Tensor0D, Tensor1D, Tensor2D, Tensor3D


class MetropolisHastingsSampler:
    """A robust Metropolis-Hastings sampler with adaptive step size."""

    pdf_fn: Callable[[Tensor3D], tuple[Tensor2D, list[torch.Tensor]]]
    init_state: Tensor3D
    n_chains: int
    init_step_size: float
    adapt_rate: float
    target_accept_rate: float
    current_state: Tensor3D
    current_pdf: Tensor2D
    current_aux: list[torch.Tensor]
    step_sizes: Tensor1D
    n_samples: Tensor0D
    n_accepted: Tensor1D

    def __init__(
        self,
        pdf_fn: Callable[[Tensor3D], tuple[Tensor2D, list[torch.Tensor]]],
        init_state: Tensor3D,
        n_chains: int,
        init_step_size: float,
        adapt_rate: float,
        target_accept_rate: float,
    ):
        """Initialize the Metropolis-Hastings sampler kernel.

        Args:
            pdf_fn (Callable[[Tensor3D], tuple[Tensor2D, list[torch.Tensor]]]): pdf with aux.
            init_state (Tensor3D): Starting state for the chain.
            n_chains (int): The number of parallel chains to spawn.
            init_step_size (float, optional): Kernel step in Metropolis Hastings.
            adapt_rate (float, optional): Adaptation rate for the step_size.
            target_accept_rate (float, optional): Mean acceptance target.

        Raises:
            RuntimeError: If the initial log prob fails to be computed.
        """
        self.pdf_fn = pdf_fn
        self.n_chains = n_chains
        self.adapt_rate = adapt_rate
        self.target_accept_rate = target_accept_rate

        # Initialize state
        self.current_state = init_state

        # Compute initial log pdf
        self.current_pdf, self.current_aux = self.pdf_fn(self.current_state)

        # Steps initialization
        self.step_sizes = torch.full(
            (init_state.shape[-2],), init_step_size, dtype=torch.float32
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
            ValueError: If adapt_rate is not strictly positive.
            ValueError: If target_accept_rate is not in (0, 1).
        """
        if self.n_chains <= 0:
            raise ValueError(f"n_chains must be strictly positive, got {self.n_chains}")
        if self.step_sizes[0] <= 0:
            raise ValueError(
                f"init_step_size must be strictly positive, got {self.step_sizes[0]}"
            )
        if self.adapt_rate <= 0:
            raise ValueError(
                f"adapt_rate must be strictly positive, got {self.adapt_rate}"
            )
        if not 0 < self.target_accept_rate < 1:
            raise ValueError(
                f"target_accept_rate must be in (0, 1), got {self.target_accept_rate}"
            )

    def step(self) -> tuple[tuple[Tensor2D, Tensor2D], list[torch.Tensor]]:
        """Performs a single kernel step.

        Returns:
            list[torch.Tensor]: A tuple containing state and pdf.
        """
        # Detach current state to avoid gradient accumulation
        self.current_state = self.current_state.detach()
        self.current_pdf = self.current_pdf.detach()

        # Generate proposal isotropic noise
        noise = torch.randn_like(self.current_state, dtype=torch.float32)

        # Get the proposal
        proposed_state = self.current_state + noise * self.step_sizes.view(1, -1, 1)
        proposed_pdf, proposed_aux = self.pdf_fn(proposed_state)
        pdf_diff = proposed_pdf - self.current_pdf

        # Vectorized acceptance decision
        log_uniform = torch.log(torch.rand_like(pdf_diff))
        accept_mask = log_uniform < pdf_diff

        self.current_state[accept_mask] = proposed_state[accept_mask]
        self.current_pdf[accept_mask] = self.current_pdf[accept_mask]

        for i, _ in enumerate(self.current_aux):
            self.current_aux[i][accept_mask] = proposed_aux[i][accept_mask]

        # Update statistics
        self.n_samples += 1
        self.n_accepted += accept_mask.to(torch.float32).mean(dim=0)

        self._adapt_step_sizes(accept_mask)

        return (self.current_state, self.current_pdf), self.current_aux

    def _adapt_step_sizes(self, accept_mask: Tensor1D):
        adaptation = (
            accept_mask.to(torch.float32).mean(dim=0) - self.target_accept_rate
        ) * self.adapt_rate
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
        n_iterations: int,
        cont_warmup: int,
        job: Callable[[int], None],
        desc: str,
        verbose: bool,
    ) -> None:
        """Loops while subsampling.

        Args:
            n_iterations (int): The number of iterations to do.
            cont_warmup (int): The sublamping MCMC number.
            job (Callable[[], None]): The function to execute.
            desc (str): The description during the loop.
            verbose (bool): Wheter or not to show progress.

        Raises:
            ValueError: If n_iter is not greater or equal to one.
            RuntimeError: If an iteration fails.
        """
        for iteration in tqdm(range(n_iterations), desc=desc, disable=not verbose):
            try:
                self.warmup(cont_warmup)
                job(iteration)
            except Exception as e:
                raise RuntimeError(
                    f"Error in Metropolis Hastings iteration {iteration}: {e}"
                ) from e

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
