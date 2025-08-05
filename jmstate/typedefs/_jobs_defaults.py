from typing import Any, Final

from ..jobs._computation import ComputeCriteria, ComputeEBEs, ComputeFIM
from ..jobs._fitting import DeterministicFit, RandomFit
from ..jobs._prediction import PredictSurvLogps, PredictTrajectories, PredictY
from ._defs import Job

# Constants
DEFAULT_HYPERPARAMETERS: Final[dict[type[Job], dict[str, Any]]] = {
    ComputeCriteria: {
        "max_iterations": 200,
        "n_chains": 10,
        "warmup": 300,
        "n_steps": 5,
    },
    ComputeEBEs: {"max_iterations": 200, "n_chains": 10, "warmup": 300, "n_steps": 5},
    ComputeFIM: {"max_iterations": 200, "n_chains": 10, "warmup": 300, "n_steps": 5},
    DeterministicFit: {
        "max_iterations": 500,
        "n_chains": 1,
        "warmup": 0,
        "n_steps": 0,
    },
    RandomFit: {"max_iterations": 500, "n_chains": 10, "warmup": 300, "n_steps": 5},
    PredictSurvLogps: {
        "max_iterations": 200,
        "n_chains": 10,
        "warmup": 300,
        "n_steps": 5,
    },
    PredictTrajectories: {
        "max_iterations": 200,
        "n_chains": 10,
        "warmup": 300,
        "n_steps": 5,
    },
    PredictY: {"max_iterations": 200, "n_chains": 10, "warmup": 300, "n_steps": 5},
}
