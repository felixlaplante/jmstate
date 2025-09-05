from typing import Any, Final

from ..jobs._computation import ComputeCriteria, ComputeEBEs, ComputeFIM
from ..jobs._fitting import Fit
from ..jobs._prediction import PredictSurvLogps, PredictTrajectories, PredictY
from ._defs import Job

# Constants
DEFAULT_HYPERPARAMETERS_FIELDS: Final[tuple[str, ...]] = (
    "max_iterations",
    "n_chains",
    "warmup",
    "n_steps",
)


DEFAULT_HYPERPARAMETERS: Final[dict[type[Job], dict[str, Any]]] = {
    ComputeCriteria: {
        "max_iterations": 100,
        "n_chains": 10,
        "warmup": 200,
        "n_steps": 10,
    },
    ComputeEBEs: {"max_iterations": 100, "n_chains": 10, "warmup": 200, "n_steps": 10},
    ComputeFIM: {"max_iterations": 100, "n_chains": 10, "warmup": 200, "n_steps": 10},
    Fit: {"max_iterations": 500, "n_chains": 10, "warmup": 200, "n_steps": 10},
    PredictSurvLogps: {
        "max_iterations": 100,
        "n_chains": 10,
        "warmup": 200,
        "n_steps": 10,
    },
    PredictTrajectories: {
        "max_iterations": 100,
        "n_chains": 10,
        "warmup": 200,
        "n_steps": 10,
    },
    PredictY: {"max_iterations": 100, "n_chains": 10, "warmup": 200, "n_steps": 10},
}
