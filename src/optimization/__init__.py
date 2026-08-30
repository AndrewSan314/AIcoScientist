from __future__ import annotations

from typing import Any

from .backend import OptimizerBackend
from .botorch_backend import BoTorchBackend
from .candidates import normalize_candidate_schema, remove_observed
from .constraints import apply_constraints
from .finite_pool import FiniteCandidatePool
from .objective import OptimizationObjective
from .proposal import CandidateProposal, ExperimentProposal


def __getattr__(name: str) -> Any:
    if name == "recommend":
        from .recommender import recommend
        return recommend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "apply_constraints",
    "BoTorchBackend",
    "CandidateProposal",
    "ExperimentProposal",
    "FiniteCandidatePool",
    "normalize_candidate_schema",
    "OptimizationObjective",
    "OptimizerBackend",
    "recommend",
    "remove_observed",
]
