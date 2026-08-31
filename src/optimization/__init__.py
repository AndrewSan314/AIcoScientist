from __future__ import annotations

from typing import Any

from .backend import (
    AcquisitionEvaluationError,
    OptimizerBackend,
    RETIRED_STRATEGIES,
    STRATEGY_ALIASES,
    SUPPORTED_STRATEGIES,
    UnsupportedStrategyError,
    resolve_strategy,
)
from .botorch_backend import BoTorchBackend
from .candidates import normalize_candidate_schema, remove_observed
from .constraints import apply_constraints
from .finite_pool import FiniteCandidatePool, compute_candidate_pool_fingerprint
from .objective import OptimizationObjective
from .proposal import CandidateProposal, ExperimentProposal


def __getattr__(name: str) -> Any:
    if name == "recommend":
        from .recommender import recommend
        return recommend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AcquisitionEvaluationError",
    "apply_constraints",
    "BoTorchBackend",
    "CandidateProposal",
    "compute_candidate_pool_fingerprint",
    "ExperimentProposal",
    "FiniteCandidatePool",
    "normalize_candidate_schema",
    "OptimizationObjective",
    "OptimizerBackend",
    "recommend",
    "remove_observed",
    "resolve_strategy",
    "RETIRED_STRATEGIES",
    "STRATEGY_ALIASES",
    "SUPPORTED_STRATEGIES",
    "UnsupportedStrategyError",
]

