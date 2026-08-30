from __future__ import annotations

from typing import Any

from .acquisition import ucb
from .candidates import normalize_candidate_schema, remove_observed
from .constraints import apply_constraints


def __getattr__(name: str) -> Any:
    if name == "recommend":
        from .recommender import recommend
        return recommend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "apply_constraints",
    "normalize_candidate_schema",
    "recommend",
    "remove_observed",
    "ucb",
]
