from __future__ import annotations

from typing import Any

from .metrics import best_seen, simple_regret, top_k_hit_rate
from .oracle import OfflineOracle
from .replay import replay


def __getattr__(name: str) -> Any:
    if name == "run_severson_benchmark":
        from .severson_benchmark import run_severson_benchmark
        return run_severson_benchmark
    elif name == "run_dynamic_cycling_benchmark":
        from .dynamic_cycling_benchmark import run_dynamic_cycling_benchmark
        return run_dynamic_cycling_benchmark
    elif name == "run_attia_benchmark":
        from .attia_benchmark import run_attia_benchmark
        return run_attia_benchmark
    elif name == "run_attia_continuous_benchmark":
        from .attia_continuous_benchmark import run_attia_continuous_benchmark
        return run_attia_continuous_benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OfflineOracle",
    "best_seen",
    "replay",
    "simple_regret",
    "top_k_hit_rate",
    "run_severson_benchmark",
    "run_dynamic_cycling_benchmark",
]
