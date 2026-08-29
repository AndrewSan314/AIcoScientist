from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def _validate_objective(objective: str) -> None:
    if objective not in {"maximize", "minimize"}:
        raise ValueError("objective must be 'maximize' or 'minimize'")


def best_seen(values: Iterable[float], objective: str = "maximize") -> float:
    _validate_objective(objective)
    values = list(values)
    if not values:
        raise ValueError("values must not be empty")
    return (max if objective == "maximize" else min)(values)


def simple_regret(best: float, global_best: float, objective: str = "maximize") -> float:
    _validate_objective(objective)
    return float(global_best - best if objective == "maximize" else best - global_best)


def top_k_hit_rate(
    recommended: Sequence,
    ground_truth: Mapping | Sequence,
    k: int,
    objective: str = "maximize",
) -> float:
    _validate_objective(objective)
    if k <= 0:
        raise ValueError("k must be positive")
    if isinstance(ground_truth, Mapping):
        ranked = sorted(
            ground_truth,
            key=ground_truth.get,
            reverse=objective == "maximize",
        )
        truth = set(ranked[:k])
    else:
        truth = set(list(ground_truth)[:k])
    return len(set(list(recommended)[:k]) & truth) / len(truth) if truth else 0.0
