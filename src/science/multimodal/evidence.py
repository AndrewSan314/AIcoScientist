from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def entropy(beliefs: Mapping[str, float]) -> float:
    probs = np.asarray(list(beliefs.values()), dtype=np.float64)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs))) if len(probs) else 0.0


def bayesian_update(
    beliefs: Mapping[str, float],
    log_likelihoods: Mapping[str, float],
) -> dict[str, float]:
    """Updates hypothesis beliefs in log space and normalizes defensively."""
    keys = list(beliefs)
    if not keys:
        return {}
    log_scores = np.array(
        [np.log(max(float(beliefs[k]), 1e-12)) + float(log_likelihoods.get(k, -1000.0)) for k in keys],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(log_scores)):
        return {k: 1.0 / len(keys) for k in keys}
    log_scores -= np.max(log_scores)
    probs = np.exp(np.clip(log_scores, -700.0, 0.0))
    probs /= np.sum(probs)
    return {k: float(p) for k, p in zip(keys, probs)}


class MultimodalEvidenceLedger:
    """Append-only in-memory evidence ledger with optional JSONL persistence."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(event)
        self.events.append(record)
        return record

    def write_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = ["MultimodalEvidenceLedger", "bayesian_update", "entropy"]
