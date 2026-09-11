from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence

from .config import REFERENCE_SLURRY_STEPS


class ConvergenceStatus(StrEnum):
    INSUFFICIENT_CHECKPOINTS = "INSUFFICIENT_CHECKPOINTS"
    STABILITY_OBSERVED = "STABILITY_OBSERVED"
    NOT_STABLE = "NOT_STABLE"
    REFERENCE_NOT_AVAILABLE = "REFERENCE_NOT_AVAILABLE"
    VALIDATED_AGAINST_REFERENCE = "VALIDATED_AGAINST_REFERENCE"


@dataclass(frozen=True)
class Checkpoint:
    step: int
    metrics: Mapping[str, float]
    wall_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.step < 0 or any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("checkpoint steps and metrics must be finite and non-negative")


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    short_value: float
    reference_value: float
    absolute_error: float
    relative_error: float | None
    normalized_error: float


@dataclass(frozen=True)
class ConvergenceReport:
    status: ConvergenceStatus
    checkpoint_count: int
    requested_steps: int | None
    reference_steps: int
    reference_available: bool
    comparisons: tuple[MetricComparison, ...] = ()
    available_metrics: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


def checkpoints_from_manifest(manifest: Mapping[str, Any]) -> tuple[Checkpoint, ...]:
    progress = manifest.get("progress", {})
    if not isinstance(progress, Mapping):
        raise ValueError("ARTISTIC progress must be a mapping")
    raw = manifest.get("checkpoints", progress.get("checkpoints", progress.get("checkpoint_steps", [])))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("ARTISTIC checkpoints must be a sequence")
    result: list[Checkpoint] = []
    for item in raw:
        if isinstance(item, int):
            item = {"step": item}
        if not isinstance(item, Mapping) or "step" not in item:
            raise ValueError("ARTISTIC checkpoint entries need a step")
        metrics = item.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("ARTISTIC checkpoint metrics must be a mapping")
        result.append(Checkpoint(int(item["step"]), {str(name): float(value) for name, value in metrics.items()}, item.get("wall_seconds")))
    return tuple(result)


def stability_status(checkpoints: Sequence[Checkpoint], *, tolerance: float = 0.01, minimum_checkpoints: int = 3) -> ConvergenceStatus:
    if len(checkpoints) < minimum_checkpoints:
        return ConvergenceStatus.INSUFFICIENT_CHECKPOINTS
    names = set(checkpoints[-1].metrics)
    if not names:
        return ConvergenceStatus.INSUFFICIENT_CHECKPOINTS
    recent = checkpoints[-minimum_checkpoints:]
    for name in names:
        values = [float(item.metrics[name]) for item in recent if name in item.metrics]
        if len(values) != len(recent):
            return ConvergenceStatus.INSUFFICIENT_CHECKPOINTS
        scale = max(abs(values[-1]), 1e-12)
        if max(abs(value - values[-1]) for value in values[:-1]) / scale > tolerance:
            return ConvergenceStatus.NOT_STABLE
    return ConvergenceStatus.STABILITY_OBSERVED


def compare_to_reference(short: Checkpoint, reference: Checkpoint, *, reference_steps: int = REFERENCE_SLURRY_STEPS) -> tuple[MetricComparison, ...]:
    if reference.step < reference_steps:
        raise ValueError(f"reference comparison requires a {reference_steps:,}-step reference checkpoint")
    comparisons: list[MetricComparison] = []
    for name in sorted(set(short.metrics) & set(reference.metrics)):
        short_value = float(short.metrics[name])
        reference_value = float(reference.metrics[name])
        absolute = abs(short_value - reference_value)
        relative = absolute / abs(reference_value) if reference_value else None
        comparisons.append(MetricComparison(name, short_value, reference_value, absolute, relative, absolute / max(abs(reference_value), 1e-12)))
    return tuple(comparisons)


def build_convergence_report(
    checkpoints: Sequence[Checkpoint],
    *,
    reference_checkpoints: Sequence[Checkpoint] | None = None,
    requested_steps: int | None = None,
    reference_steps: int = REFERENCE_SLURRY_STEPS,
    tolerance: float = 0.01,
) -> ConvergenceReport:
    available_metrics = tuple(sorted({name for item in checkpoints for name in item.metrics}))
    if not reference_checkpoints:
        return ConvergenceReport(
            ConvergenceStatus.REFERENCE_NOT_AVAILABLE, len(checkpoints), requested_steps, reference_steps, False,
            available_metrics=available_metrics, diagnostics=("No 20M-step reference output was supplied; no equivalence claim is allowed.",),
        )
    reference = max(reference_checkpoints, key=lambda item: item.step)
    if reference.step < reference_steps:
        return ConvergenceReport(
            ConvergenceStatus.REFERENCE_NOT_AVAILABLE, len(checkpoints), requested_steps, reference_steps, False,
            available_metrics=available_metrics, diagnostics=("The supplied reference checkpoints do not reach the exact 20M-step horizon.",),
        )
    if len(checkpoints) < 2:
        return ConvergenceReport(
            ConvergenceStatus.INSUFFICIENT_CHECKPOINTS, len(checkpoints), requested_steps, reference_steps, True,
            available_metrics=available_metrics,
        )
    comparisons = compare_to_reference(max(checkpoints, key=lambda item: item.step), reference, reference_steps=reference_steps)
    status = stability_status(checkpoints, tolerance=tolerance)
    final_status = {
        ConvergenceStatus.INSUFFICIENT_CHECKPOINTS: ConvergenceStatus.INSUFFICIENT_CHECKPOINTS,
        ConvergenceStatus.NOT_STABLE: ConvergenceStatus.NOT_STABLE,
        ConvergenceStatus.STABILITY_OBSERVED: ConvergenceStatus.VALIDATED_AGAINST_REFERENCE,
    }[status]
    return ConvergenceReport(
        final_status, len(checkpoints), requested_steps, reference_steps, True, comparisons,
        available_metrics=available_metrics,
    )
