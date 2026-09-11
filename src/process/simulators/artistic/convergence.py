from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence

from .config import REFERENCE_SLURRY_STEPS


class ConvergenceStatus(StrEnum):
    INSUFFICIENT_CHECKPOINTS = "INSUFFICIENT_CHECKPOINTS"
    STABILITY_OBSERVED = "STABILITY_OBSERVED"
    NOT_STABLE = "NOT_STABLE"
    REFERENCE_NOT_AVAILABLE = "REFERENCE_NOT_AVAILABLE"
    REFERENCE_METRIC_MISSING = "REFERENCE_METRIC_MISSING"
    REFERENCE_OUTSIDE_TOLERANCE = "REFERENCE_OUTSIDE_TOLERANCE"
    VALIDATED_AGAINST_REFERENCE = "VALIDATED_AGAINST_REFERENCE"


class ReferenceAgreementStatus(StrEnum):
    REFERENCE_NOT_AVAILABLE = "REFERENCE_NOT_AVAILABLE"
    REFERENCE_METRIC_MISSING = "REFERENCE_METRIC_MISSING"
    REFERENCE_OUTSIDE_TOLERANCE = "REFERENCE_OUTSIDE_TOLERANCE"
    REFERENCE_WITHIN_TOLERANCE = "REFERENCE_WITHIN_TOLERANCE"


@dataclass(frozen=True)
class MetricTolerancePolicy:
    """Explicit metric error policy; an absent rule cannot validate agreement."""

    relative_tolerances: Mapping[str, float] = field(default_factory=dict)
    absolute_tolerances: Mapping[str, float] = field(default_factory=dict)
    default_relative_tolerance: float | None = None
    near_zero_reference_threshold: float = 1e-12
    required_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (*self.relative_tolerances.items(), *self.absolute_tolerances.items()):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"tolerance for metric {name!r} must be finite and non-negative")
        if self.default_relative_tolerance is not None and (
            not math.isfinite(float(self.default_relative_tolerance)) or float(self.default_relative_tolerance) < 0
        ):
            raise ValueError("default relative tolerance must be finite and non-negative")
        if not math.isfinite(float(self.near_zero_reference_threshold)) or self.near_zero_reference_threshold <= 0:
            raise ValueError("near-zero reference threshold must be positive and finite")
        if len(set(self.required_metrics)) != len(self.required_metrics):
            raise ValueError("required metrics must be unique")

    def metrics(self, short: Checkpoint, reference: Checkpoint) -> tuple[str, ...]:
        return self.required_metrics or tuple(sorted(set(short.metrics) | set(reference.metrics)))

    def rule(self, metric: str, reference_value: float) -> tuple[str, float] | None:
        if abs(reference_value) <= self.near_zero_reference_threshold:
            tolerance = self.absolute_tolerances.get(metric)
            return ("absolute", float(tolerance)) if tolerance is not None else None
        tolerance = self.relative_tolerances.get(metric, self.default_relative_tolerance)
        return ("relative", float(tolerance)) if tolerance is not None else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_tolerances": dict(sorted(self.relative_tolerances.items())),
            "absolute_tolerances": dict(sorted(self.absolute_tolerances.items())),
            "default_relative_tolerance": self.default_relative_tolerance,
            "near_zero_reference_threshold": self.near_zero_reference_threshold,
            "required_metrics": list(self.required_metrics),
        }


@dataclass(frozen=True)
class Checkpoint:
    step: int
    metrics: Mapping[str, float]
    wall_seconds: float | None = None
    stage: str | None = None
    source_log: str | None = None

    def __post_init__(self) -> None:
        if self.step < 0 or any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("checkpoint steps and metrics must be finite and non-negative")
        if self.wall_seconds is not None and (not math.isfinite(float(self.wall_seconds)) or self.wall_seconds < 0):
            raise ValueError("checkpoint wall time must be finite and non-negative")


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    short_value: float
    reference_value: float
    absolute_error: float
    relative_error: float | None
    normalized_error: float | None
    error_rule: str | None = None
    tolerance: float | None = None
    within_tolerance: bool = False


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
    stability_status: ConvergenceStatus = ConvergenceStatus.INSUFFICIENT_CHECKPOINTS
    reference_agreement_status: ReferenceAgreementStatus = ReferenceAgreementStatus.REFERENCE_NOT_AVAILABLE
    required_metrics: tuple[str, ...] = ()
    missing_metrics: tuple[str, ...] = ()
    tolerance_policy: Mapping[str, Any] = field(default_factory=dict)


def checkpoints_from_manifest(manifest: Mapping[str, Any]) -> tuple[Checkpoint, ...]:
    progress = manifest.get("progress", {})
    if not isinstance(progress, Mapping):
        raise ValueError("ARTISTIC progress must be a mapping")
    raw = manifest.get("checkpoints", progress.get("checkpoints", progress.get("checkpoint_steps", [])))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("ARTISTIC checkpoints must be a sequence")
    result: list[Checkpoint] = []
    for item in raw:
        if isinstance(item, int) and not isinstance(item, bool):
            item = {"step": item}
        if not isinstance(item, Mapping) or "step" not in item:
            raise ValueError("ARTISTIC checkpoint entries need a step")
        metrics = item.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("ARTISTIC checkpoint metrics must be a mapping")
        result.append(Checkpoint(
            int(item["step"]), {str(name): float(value) for name, value in metrics.items()}, item.get("wall_seconds"),
            item.get("stage"), item.get("source_log"),
        ))
    return tuple(sorted(result, key=lambda item: item.step))


def stability_status(checkpoints: Sequence[Checkpoint], *, tolerance: float = 0.01, minimum_checkpoints: int = 3) -> ConvergenceStatus:
    ordered = tuple(sorted(checkpoints, key=lambda item: item.step))
    if len(ordered) < minimum_checkpoints:
        return ConvergenceStatus.INSUFFICIENT_CHECKPOINTS
    names = set(ordered[-1].metrics)
    if not names:
        return ConvergenceStatus.INSUFFICIENT_CHECKPOINTS
    recent = ordered[-minimum_checkpoints:]
    for name in names:
        values = [float(item.metrics[name]) for item in recent if name in item.metrics]
        if len(values) != len(recent):
            return ConvergenceStatus.INSUFFICIENT_CHECKPOINTS
        scale = max(abs(values[-1]), 1e-12)
        if max(abs(value - values[-1]) for value in values[:-1]) / scale > tolerance:
            return ConvergenceStatus.NOT_STABLE
    return ConvergenceStatus.STABILITY_OBSERVED


def compare_to_reference(
    short: Checkpoint,
    reference: Checkpoint,
    *,
    reference_steps: int = REFERENCE_SLURRY_STEPS,
    tolerance_policy: MetricTolerancePolicy | None = None,
    tolerance: float | None = None,
) -> tuple[MetricComparison, ...]:
    if reference.step != reference_steps:
        raise ValueError(f"reference comparison requires an exact {reference_steps:,}-step reference checkpoint")
    if tolerance_policy is None and tolerance is not None:
        tolerance_policy = MetricTolerancePolicy(default_relative_tolerance=tolerance)
    policy = tolerance_policy or MetricTolerancePolicy()
    comparisons: list[MetricComparison] = []
    for name in sorted(set(short.metrics) & set(reference.metrics)):
        short_value = float(short.metrics[name])
        reference_value = float(reference.metrics[name])
        absolute = abs(short_value - reference_value)
        relative = absolute / abs(reference_value) if abs(reference_value) > policy.near_zero_reference_threshold else None
        rule = policy.rule(name, reference_value)
        within = False if rule is None else (absolute <= rule[1] if rule[0] == "absolute" else relative is not None and relative <= rule[1])
        comparisons.append(MetricComparison(
            name, short_value, reference_value, absolute, relative, relative,
            rule[0] if rule else None, rule[1] if rule else None, within,
        ))
    return tuple(comparisons)


def build_convergence_report(
    checkpoints: Sequence[Checkpoint],
    *,
    reference_checkpoints: Sequence[Checkpoint] | None = None,
    requested_steps: int | None = None,
    reference_steps: int = REFERENCE_SLURRY_STEPS,
    tolerance: float | None = None,
    tolerance_policy: MetricTolerancePolicy | None = None,
    stability_tolerance: float = 0.01,
) -> ConvergenceReport:
    short = tuple(sorted(checkpoints, key=lambda item: item.step))
    available_metrics = tuple(sorted({name for item in short for name in item.metrics}))
    if tolerance_policy is None and tolerance is not None:
        tolerance_policy = MetricTolerancePolicy(default_relative_tolerance=tolerance)
        stability_tolerance = tolerance
    policy = tolerance_policy or MetricTolerancePolicy()
    stable = stability_status(short, tolerance=stability_tolerance)
    base = {
        "checkpoint_count": len(short), "requested_steps": requested_steps, "reference_steps": reference_steps,
        "available_metrics": available_metrics, "stability_status": stable,
        "tolerance_policy": policy.as_dict(),
    }
    exact_reference = next((item for item in sorted(reference_checkpoints or (), key=lambda item: item.step) if item.step == reference_steps), None)
    if exact_reference is None:
        reason = f"No checkpoint at the exact {reference_steps:,}-step reference horizon was supplied."
        if reference_checkpoints:
            reason = f"The supplied reference checkpoints do not contain the exact {reference_steps:,}-step horizon."
        return ConvergenceReport(
            ConvergenceStatus.REFERENCE_NOT_AVAILABLE, reference_available=False,
            reference_agreement_status=ReferenceAgreementStatus.REFERENCE_NOT_AVAILABLE,
            diagnostics=(reason,), **base,
        )
    if not short or stable == ConvergenceStatus.INSUFFICIENT_CHECKPOINTS:
        return ConvergenceReport(
            ConvergenceStatus.INSUFFICIENT_CHECKPOINTS, reference_available=True,
            reference_agreement_status=ReferenceAgreementStatus.REFERENCE_NOT_AVAILABLE,
            diagnostics=("Short-horizon internal stability needs at least three metric-bearing checkpoints.",), **base,
        )
    short_latest = short[-1]
    required_metrics = policy.metrics(short_latest, exact_reference)
    missing_metrics = tuple(sorted(set(required_metrics) - (set(short_latest.metrics) & set(exact_reference.metrics))))
    comparisons = compare_to_reference(short_latest, exact_reference, reference_steps=reference_steps, tolerance_policy=policy)
    if missing_metrics:
        agreement = ReferenceAgreementStatus.REFERENCE_METRIC_MISSING
        diagnostics = (f"Required reference metrics are missing from one side: {', '.join(missing_metrics)}.",)
    elif not comparisons or any(comparison.tolerance is None for comparison in comparisons):
        agreement = ReferenceAgreementStatus.REFERENCE_OUTSIDE_TOLERANCE
        diagnostics = ("No defensible tolerance was supplied for every required metric; no validation claim is allowed.",)
    elif not all(comparison.within_tolerance for comparison in comparisons):
        agreement = ReferenceAgreementStatus.REFERENCE_OUTSIDE_TOLERANCE
        diagnostics = ("Short-horizon metrics fall outside the explicit reference tolerance policy.",)
    else:
        agreement = ReferenceAgreementStatus.REFERENCE_WITHIN_TOLERANCE
        diagnostics = ()
    if stable == ConvergenceStatus.NOT_STABLE:
        status = ConvergenceStatus.NOT_STABLE
    elif agreement == ReferenceAgreementStatus.REFERENCE_METRIC_MISSING:
        status = ConvergenceStatus.REFERENCE_METRIC_MISSING
    elif agreement != ReferenceAgreementStatus.REFERENCE_WITHIN_TOLERANCE:
        status = ConvergenceStatus.REFERENCE_OUTSIDE_TOLERANCE
    else:
        status = ConvergenceStatus.VALIDATED_AGAINST_REFERENCE
    return ConvergenceReport(
        status, reference_available=True, comparisons=comparisons, required_metrics=required_metrics,
        missing_metrics=missing_metrics, reference_agreement_status=agreement, diagnostics=diagnostics, **base,
    )
