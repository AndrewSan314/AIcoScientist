from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Mapping, Sequence

from .config import FidelityMode, REFERENCE_SLURRY_STEPS


class ConvergenceStatus(StrEnum):
    INSUFFICIENT_CHECKPOINTS = "INSUFFICIENT_CHECKPOINTS"
    INSUFFICIENT_STEP_SPAN = "INSUFFICIENT_STEP_SPAN"
    STABILITY_OBSERVED = "STABILITY_OBSERVED"
    NOT_STABLE = "NOT_STABLE"
    REFERENCE_NOT_AVAILABLE = "REFERENCE_NOT_AVAILABLE"
    REFERENCE_METRIC_MISSING = "REFERENCE_METRIC_MISSING"
    REFERENCE_OUTSIDE_TOLERANCE = "REFERENCE_OUTSIDE_TOLERANCE"
    VALIDATED_AGAINST_REFERENCE = "VALIDATED_AGAINST_REFERENCE"
    INCOMPATIBLE_REFERENCE_EVIDENCE = "INCOMPATIBLE_REFERENCE_EVIDENCE"


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
class StabilityPolicy:
    """Diagnostic-only stability policy; defaults are software safeguards, not validation claims."""

    minimum_checkpoints: int = 10
    minimum_step_span: int = 0
    relative_tolerance: float = 0.01
    required_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.minimum_checkpoints) < 1:
            raise ValueError("minimum_checkpoints must be positive")
        if int(self.minimum_step_span) < 0:
            raise ValueError("minimum_step_span must be non-negative")
        if not math.isfinite(float(self.relative_tolerance)) or self.relative_tolerance < 0:
            raise ValueError("relative_tolerance must be finite and non-negative")
        if len(set(self.required_metrics)) != len(self.required_metrics):
            raise ValueError("required stability metrics must be unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_checkpoints": self.minimum_checkpoints,
            "minimum_step_span": self.minimum_step_span,
            "relative_tolerance": self.relative_tolerance,
            "required_metrics": list(self.required_metrics),
        }


@dataclass(frozen=True)
class Checkpoint:
    """A dynamics-relative checkpoint; raw LAMMPS step is audit metadata."""
    step: int
    metrics: Mapping[str, float]
    wall_seconds: float | None = None
    stage: str | None = None
    source_log: str | None = None
    raw_step: int | None = None
    dynamics_step: int | None = None
    phase: str = "dynamics"

    def __post_init__(self) -> None:
        if self.step < 0 or (self.raw_step is not None and self.raw_step < 0) or (self.dynamics_step is not None and self.dynamics_step < 0):
            raise ValueError("checkpoint steps and metrics must be finite and non-negative")
        if self.phase not in {"minimization", "dynamics"}:
            raise ValueError("checkpoint phase must be minimization or dynamics")
        if any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("checkpoint metrics must be finite")
        if self.wall_seconds is not None and (not math.isfinite(float(self.wall_seconds)) or self.wall_seconds < 0):
            raise ValueError("checkpoint wall time must be finite and non-negative")


@dataclass(frozen=True)
class ConvergenceRunEvidence:
    """Manifest evidence that binds a run to one exact physics case."""

    run_id: str
    recipe_fingerprint: str
    pinned_commit: str
    pinned_source_tree_hash: str
    physics_config_fingerprint: str
    fidelity_mode: FidelityMode
    fidelity_identity: str
    requested_dynamic_steps: int
    dump_interval_steps: int
    checkpoints: tuple[Checkpoint, ...]
    simulation_manifest_hash: str
    successful: bool

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
            self.run_id, self.recipe_fingerprint, self.pinned_commit, self.pinned_source_tree_hash,
            self.physics_config_fingerprint, self.fidelity_identity, self.simulation_manifest_hash,
        )) or self.requested_dynamic_steps <= 0 or self.dump_interval_steps <= 0:
            raise ValueError("convergence evidence requires complete run and physics identities")
        mode = FidelityMode(self.fidelity_mode)
        object.__setattr__(self, "fidelity_mode", mode)
        if any(item.phase != "dynamics" for item in self.checkpoints):
            raise ValueError("convergence evidence checkpoints must be dynamics phase")
        if mode == FidelityMode.REFERENCE and self.requested_dynamic_steps != REFERENCE_SLURRY_STEPS:
            raise ValueError("reference evidence requires exactly 20,000,000 dynamics-relative steps")
        if mode == FidelityMode.SHORT_HORIZON and self.requested_dynamic_steps >= REFERENCE_SLURRY_STEPS:
            raise ValueError("short-horizon evidence cannot claim the reference horizon")

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any], *, simulation_manifest_hash: str | None = None) -> "ConvergenceRunEvidence":
        return cls(
            run_id=str(manifest.get("run_id", "")), recipe_fingerprint=str(manifest.get("recipe_fingerprint", "")),
            pinned_commit=str(manifest.get("checked_out_commit", manifest.get("pinned_upstream_commit", ""))),
            pinned_source_tree_hash=str(manifest.get("source_tree_hash", "")),
            physics_config_fingerprint=str(manifest.get("physics_config_fingerprint", "")),
            fidelity_mode=FidelityMode(str(manifest.get("fidelity_mode", ""))),
            fidelity_identity=str(manifest.get("fidelity_identity", "")),
            requested_dynamic_steps=int(manifest.get("requested_slurry_steps", 0)),
            dump_interval_steps=int(manifest.get("dump_interval_steps", 0)),
            checkpoints=checkpoints_from_manifest(manifest),
            simulation_manifest_hash=simulation_manifest_hash or str(manifest.get("simulation_manifest_hash", "")),
            successful=str(manifest.get("status", "")) == "Success",
        )

    def compatibility_with(self, reference: "ConvergenceRunEvidence") -> tuple[bool, str]:
        if reference.fidelity_mode != FidelityMode.REFERENCE or reference.requested_dynamic_steps != REFERENCE_SLURRY_STEPS:
            return False, "reference evidence is not a successful exact 20,000,000-step REFERENCE run"
        if not reference.successful:
            return False, "reference evidence is not successful"
        if self.fidelity_mode != FidelityMode.SHORT_HORIZON:
            return False, "short evidence must remain SHORT_HORIZON"
        fields = (
            ("recipe_fingerprint", self.recipe_fingerprint, reference.recipe_fingerprint),
            ("pinned_commit", self.pinned_commit, reference.pinned_commit),
            ("pinned_source_tree_hash", self.pinned_source_tree_hash, reference.pinned_source_tree_hash),
            ("physics_config_fingerprint", self.physics_config_fingerprint, reference.physics_config_fingerprint),
        )
        for name, short_value, reference_value in fields:
            if short_value != reference_value:
                return False, f"reference evidence {name} differs"
        if not self.successful:
            return False, "short evidence is not successful"
        if not any(item.step == REFERENCE_SLURRY_STEPS for item in reference.checkpoints):
            return False, "reference evidence lacks an exact 20,000,000 dynamics checkpoint"
        return True, ""


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
    stability_policy: Mapping[str, Any] = field(default_factory=dict)


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
        if not isinstance(item, Mapping):
            raise ValueError("ARTISTIC checkpoint entries need a step")
        phase = str(item.get("phase", "dynamics"))
        dynamics_step = item.get("dynamics_step", item.get("step"))
        if phase == "minimization" or dynamics_step is None:
            continue
        if "step" not in item and "dynamics_step" not in item:
            raise ValueError("ARTISTIC checkpoint entries need a dynamics-relative step")
        metrics = item.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("ARTISTIC checkpoint metrics must be a mapping")
        result.append(Checkpoint(
            int(dynamics_step), {str(name): float(value) for name, value in metrics.items()}, item.get("wall_seconds"),
            item.get("stage"), item.get("source_log"), item.get("raw_step"), int(dynamics_step), phase,
        ))
    return tuple(sorted(result, key=lambda item: item.step))


def stability_status(
    checkpoints: Sequence[Checkpoint], *, policy: StabilityPolicy | None = None,
    tolerance: float | None = None, minimum_checkpoints: int | None = None,
    minimum_step_span: int | None = None,
) -> ConvergenceStatus:
    policy = policy or StabilityPolicy()
    overrides = {}
    if tolerance is not None:
        overrides["relative_tolerance"] = tolerance
    if minimum_checkpoints is not None:
        overrides["minimum_checkpoints"] = minimum_checkpoints
    if minimum_step_span is not None:
        overrides["minimum_step_span"] = minimum_step_span
    if overrides:
        policy = replace(policy, **overrides)
    ordered = tuple(sorted(checkpoints, key=lambda item: item.step))
    if len(ordered) < policy.minimum_checkpoints:
        return ConvergenceStatus.INSUFFICIENT_CHECKPOINTS
    if ordered[-1].step - ordered[0].step < policy.minimum_step_span:
        return ConvergenceStatus.INSUFFICIENT_STEP_SPAN
    recent = ordered[-policy.minimum_checkpoints:]
    names = policy.required_metrics or tuple(sorted(recent[-1].metrics))
    if not names:
        return ConvergenceStatus.INSUFFICIENT_CHECKPOINTS
    for name in names:
        values = [float(item.metrics[name]) for item in recent if name in item.metrics]
        if len(values) != len(recent):
            return ConvergenceStatus.INSUFFICIENT_CHECKPOINTS
        scale = max(abs(values[-1]), 1e-12)
        if max(abs(value - values[-1]) for value in values[:-1]) / scale > policy.relative_tolerance:
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
    if reference_steps != REFERENCE_SLURRY_STEPS:
        raise ValueError(f"reference horizon is fixed at exactly {REFERENCE_SLURRY_STEPS:,} steps")
    reference_dynamic_step = reference.dynamics_step if reference.dynamics_step is not None else reference.step
    if reference.phase != "dynamics" or reference_dynamic_step != reference_steps:
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
    short_evidence: ConvergenceRunEvidence | None = None,
    reference_evidence: ConvergenceRunEvidence | None = None,
    requested_steps: int | None = None,
    reference_steps: int = REFERENCE_SLURRY_STEPS,
    tolerance: float | None = None,
    tolerance_policy: MetricTolerancePolicy | None = None,
    stability_tolerance: float | None = None,
    stability_policy: StabilityPolicy | None = None,
) -> ConvergenceReport:
    short = tuple(sorted(short_evidence.checkpoints if short_evidence is not None else checkpoints, key=lambda item: item.step))
    if short_evidence is not None and requested_steps is None:
        requested_steps = short_evidence.requested_dynamic_steps
    available_metrics = tuple(sorted({name for item in short for name in item.metrics}))
    if tolerance_policy is None and tolerance is not None:
        tolerance_policy = MetricTolerancePolicy(default_relative_tolerance=tolerance)
    metric_policy = tolerance_policy or MetricTolerancePolicy()
    if stability_policy is None:
        stability_policy = StabilityPolicy(relative_tolerance=stability_tolerance if stability_tolerance is not None else (tolerance if tolerance is not None else 0.01))
    elif stability_tolerance is not None:
        stability_policy = replace(stability_policy, relative_tolerance=stability_tolerance)
    stable = stability_status(short, policy=stability_policy)
    base = {
        "checkpoint_count": len(short), "requested_steps": requested_steps, "reference_steps": reference_steps,
        "available_metrics": available_metrics, "stability_status": stable,
        "tolerance_policy": metric_policy.as_dict(), "stability_policy": stability_policy.as_dict(),
    }
    if reference_steps != REFERENCE_SLURRY_STEPS:
        return ConvergenceReport(
            ConvergenceStatus.REFERENCE_NOT_AVAILABLE, reference_available=False,
            reference_agreement_status=ReferenceAgreementStatus.REFERENCE_NOT_AVAILABLE,
            diagnostics=(f"Reference validation is fixed to exactly {REFERENCE_SLURRY_STEPS:,} steps.",), **base,
        )
    if short_evidence is not None or reference_evidence is not None:
        if short_evidence is None or reference_evidence is None:
            return ConvergenceReport(
                ConvergenceStatus.INCOMPATIBLE_REFERENCE_EVIDENCE, reference_available=False,
                reference_agreement_status=ReferenceAgreementStatus.REFERENCE_NOT_AVAILABLE,
                diagnostics=("Both short-horizon and reference ConvergenceRunEvidence are required.",), **base,
            )
        compatible, reason = short_evidence.compatibility_with(reference_evidence)
        if not compatible:
            return ConvergenceReport(
                ConvergenceStatus.INCOMPATIBLE_REFERENCE_EVIDENCE, reference_available=False,
                reference_agreement_status=ReferenceAgreementStatus.REFERENCE_NOT_AVAILABLE,
                diagnostics=(reason,), **base,
            )
        evidence_checkpoints = reference_evidence.checkpoints
    elif reference_checkpoints is not None:
        return ConvergenceReport(
            ConvergenceStatus.INCOMPATIBLE_REFERENCE_EVIDENCE, reference_available=False,
            reference_agreement_status=ReferenceAgreementStatus.REFERENCE_NOT_AVAILABLE,
            diagnostics=("Reference comparison requires typed ConvergenceRunEvidence; raw checkpoints are insufficient.",), **base,
        )
    else:
        evidence_checkpoints = ()
    exact_reference = next((item for item in sorted(evidence_checkpoints, key=lambda item: item.step) if item.step == reference_steps), None)
    if exact_reference is None:
        reason = f"No checkpoint at the exact {reference_steps:,}-step reference horizon was supplied."
        if evidence_checkpoints:
            reason = f"The supplied reference checkpoints do not contain the exact {reference_steps:,}-step horizon."
        return ConvergenceReport(
            ConvergenceStatus.REFERENCE_NOT_AVAILABLE, reference_available=False,
            reference_agreement_status=ReferenceAgreementStatus.REFERENCE_NOT_AVAILABLE,
            diagnostics=(reason,), **base,
        )
    if not short or stable in {ConvergenceStatus.INSUFFICIENT_CHECKPOINTS, ConvergenceStatus.INSUFFICIENT_STEP_SPAN}:
        return ConvergenceReport(
            stable, reference_available=True,
            reference_agreement_status=ReferenceAgreementStatus.REFERENCE_NOT_AVAILABLE,
            diagnostics=(f"Short-horizon stability policy returned {stable.value}.",), **base,
        )
    short_latest = short[-1]
    required_metrics = metric_policy.metrics(short_latest, exact_reference)
    missing_metrics = tuple(sorted(set(required_metrics) - (set(short_latest.metrics) & set(exact_reference.metrics))))
    comparisons = compare_to_reference(short_latest, exact_reference, reference_steps=reference_steps, tolerance_policy=metric_policy)
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
