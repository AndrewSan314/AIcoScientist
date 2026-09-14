from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Callable, Mapping

from .information_horizon import HorizonView
from .stages import ProcessStage, stage_precedes


@dataclass(frozen=True)
class EvidenceOption:
    modality_id: str
    stage: ProcessStage
    available: bool
    cost: float
    latency_seconds: float
    provenance: Any = None
    reveals_source_observation: bool = False
    evidence_kind: str = "PHYSICAL_HISTORICAL"
    fidelity: str = "SOURCE"
    required_context: Mapping[str, Any] | None = None
    observed: bool = False
    compatible: bool = True

    def __post_init__(self) -> None:
        if (
            not self.modality_id.strip() or not self.evidence_kind.strip() or not self.fidelity.strip() or self.cost < 0 or self.latency_seconds < 0
            or not math.isfinite(float(self.cost)) or not math.isfinite(float(self.latency_seconds))
        ):
            raise ValueError("evidence option needs an id and non-negative cost/latency")


@dataclass(frozen=True)
class EvidenceReplayResult:
    selected_modality_id: str | None
    visible_observations: Mapping[str, Any]
    acquired_cost: float
    acquired_latency_seconds: float


class EvidenceAcquisitionPolicy(str, Enum):
    RANDOM_LEGAL_EVIDENCE = "RANDOM_LEGAL_EVIDENCE"
    CHEAPEST_LEGAL_EVIDENCE = "CHEAPEST_LEGAL_EVIDENCE"
    MAX_PREDICTIVE_VARIANCE_REDUCTION = "MAX_PREDICTIVE_VARIANCE_REDUCTION"
    EXPECTED_INFORMATION_VALUE_PER_COST = "EXPECTED_INFORMATION_VALUE_PER_COST"
    NO_ADDITIONAL_EVIDENCE = "NO_ADDITIONAL_EVIDENCE"


@dataclass(frozen=True)
class EvidencePolicyReplayEvaluation:
    policy: EvidenceAcquisitionPolicy
    selected_modality_id: str | None
    evidence_cost: float
    evidence_latency_seconds: float
    final_decision_regret_before: float | None
    final_decision_regret_after: float | None
    prediction_error_before: float | None
    prediction_error_after: float | None
    predictive_uncertainty_before: float | None
    predictive_uncertainty_after: float | None
    utility_per_cost: float | None


def _choose_scored_evidence(options: tuple[EvidenceOption, ...], estimates: Mapping[str, float], *, divide_by_cost: bool) -> EvidenceOption | None:
    option_ids = {option.modality_id for option in options}
    if len(option_ids) != len(options) or set(estimates) != option_ids:
        raise ValueError("estimated evidence scores must be declared for exactly the available evidence options")
    if any(not option.available or option.observed or not option.compatible for option in options):
        raise ValueError("evidence policies reject unavailable, observed, or incompatible options")
    scored: list[tuple[float, str, EvidenceOption]] = []
    for option in options:
        estimate = float(estimates[option.modality_id])
        if not math.isfinite(estimate) or estimate < 0:
            raise ValueError("estimated evidence scores must be finite and non-negative")
        if estimate:
            scored.append((estimate / (option.cost or 1.0) if divide_by_cost else estimate, option.modality_id, option))
    return max(scored, default=(0.0, "", None), key=lambda item: (item[0], item[1]))[2]


def choose_evidence(
    options: tuple[EvidenceOption, ...],
    policy: EvidenceAcquisitionPolicy,
    *,
    estimated_variance_reduction: Mapping[str, float] | None = None,
    estimated_decision_utility: Mapping[str, float] | None = None,
    seed: int = 0,
) -> EvidenceOption | None:
    """Choose from declared pre-reveal estimates; this function never accesses withheld evidence."""
    if policy is EvidenceAcquisitionPolicy.NO_ADDITIONAL_EVIDENCE:
        return None
    if any(not option.available or option.observed or not option.compatible for option in options):
        raise ValueError("evidence policies reject unavailable, observed, or incompatible options")
    if policy is EvidenceAcquisitionPolicy.RANDOM_LEGAL_EVIDENCE:
        if len({option.modality_id for option in options}) != len(options):
            raise ValueError("evidence option ids must be unique")
        return sorted(options, key=lambda option: option.modality_id)[seed % len(options)] if options else None
    if policy is EvidenceAcquisitionPolicy.CHEAPEST_LEGAL_EVIDENCE:
        return min(options, default=None, key=lambda option: (option.cost, option.latency_seconds, option.modality_id))
    if policy is EvidenceAcquisitionPolicy.MAX_PREDICTIVE_VARIANCE_REDUCTION:
        if estimated_variance_reduction is None:
            raise ValueError("variance-reduction policy requires declared variance estimates")
        return _choose_scored_evidence(options, estimated_variance_reduction, divide_by_cost=False)
    if policy is EvidenceAcquisitionPolicy.EXPECTED_INFORMATION_VALUE_PER_COST:
        if estimated_decision_utility is None:
            raise ValueError("EVI-per-cost policy requires declared decision-utility estimates")
        return _choose_scored_evidence(options, estimated_decision_utility, divide_by_cost=True)
    raise ValueError(f"unsupported evidence acquisition policy: {policy!r}")


def _optional_metric(metrics: Mapping[str, float], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"evidence replay metric {name!r} must be finite")
    return value


def evaluate_blinded_evidence_policy(
    visible_observations: Mapping[str, Any],
    withheld_observations: Mapping[str, Any],
    options: list[EvidenceOption],
    *,
    policy: EvidenceAcquisitionPolicy,
    decision_stage: ProcessStage | HorizonView,
    evaluate_visible: Callable[[Mapping[str, Any]], Mapping[str, float]],
    estimated_variance_reduction: Mapping[str, float] | None = None,
    estimated_decision_utility: Mapping[str, float] | None = None,
    seed: int = 0,
) -> EvidencePolicyReplayEvaluation:
    """Evaluate an evidence policy through the same select-before-reveal firewall."""
    before = dict(evaluate_visible(dict(visible_observations)))
    replay = replay_blinded_evidence(
        visible_observations,
        withheld_observations,
        options,
        lambda _visible, choices: choose_evidence(choices, policy, estimated_variance_reduction=estimated_variance_reduction, estimated_decision_utility=estimated_decision_utility, seed=seed),
        decision_stage=decision_stage,
    )
    after = dict(evaluate_visible(replay.visible_observations))
    regret_before, regret_after = _optional_metric(before, "final_decision_regret"), _optional_metric(after, "final_decision_regret")
    utility_per_cost = (regret_before - regret_after) / replay.acquired_cost if replay.acquired_cost and regret_before is not None and regret_after is not None else None
    return EvidencePolicyReplayEvaluation(
        policy, replay.selected_modality_id, replay.acquired_cost, replay.acquired_latency_seconds,
        regret_before, regret_after,
        _optional_metric(before, "prediction_error"), _optional_metric(after, "prediction_error"),
        _optional_metric(before, "predictive_uncertainty"), _optional_metric(after, "predictive_uncertainty"),
        utility_per_cost,
    )


def choose_cost_aware_evidence(
    options: tuple[EvidenceOption, ...],
    estimated_utility: Mapping[str, float],
) -> EvidenceOption | None:
    """Select from declared, pre-reveal utility estimates; never inspect withheld data."""
    return choose_evidence(
        options,
        EvidenceAcquisitionPolicy.EXPECTED_INFORMATION_VALUE_PER_COST,
        estimated_decision_utility=estimated_utility,
    )


def replay_blinded_evidence(
    visible_observations: Mapping[str, Any],
    withheld_observations: Mapping[str, Any],
    options: list[EvidenceOption],
    choose: Callable[[Mapping[str, Any], tuple[EvidenceOption, ...]], EvidenceOption | None],
    *,
    decision_stage: ProcessStage | HorizonView,
) -> EvidenceReplayResult:
    """Select first, then reveal only a source-backed observation legal at the horizon."""
    horizon_stage = decision_stage.decision_stage if isinstance(decision_stage, HorizonView) else decision_stage
    if not isinstance(horizon_stage, ProcessStage):
        raise TypeError("decision_stage must be a ProcessStage or HorizonView")
    by_id = {option.modality_id: option for option in options}
    if len(by_id) != len(options):
        raise ValueError("evidence option ids must be unique")
    illegal = [
        option.modality_id for option in options
        if not option.available or not option.reveals_source_observation or not stage_precedes(option.stage, horizon_stage)
    ]
    if illegal:
        raise ValueError(f"evidence options are unavailable, non-source-backed, or beyond {horizon_stage.value}: {illegal}")
    selected = choose(dict(visible_observations), tuple(options))
    if selected is None:
        return EvidenceReplayResult(None, dict(visible_observations), 0.0, 0.0)
    if by_id.get(selected.modality_id) != selected or not selected.available or not selected.reveals_source_observation:
        raise ValueError("policy selected an unavailable or non-source-backed evidence option")
    if selected.modality_id not in withheld_observations:
        raise ValueError(f"selected evidence is not present in the blinded source set: {selected.modality_id}")
    revealed = dict(visible_observations)
    revealed[selected.modality_id] = withheld_observations[selected.modality_id]
    return EvidenceReplayResult(selected.modality_id, revealed, selected.cost, selected.latency_seconds)
