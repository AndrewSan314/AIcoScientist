from __future__ import annotations

from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if (
            not self.modality_id.strip() or self.cost < 0 or self.latency_seconds < 0
            or not math.isfinite(float(self.cost)) or not math.isfinite(float(self.latency_seconds))
        ):
            raise ValueError("evidence option needs an id and non-negative cost/latency")


@dataclass(frozen=True)
class EvidenceReplayResult:
    selected_modality_id: str | None
    visible_observations: Mapping[str, Any]
    acquired_cost: float
    acquired_latency_seconds: float


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
