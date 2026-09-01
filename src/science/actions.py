from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


class ExperimentActionType(str, Enum):
    """Supported scientific experimental measurement action types (Au-Ir-Rh compatibility)."""

    XRD = "XRD"
    PROPERTY = "PROPERTY"


ActionType = str | ExperimentActionType


def normalize_action_type(action_type: ActionType) -> str:
    """Normalizes an action type (enum or string) to a canonical string identifier."""
    if isinstance(action_type, Enum):
        return str(action_type.value)
    return str(action_type)


@dataclass(frozen=True)
class ScientificAction:
    """A concrete scientific measurement action requested for a specific physical candidate."""

    action_id: str
    candidate_id: str
    action_type: ActionType
    estimated_cost: float = 1.0
    requested_at_step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "candidate_id": self.candidate_id,
            "action_type": normalize_action_type(self.action_type),
            "estimated_cost": self.estimated_cost,
            "requested_at_step": self.requested_at_step,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScientificAction:
        raw_act = data["action_type"]
        try:
            act_type: ActionType = ExperimentActionType(raw_act)
        except ValueError:
            act_type = str(raw_act)
        return cls(
            action_id=str(data["action_id"]),
            candidate_id=str(data["candidate_id"]),
            action_type=act_type,
            estimated_cost=float(data.get("estimated_cost", 1.0)),
            requested_at_step=int(data.get("requested_at_step", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ExperimentOutcome:
    """The revealed observation returned by the oracle or domain connector."""

    action_id: str
    candidate_id: str
    action_type: ActionType
    revealed_data: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    oracle_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    canonical_observation: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "candidate_id": self.candidate_id,
            "action_type": normalize_action_type(self.action_type),
            "revealed_data": dict(self.revealed_data),
            "provenance": dict(self.provenance),
            "oracle_timestamp": self.oracle_timestamp,
            "canonical_observation": self.canonical_observation,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentOutcome:
        act_type_raw = data.get("action_type", "XRD")
        try:
            act_type = ExperimentActionType(str(act_type_raw).upper())
        except ValueError:
            act_type = str(act_type_raw)
        return cls(
            action_id=str(data["action_id"]),
            candidate_id=str(data["candidate_id"]),
            action_type=act_type,
            revealed_data=dict(data["revealed_data"]),
            provenance=dict(data.get("provenance", {})),
            oracle_timestamp=str(data.get("oracle_timestamp", datetime.now(timezone.utc).isoformat())),
            canonical_observation=data.get("canonical_observation"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class CounterfactualAlternative:
    """A scored alternative action evaluated during next-best-experiment decision making."""

    candidate_id: str
    action_type: ActionType
    total_value: float
    scientific_information_value: float
    discovery_value: float
    cost_penalty: float
    hypothesis_id: str
    contrastive_rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_type": normalize_action_type(self.action_type),
            "total_value": self.total_value,
            "scientific_information_value": self.scientific_information_value,
            "discovery_value": self.discovery_value,
            "cost_penalty": self.cost_penalty,
            "hypothesis_id": self.hypothesis_id,
            "contrastive_rationale": self.contrastive_rationale,
        }


@dataclass(frozen=True)
class ActionRecommendation:
    """The structured recommendation output from the next-best-experiment policy."""

    action: ScientificAction
    total_value: float
    scientific_information_value: float
    discovery_value: float
    cost_penalty: float
    hypothesis_id: str
    rationale: str
    falsification_criterion: str
    supporting_evidence: list[str] = field(default_factory=list)
    uncertainty_summary: dict[str, float] = field(default_factory=dict)
    alternatives: list[CounterfactualAlternative] = field(default_factory=list)
    domain_id: str | None = None
    modality_name: str | None = None
    objective_name: str | None = None
    objective_units: str | None = None
    raw_hig: float | None = None
    expected_posterior_entropy: float | None = None
    current_beliefs: dict[str, float] = field(default_factory=dict)
    optimizer_status: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "total_value": self.total_value,
            "scientific_information_value": self.scientific_information_value,
            "discovery_value": self.discovery_value,
            "cost_penalty": self.cost_penalty,
            "hypothesis_id": self.hypothesis_id,
            "rationale": self.rationale,
            "falsification_criterion": self.falsification_criterion,
            "supporting_evidence": list(self.supporting_evidence),
            "uncertainty_summary": dict(self.uncertainty_summary),
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "domain_id": self.domain_id,
            "modality_name": self.modality_name,
            "objective_name": self.objective_name,
            "objective_units": self.objective_units,
            "raw_hig": self.raw_hig,
            "expected_posterior_entropy": self.expected_posterior_entropy,
            "current_beliefs": dict(self.current_beliefs),
            "optimizer_status": dict(self.optimizer_status),
            "metadata": dict(self.metadata),
        }
