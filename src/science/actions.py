from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


class ExperimentActionType(str, Enum):
    """Supported scientific experimental measurement action types."""

    XRD = "XRD"
    PROPERTY = "PROPERTY"


@dataclass(frozen=True)
class ScientificAction:
    """A concrete scientific measurement action requested for a specific physical candidate."""

    action_id: str
    candidate_id: str
    action_type: ExperimentActionType
    estimated_cost: float = 1.0
    requested_at_step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "candidate_id": self.candidate_id,
            "action_type": self.action_type.value,
            "estimated_cost": self.estimated_cost,
            "requested_at_step": self.requested_at_step,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScientificAction:
        return cls(
            action_id=str(data["action_id"]),
            candidate_id=str(data["candidate_id"]),
            action_type=ExperimentActionType(data["action_type"]),
            estimated_cost=float(data.get("estimated_cost", 1.0)),
            requested_at_step=int(data.get("requested_at_step", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ExperimentOutcome:
    """The ground-truth experimental outcome revealed by executing an action via the offline oracle."""

    action_id: str
    candidate_id: str
    action_type: ExperimentActionType
    revealed_data: dict[str, Any]  # e.g., {'k0': 0.0142} or {'two_theta': [...], 'intensity': [...]}
    provenance: dict[str, Any] = field(default_factory=dict)
    oracle_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "candidate_id": self.candidate_id,
            "action_type": self.action_type.value,
            "revealed_data": dict(self.revealed_data),
            "provenance": dict(self.provenance),
            "oracle_timestamp": self.oracle_timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentOutcome:
        return cls(
            action_id=str(data["action_id"]),
            candidate_id=str(data["candidate_id"]),
            action_type=ExperimentActionType(data["action_type"]),
            revealed_data=dict(data["revealed_data"]),
            provenance=dict(data.get("provenance", {})),
            oracle_timestamp=str(data.get("oracle_timestamp", datetime.now(timezone.utc).isoformat())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class CounterfactualAlternative:
    """A scored alternative action evaluated during next-best-experiment decision making."""

    candidate_id: str
    action_type: ExperimentActionType
    total_value: float
    scientific_information_value: float
    discovery_value: float
    cost_penalty: float
    hypothesis_id: str
    contrastive_rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_type": self.action_type.value,
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
        }
