from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class ExperimentStage(str, Enum):
    """Lifecycle stages of a physical or simulated scientific experiment."""

    PROPOSED = "PROPOSED"
    SCHEDULED = "SCHEDULED"
    EXECUTED = "EXECUTED"
    CHARACTERIZED = "CHARACTERIZED"
    PERFORMANCE_MEASURED = "PERFORMANCE_MEASURED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Valid stage transition graph
VALID_STAGE_TRANSITIONS: dict[ExperimentStage, set[ExperimentStage]] = {
    ExperimentStage.PROPOSED: {
        ExperimentStage.SCHEDULED,
        ExperimentStage.EXECUTED,
        ExperimentStage.FAILED,
        ExperimentStage.CANCELLED,
    },
    ExperimentStage.SCHEDULED: {
        ExperimentStage.EXECUTED,
        ExperimentStage.FAILED,
        ExperimentStage.CANCELLED,
    },
    ExperimentStage.EXECUTED: {
        ExperimentStage.CHARACTERIZED,
        ExperimentStage.PERFORMANCE_MEASURED,
        ExperimentStage.COMPLETED,
        ExperimentStage.FAILED,
        ExperimentStage.CANCELLED,
    },
    ExperimentStage.CHARACTERIZED: {
        ExperimentStage.PERFORMANCE_MEASURED,
        ExperimentStage.COMPLETED,
        ExperimentStage.FAILED,
        ExperimentStage.CANCELLED,
    },
    ExperimentStage.PERFORMANCE_MEASURED: {
        ExperimentStage.CHARACTERIZED,
        ExperimentStage.COMPLETED,
        ExperimentStage.FAILED,
        ExperimentStage.CANCELLED,
    },
    ExperimentStage.COMPLETED: set(),
    ExperimentStage.FAILED: set(),
    ExperimentStage.CANCELLED: set(),
}


@dataclass
class ScientificExperimentRecord:
    """Domain-generic record representing an experiment throughout its lifecycle."""

    experiment_id: str
    candidate_id: str
    dataset_name: str
    stage: ExperimentStage = ExperimentStage.PROPOSED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # 1. Pre-experiment known features & controllable candidate variables
    pre_experiment_features: dict[str, Any] = field(default_factory=dict)
    candidate_variables: dict[str, Any] = field(default_factory=dict)

    # 2. Post-experiment physical/spectroscopic characterization (Structure/Morphology)
    characterization: dict[str, Any] = field(default_factory=dict)

    # 3. Experimental performance outcomes (Properties/Metrics)
    performance: dict[str, Any] = field(default_factory=dict)

    # Metadata & audit fields
    measurement_uncertainty: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)
    batch_id: str | None = None
    replicate_id: str | None = None
    proposal_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.stage, str):
            self.stage = ExperimentStage(self.stage)
        if not self.candidate_variables and self.pre_experiment_features:
            self.candidate_variables = dict(self.pre_experiment_features)

    def transition_to(
        self,
        new_stage: ExperimentStage | str,
        characterization: Mapping[str, Any] | None = None,
        performance: Mapping[str, Any] | None = None,
        measurement_uncertainty: Mapping[str, float] | None = None,
        quality_flags: list[str] | None = None,
        failure_reason: str | None = None,
    ) -> ScientificExperimentRecord:
        """Transitions record to a new lifecycle stage with validation."""
        target_stage = ExperimentStage(new_stage) if isinstance(new_stage, str) else new_stage
        valid_targets = VALID_STAGE_TRANSITIONS.get(self.stage, set())
        if target_stage not in valid_targets:
            raise ValueError(
                f"Invalid stage transition from {self.stage.value} to {target_stage.value}. "
                f"Valid targets are: {[s.value for s in valid_targets]}"
            )

        self.stage = target_stage
        self.updated_at = datetime.now(timezone.utc).isoformat()

        if characterization:
            self.characterization.update(dict(characterization))
        if performance:
            self.performance.update(dict(performance))
        if measurement_uncertainty:
            self.measurement_uncertainty.update(dict(measurement_uncertainty))
        if quality_flags:
            self.quality_flags.extend(quality_flags)
        if failure_reason:
            self.failure_reason = failure_reason

        return self

    def copy(self) -> ScientificExperimentRecord:
        """Creates a deep copy of this record for prospective transition validation."""
        return ScientificExperimentRecord(
            experiment_id=self.experiment_id,
            candidate_id=self.candidate_id,
            dataset_name=self.dataset_name,
            stage=self.stage,
            created_at=self.created_at,
            updated_at=self.updated_at,
            pre_experiment_features=dict(self.pre_experiment_features),
            candidate_variables=dict(self.candidate_variables),
            characterization=dict(self.characterization),
            performance=dict(self.performance),
            measurement_uncertainty=dict(self.measurement_uncertainty),
            constraints=dict(self.constraints),
            quality_flags=list(self.quality_flags),
            batch_id=self.batch_id,
            replicate_id=self.replicate_id,
            proposal_metadata=copy.deepcopy(self.proposal_metadata),
            provenance=copy.deepcopy(self.provenance),
            failure_reason=self.failure_reason,
        )

    def is_terminal(self) -> bool:
        return self.stage in {ExperimentStage.COMPLETED, ExperimentStage.FAILED, ExperimentStage.CANCELLED}

    def has_characterization(self) -> bool:
        return bool(self.characterization)

    def has_performance(self) -> bool:
        return bool(self.performance)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stage"] = self.stage.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScientificExperimentRecord:
        d = dict(data)
        if "stage" in d and isinstance(d["stage"], str):
            d["stage"] = ExperimentStage(d["stage"])
        return cls(**d)
