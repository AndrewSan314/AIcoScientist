from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from src.science.actions import ExperimentOutcome, ScientificAction
    from src.science.hypothesis_models import ScientificHypothesisModel


class ObjectiveDirection(str, Enum):
    """Direction for an optimization / discovery objective."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class ObjectiveDefinition:
    """Domain-agnostic discovery or optimization objective definition."""

    name: str
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    units: str | None = None
    target_col: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction.value,
            "units": self.units,
            "target_col": self.target_col or self.name,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObjectiveDefinition:
        raw_dir = str(data.get("direction", "maximize")).strip().lower()
        direction = ObjectiveDirection.MINIMIZE if raw_dir == "minimize" else ObjectiveDirection.MAXIMIZE
        return cls(
            name=str(data["name"]),
            direction=direction,
            units=data.get("units"),
            target_col=data.get("target_col"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ModalityDefinition:
    """Domain-agnostic specification of a physical characterization or property measurement modality."""

    name: str
    observation_kind: str  # e.g. "characterization", "objective_measurement", "spectrum", "embedding"
    cost: float = 1.0
    requires: tuple[str, ...] = ()
    objective_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def measures_objective(self, obj_name: str | None = None) -> bool:
        """Returns True if this modality measures an optimization/discovery objective."""
        if obj_name is None:
            return self.observation_kind == "objective_measurement" or bool(self.objective_names)
        return obj_name in self.objective_names

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "observation_kind": self.observation_kind,
            "cost": self.cost,
            "requires": list(self.requires),
            "objective_names": list(self.objective_names),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModalityDefinition:
        return cls(
            name=str(data["name"]),
            observation_kind=str(data.get("observation_kind", "characterization")),
            cost=float(data.get("cost", 1.0)),
            requires=tuple(str(r) for r in data.get("requires", ())),
            objective_names=tuple(str(o) for o in data.get("objective_names", ())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class MaterialDomainConfig:
    """Typed configuration contract for a pluggable material discovery domain."""

    domain_id: str
    candidate_features: tuple[str, ...]
    modalities: tuple[ModalityDefinition, ...]
    objectives: tuple[ObjectiveDefinition, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "candidate_features": list(self.candidate_features),
            "modalities": [m.to_dict() for m in self.modalities],
            "objectives": [o.to_dict() for o in self.objectives],
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class ExperimentExecutor(Protocol):
    """Protocol for executing or revealing experimental measurement actions."""

    def execute(self, action: ScientificAction) -> ExperimentOutcome:
        """Executes a scientific measurement action and returns the revealed outcome."""
        ...


@runtime_checkable
class HypothesisProvider(Protocol):
    """Protocol for constructing domain-specific scientific hypotheses."""

    def build_hypotheses(self) -> Mapping[str, ScientificHypothesisModel]:
        """Constructs and returns the set of competing scientific hypothesis models for this domain."""
        ...


@runtime_checkable
class MaterialDomainAdapter(Protocol):
    """Universal material domain adapter contract for AIcoScientist.

    Decouples core falsification, hypothesis selection, and Bayesian optimization
    from material-specific candidate schemas, characterization modalities, and experimental oracles.
    """

    @property
    def domain_id(self) -> str:
        """Unique identifier of the material system / domain."""
        ...

    def get_config(self) -> MaterialDomainConfig:
        """Returns the typed configuration contract for this domain."""
        ...

    def get_candidate_pool(self) -> pd.DataFrame:
        """Returns the visible candidate pool containing pre-experiment features only.

        STRICT FIREWALL: Contains zero unexecuted ground-truth measurements.
        """
        ...

    def get_candidate_features(self, candidate_id: str) -> Mapping[str, Any]:
        """Returns the pre-experiment feature dictionary for a specific candidate."""
        ...

    def list_valid_actions(
        self,
        state: Any = None,
    ) -> Sequence[ScientificAction]:
        """Lists currently eligible and valid scientific measurement actions."""
        ...

    def execute_or_reveal(self, action: ScientificAction) -> ExperimentOutcome:
        """Executes an action via offline oracle or physical laboratory connector."""
        ...

    def get_objectives(self) -> Sequence[ObjectiveDefinition]:
        """Returns the list of optimization and discovery objectives for this domain."""
        ...

    def get_modality_schema(self) -> Sequence[ModalityDefinition]:
        """Returns the modality definitions and cost models for this domain."""
        ...

    def get_hypothesis_provider(self) -> HypothesisProvider | None:
        """Returns the domain-specific hypothesis factory, or None if domain uses external hypotheses."""
        ...

