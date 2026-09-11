from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.process.stages import ProcessStage


@dataclass(frozen=True)
class Prediction:
    mean: float
    std: float | None = None
    units: str | None = None


@dataclass(frozen=True)
class ProcessControlProposal:
    proposal_id: str
    stage: ProcessStage | None
    controls: dict[str, Any]
    predicted_outputs: dict[str, Prediction]
    feasibility_probability: float | None
    acquisition_value: float | None
    pareto_rank: int | None
    model_version: str
    data_fingerprint: str
    source_recipe_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
