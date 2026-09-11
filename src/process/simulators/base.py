from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class SimulationStatus(StrEnum):
    SUCCESS = "Success"
    INVALID_PHYSICS_RUN = "Invalid physics run"
    NUMERICAL_FAILURE = "Numerical failure"
    TIMEOUT = "Timeout"
    ENVIRONMENT_ERROR = "Environment error"


@dataclass(frozen=True)
class SimulationResult:
    status: SimulationStatus
    run_id: str
    run_directory: Path
    stage_outputs: dict[str, dict[str, float]] = field(default_factory=dict)
    final_outputs: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()


class SimulatorBackend(Protocol):
    def execute(self, recipe: object, *, run_id: str | None = None) -> SimulationResult: ...
