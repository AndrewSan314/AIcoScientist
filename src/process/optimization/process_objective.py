from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ObjectiveSpec:
    target: str
    sense: Literal["maximize", "minimize"]
    units: str | None = None

    def __post_init__(self) -> None:
        if not self.target.strip() or self.sense not in {"maximize", "minimize"}:
            raise ValueError("objective needs a target and maximize/minimize sense")


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    type: Literal["lower", "upper", "range", "feasibility"]
    threshold: float | tuple[float, float]
    hard: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("constraint name is required")
        if self.type == "range":
            if not isinstance(self.threshold, tuple) or len(self.threshold) != 2 or self.threshold[0] > self.threshold[1]:
                raise ValueError("range constraint needs ordered (lower, upper) thresholds")
        elif isinstance(self.threshold, tuple):
            raise ValueError("only range constraints accept tuple thresholds")

    def satisfied(self, value: float) -> bool:
        if self.type == "lower":
            return value >= float(self.threshold)
        if self.type == "upper":
            return value <= float(self.threshold)
        if self.type == "range":
            lower, upper = self.threshold
            return lower <= value <= upper
        return value >= float(self.threshold)


@dataclass(frozen=True)
class ProcessOptimizationObjective:
    objectives: list[ObjectiveSpec]
    constraints: list[ConstraintSpec] = field(default_factory=list)
    reference_point: list[float] | None = None
    feasibility_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("at least one process objective is required")
        if self.reference_point is not None and len(self.reference_point) != len(self.objectives):
            raise ValueError("reference_point length must match objective count")
        if self.feasibility_threshold is not None and not 0 <= self.feasibility_threshold <= 1:
            raise ValueError("feasibility_threshold must be in [0, 1]")

    @property
    def is_multiobjective(self) -> bool:
        return len(self.objectives) > 1
