from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class OptimizationObjective:
    """Represents a clean, dataset-agnostic optimization objective specification."""

    target_name: str
    minimize: bool = False
    units: str | None = None
    bounds: tuple[float, float] | None = None
    threshold: float | None = None
    constraints: list[Any] = field(default_factory=list)
    secondary_targets: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sense(self) -> str:
        """Returns string representation of optimization direction ('minimize' or 'maximize')."""
        return "minimize" if self.minimize else "maximize"

    @property
    def is_multiobjective(self) -> bool:
        """Returns True if multiple objective targets are configured."""
        return len(self.secondary_targets) > 0 or len(self.weights) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "minimize": self.minimize,
            "sense": self.sense,
            "units": self.units,
            "bounds": self.bounds,
            "threshold": self.threshold,
            "constraints": list(self.constraints),
            "secondary_targets": list(self.secondary_targets),
            "weights": list(self.weights),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizationObjective:
        d = dict(data)
        target_name = str(d.get("target_name") or d.get("target_col") or d.get("name", ""))
        minimize = bool(d.get("minimize", False))
        if "objective" in d:
            minimize = str(d["objective"]).strip().lower() == "minimize"
        elif "sense" in d:
            minimize = str(d["sense"]).strip().lower() == "minimize"

        bounds = None
        if d.get("bounds") is not None:
            bounds = tuple(float(x) for x in d["bounds"])

        return cls(
            target_name=target_name,
            minimize=minimize,
            units=str(d["units"]) if d.get("units") is not None else None,
            bounds=bounds,
            threshold=float(d["threshold"]) if d.get("threshold") is not None else None,
            constraints=list(d.get("constraints", [])),
            secondary_targets=list(d.get("secondary_targets", [])),
            weights=[float(w) for w in d.get("weights", [])],
            metadata=dict(d.get("metadata", {})),
        )

    @classmethod
    def create(
        cls,
        target: str | OptimizationObjective,
        objective: str = "maximize",
        *,
        units: str | None = None,
        bounds: tuple[float, float] | None = None,
        constraints: Sequence[Any] | None = None,
    ) -> OptimizationObjective:
        """Factory helper constructing OptimizationObjective from string or existing instance."""
        if isinstance(target, OptimizationObjective):
            return target
        minimize = objective.strip().lower() == "minimize"
        return cls(
            target_name=str(target),
            minimize=minimize,
            units=units,
            bounds=bounds,
            constraints=list(constraints or []),
        )
