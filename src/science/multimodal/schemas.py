from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.science.domain import ModalityDefinition
from src.science.multimodal.ontology import OBSERVABLE_REGISTRY, validate_observable


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class ScientificObservable:
    """A derived observation that remains traceable to its raw measurement."""

    observable_id: str
    candidate_id: str
    modality: str
    name: str
    value: Any
    uncertainty: Any | None = None
    units: str | None = None
    raw_artifact_ref: str | None = None
    extractor_name: str | None = None
    extractor_version: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    observable_type: str = "scalar"
    observable_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for field_name in ("observable_id", "candidate_id", "modality", "name"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        if self.observable_type not in {"scalar", "vector", "categorical", "structured"}:
            raise ValueError("observable_type must be scalar, vector, categorical, or structured")
        values = np.atleast_1d(np.asarray(self.value, dtype=np.float64)) if self.observable_type in {"scalar", "vector"} else None
        explicit_names = tuple(self.observable_names or ())
        names = explicit_names
        legacy_vector = self.observable_type == "scalar" and not explicit_names and values is not None and values.size > 1
        if not names and not legacy_vector:
            names = (self.name,) if self.observable_type != "vector" else ()
        if len(set(names)) != len(names):
            raise ValueError("observable_names must be unique")
        if self.observable_type == "vector":
            if not names or len(names) != len(values):
                raise ValueError("vector observables require one named schema entry per value")
        elif len(names) != 1 and not legacy_vector:
            raise ValueError("scalar and categorical observables require exactly one observable name")
        if self.observable_type in {"scalar", "vector"}:
            values = np.asarray(self.value, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError("observable values must be finite")
        if self.uncertainty is not None:
            errors = np.asarray(self.uncertainty, dtype=np.float64)
            if not np.all(np.isfinite(errors)) or np.any(errors < 0):
                raise ValueError("observable uncertainty must be finite and non-negative")
        for name in names:
            if name in {"test", "controlled_reveal", "canonical_replay_observation"}:
                continue
            definition = OBSERVABLE_REGISTRY[name]
            if definition.modality != self.modality.upper():
                raise ValueError(
                    f"observable {name!r} belongs to {definition.modality}, not {self.modality.upper()}"
                )
            validate_observable(name, self.value if len(names) == 1 else np.asarray(self.value)[names.index(name)], self.uncertainty)
        object.__setattr__(self, "observable_names", names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observable_id": self.observable_id,
            "candidate_id": self.candidate_id,
            "modality": self.modality,
            "name": self.name,
            "value": _jsonable(self.value),
            "uncertainty": _jsonable(self.uncertainty),
            "units": self.units,
            "raw_artifact_ref": self.raw_artifact_ref,
            "extractor_name": self.extractor_name,
            "extractor_version": self.extractor_version,
            "provenance": _jsonable(dict(self.provenance)),
            "timestamp": self.timestamp,
            "observable_type": self.observable_type,
            "observable_names": list(self.observable_names),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScientificObservable:
        return cls(
            observable_id=str(data["observable_id"]),
            candidate_id=str(data["candidate_id"]),
            modality=str(data["modality"]),
            name=str(data["name"]),
            value=data.get("value"),
            uncertainty=data.get("uncertainty"),
            units=data.get("units"),
            raw_artifact_ref=data.get("raw_artifact_ref"),
            extractor_name=data.get("extractor_name"),
            extractor_version=data.get("extractor_version"),
            provenance=dict(data.get("provenance", {})),
            timestamp=data.get("timestamp"),
            observable_type=str(data.get("observable_type", "scalar")),
            observable_names=tuple(data.get("observable_names", ())) or None,
        )


@dataclass(frozen=True)
class ScientificObservableBundle:
    """Schema-driven collection of observations from one candidate and modality."""

    candidate_id: str
    modality: str
    observables: tuple[ScientificObservable, ...]

    def __post_init__(self) -> None:
        if not str(self.candidate_id).strip() or not str(self.modality).strip():
            raise ValueError("candidate_id and modality must be non-empty")
        if not self.observables:
            raise ValueError("an observable bundle cannot be empty")
        names = [name for obs in self.observables for name in obs.observable_names]
        if len(names) != len(set(names)):
            raise ValueError("observable bundle names must be unique")
        for obs in self.observables:
            if obs.candidate_id != self.candidate_id or obs.modality.upper() != self.modality.upper():
                raise ValueError("all bundle members must share candidate_id and modality")

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(name for obs in self.observables for name in obs.observable_names)

    def vector(self, required_names: tuple[str, ...] | list[str]) -> np.ndarray:
        by_name = {name: obs.value for obs in self.observables for name in obs.observable_names}
        required = tuple(required_names)
        missing = [name for name in required if name not in by_name]
        if missing:
            raise ValueError(f"observable bundle missing required names: {missing}")
        return np.asarray([by_name[name] for name in required], dtype=np.float64)


__all__ = ["ModalityDefinition", "ScientificObservable", "ScientificObservableBundle"]
