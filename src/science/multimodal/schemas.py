from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.science.domain import ModalityDefinition


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

    def __post_init__(self) -> None:
        for field_name in ("observable_id", "candidate_id", "modality", "name"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        if self.observable_type not in {"scalar", "vector", "categorical", "structured"}:
            raise ValueError("observable_type must be scalar, vector, categorical, or structured")

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
        )


__all__ = ["ModalityDefinition", "ScientificObservable"]
