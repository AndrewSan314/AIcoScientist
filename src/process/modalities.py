from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping
from typing import Any

from .stages import ProcessStage


class ModalityType(str, Enum):
    PROCESS_TABULAR = "PROCESS_TABULAR"
    MACHINE_TIME_SERIES = "MACHINE_TIME_SERIES"
    THERMAL_TIME_SERIES = "THERMAL_TIME_SERIES"
    OPTICAL_IMAGE = "OPTICAL_IMAGE"
    SEM_IMAGE = "SEM_IMAGE"
    EDS_MAP = "EDS_MAP"
    XCT_VOLUME = "XCT_VOLUME"
    ULTRASOUND_SIGNAL = "ULTRASOUND_SIGNAL"
    ULTRASOUND_SPECTRUM = "ULTRASOUND_SPECTRUM"
    EIS_SPECTRUM = "EIS_SPECTRUM"
    XRD_SPECTRUM = "XRD_SPECTRUM"
    FORMATION_CURVE = "FORMATION_CURVE"
    CYCLING_CURVE = "CYCLING_CURVE"
    SCALAR_METROLOGY = "SCALAR_METROLOGY"
    ENVIRONMENT_METADATA = "ENVIRONMENT_METADATA"


@dataclass(frozen=True)
class ModalitySlotSpec:
    """Frozen semantic contract between a source modality and a model input."""

    slot_name: str
    stage: ProcessStage
    modality_type: ModalityType
    model_input_name: str
    expected_input_dim: int
    required: bool = False
    allowed_missing: bool = True
    units: str | None = None
    expected_shape: tuple[int, ...] | None = None
    preprocessing_fingerprint: str | None = None
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.slot_name.strip() or not self.model_input_name.strip() or not self.schema_version.strip():
            raise ValueError("modality slots require names and a schema version")
        if not isinstance(self.stage, ProcessStage) or not isinstance(self.modality_type, ModalityType):
            raise TypeError("modality slots require canonical stage and modality type values")
        if int(self.expected_input_dim) <= 0:
            raise ValueError("modality slot expected_input_dim must be positive")
        if self.expected_shape is not None and any(int(size) <= 0 for size in self.expected_shape):
            raise ValueError("modality slot expected_shape dimensions must be positive")

    def matches(self, observation: "ModalityObservation") -> bool:
        if observation.observed_at_stage != self.stage or observation.modality_type != self.modality_type:
            return False
        if self.units is not None and observation.units != self.units:
            return False
        if self.expected_shape is not None and observation.shape is not None and tuple(observation.shape) != self.expected_shape:
            return False
        if self.preprocessing_fingerprint is not None:
            provenance = observation.provenance
            observed = provenance.get("preprocessing_fingerprint") if isinstance(provenance, Mapping) else None
            if observed != self.preprocessing_fingerprint:
                return False
        return True

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.schema_version, self.slot_name, self.stage.value, self.modality_type.value,
            self.model_input_name, self.expected_input_dim, self.required, self.allowed_missing,
            self.units, self.expected_shape, self.preprocessing_fingerprint,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "slot_name": self.slot_name,
            "stage": self.stage.value, "modality_type": self.modality_type.value,
            "model_input_name": self.model_input_name, "expected_input_dim": self.expected_input_dim,
            "required": self.required, "allowed_missing": self.allowed_missing,
            "units": self.units, "expected_shape": self.expected_shape,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
        }


@dataclass(frozen=True)
class ModalityObservation:
    modality_id: str
    modality_type: ModalityType
    observed_at_stage: ProcessStage
    source_path: str | None = None
    values: Any = None
    units: str | None = None
    shape: tuple[int, ...] | None = None
    missing_reason: str | None = None
    quality_score: float | None = None
    provenance: Any = None

    def __post_init__(self) -> None:
        if not self.modality_id.strip():
            raise ValueError("modality_id is required")
        missing = self.missing_reason is not None
        if missing == (self.values is not None):
            raise ValueError("a modality must be observed or explicitly missing, never both/neither")
        if missing and not str(self.missing_reason).strip():
            raise ValueError("missing_reason must be non-empty when values are unavailable")
        if self.shape is not None and any(int(size) <= 0 for size in self.shape):
            raise ValueError("modality shape dimensions must be positive")
        if self.quality_score is not None and not 0.0 <= float(self.quality_score) <= 1.0:
            raise ValueError("quality_score must be between 0 and 1")

    @property
    def is_missing(self) -> bool:
        return self.missing_reason is not None

    def to_dict(self) -> dict[str, Any]:
        provenance = self.provenance.to_dict() if hasattr(self.provenance, "to_dict") else self.provenance
        return {
            "modality_id": self.modality_id,
            "modality_type": self.modality_type.value,
            "observed_at_stage": self.observed_at_stage.value,
            "source_path": self.source_path,
            "values": self.values,
            "units": self.units,
            "shape": list(self.shape) if self.shape else None,
            "missing_reason": self.missing_reason,
            "quality_score": self.quality_score,
            "provenance": provenance,
        }
