from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
