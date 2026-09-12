from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping
from typing import Any

import torch

from .stages import ProcessStage


SOURCE_VALUES_PREPROCESSING_FINGERPRINT = "source-values/torch-float32-v1"


def _canonical(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _canonical(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("modality identity cannot contain non-finite values")
        return value
    if hasattr(value, "tolist"):
        return _canonical(value.tolist())
    return str(value)


def _provenance_dict(provenance: Any) -> Mapping[str, Any]:
    if hasattr(provenance, "to_dict"):
        provenance = provenance.to_dict()
    return provenance if isinstance(provenance, Mapping) else {}


def source_values_fingerprint(values: Any) -> str:
    """Hash decoded source values using the canonical modality identity form."""
    payload = json.dumps(_canonical(values), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def source_modality_fingerprint(observation: "ModalityObservation") -> str:
    """Hash file content identity together with the values decoded from it."""
    provenance = _provenance_dict(observation.provenance)
    raw_hashes = provenance.get("raw_hashes")
    content_hash = provenance.get("content_hash") or provenance.get("source_content_hash")
    decoded_values_fingerprint = source_values_fingerprint(observation.values)
    if observation.source_path:
        if not raw_hashes and not content_hash:
            raise ValueError("file-backed modality requires a source content hash for production binding")
        source_identity = {
            "content_hash": str(content_hash) if content_hash else None,
            "raw_hashes": _canonical(raw_hashes or {}),
            "decoded_values_fingerprint": decoded_values_fingerprint,
        }
    else:
        source_identity = {"values_fingerprint": decoded_values_fingerprint}
    payload = {
        "modality_id": observation.modality_id,
        "modality_type": observation.modality_type.value,
        "observed_at_stage": observation.observed_at_stage.value,
        "source_identity": source_identity,
        "units": observation.units,
        "shape": observation.shape,
        "missing_reason": observation.missing_reason,
    }
    return hashlib.sha256(json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous CPU bytes without materializing a list."""
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise TypeError("modality tensor fingerprint requires a dense torch.Tensor")
    value = tensor.detach().cpu().contiguous()
    try:
        raw = value.view(torch.uint8).numpy().tobytes()
    except (RuntimeError, TypeError) as exc:
        raise TypeError("modality tensor fingerprint requires a supported dense tensor dtype") from exc
    digest = hashlib.sha256()
    for part in (str(value.dtype), json.dumps(list(value.shape), separators=(",", ":"))):
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    digest.update(raw)
    return digest.hexdigest()


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


_SOURCE_VALUE_MODALITIES = frozenset({
    ModalityType.PROCESS_TABULAR, ModalityType.MACHINE_TIME_SERIES, ModalityType.THERMAL_TIME_SERIES,
    ModalityType.ULTRASOUND_SIGNAL, ModalityType.ULTRASOUND_SPECTRUM, ModalityType.EIS_SPECTRUM,
    ModalityType.XRD_SPECTRUM, ModalityType.FORMATION_CURVE, ModalityType.CYCLING_CURVE,
    ModalityType.SCALAR_METROLOGY,
})


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
        observed_preprocessing = _provenance_dict(observation.provenance).get("preprocessing_fingerprint")
        if observed_preprocessing is not None and observed_preprocessing != self.effective_preprocessing_fingerprint:
            return False
        return True

    @property
    def effective_preprocessing_fingerprint(self) -> str | None:
        if self.preprocessing_fingerprint is not None:
            return self.preprocessing_fingerprint
        if self.modality_type in _SOURCE_VALUE_MODALITIES:
            return SOURCE_VALUES_PREPROCESSING_FINGERPRINT
        return None

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.schema_version, self.slot_name, self.stage.value, self.modality_type.value,
            self.model_input_name, self.expected_input_dim, self.required, self.allowed_missing,
            self.units, self.expected_shape, self.effective_preprocessing_fingerprint,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "slot_name": self.slot_name,
            "stage": self.stage.value, "modality_type": self.modality_type.value,
            "model_input_name": self.model_input_name, "expected_input_dim": self.expected_input_dim,
            "required": self.required, "allowed_missing": self.allowed_missing,
            "units": self.units, "expected_shape": self.expected_shape,
            "preprocessing_fingerprint": self.effective_preprocessing_fingerprint,
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


@dataclass(frozen=True)
class SourceBoundModalityInput:
    """Encoded modality content bound to one exact source observation and slot."""

    source_modality_id: str
    source_modality_fingerprint: str
    slot_name: str
    model_input_name: str
    preprocessing_fingerprint: str
    tensor: torch.Tensor
    tensor_fingerprint: str
    source_kind: str
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not all(str(value).strip() for value in (
            self.source_modality_id, self.source_modality_fingerprint, self.slot_name,
            self.model_input_name, self.preprocessing_fingerprint, self.tensor_fingerprint,
            self.source_kind,
        )):
            raise ValueError("source-bound modality input requires complete identity")
        if tensor_fingerprint(self.tensor) != self.tensor_fingerprint:
            raise ValueError("source-bound modality tensor fingerprint is invalid")

    @classmethod
    def from_observation(cls, observation: ModalityObservation, slot: ModalitySlotSpec) -> "SourceBoundModalityInput":
        if observation.is_missing:
            raise ValueError("missing source modalities cannot produce encoded tensors")
        if not slot.matches(observation):
            raise ValueError("source modality does not match its semantic slot")
        preprocessing = slot.effective_preprocessing_fingerprint
        if preprocessing is None or observation.modality_type not in _SOURCE_VALUE_MODALITIES:
            raise ValueError(f"no trusted preprocessing pipeline exists for slot {slot.slot_name!r}")
        if preprocessing != SOURCE_VALUES_PREPROCESSING_FINGERPRINT:
            raise ValueError(f"unsupported preprocessing contract for slot {slot.slot_name!r}")
        if observation.source_path:
            decoded_values_fingerprint = _provenance_dict(observation.provenance).get("decoded_values_fingerprint")
            if decoded_values_fingerprint != source_values_fingerprint(observation.values):
                raise ValueError("file-backed modality requires a trusted decoded_values_fingerprint")
        try:
            tensor = torch.as_tensor(observation.values, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"source modality {observation.modality_id!r} is not supported by the trusted numeric preprocessor") from exc
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        if not torch.isfinite(tensor).all():
            raise ValueError("trusted source modality preprocessing must produce a finite tensor")
        return cls(
            source_modality_id=observation.modality_id,
            source_modality_fingerprint=source_modality_fingerprint(observation),
            slot_name=slot.slot_name,
            model_input_name=slot.model_input_name,
            preprocessing_fingerprint=preprocessing,
            tensor=tensor,
            tensor_fingerprint=tensor_fingerprint(tensor),
            source_kind=observation.modality_type.value,
            provenance=_provenance_dict(observation.provenance),
        )

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        source_modality_id: str,
        slot: ModalitySlotSpec | None = None,
        slot_name: str | None = None,
        model_input_name: str | None = None,
        preprocessing_fingerprint: str | None = None,
        source_modality_fingerprint: str = "test-only",
        source_kind: str = "TEST_ONLY",
        provenance: Mapping[str, Any] | None = None,
        test_only: bool = False,
    ) -> "SourceBoundModalityInput":
        if not test_only:
            raise ValueError("wrapping an arbitrary modality tensor is test-only")
        if slot is not None:
            slot_name = slot.slot_name
            model_input_name = slot.model_input_name
            preprocessing_fingerprint = preprocessing_fingerprint or slot.effective_preprocessing_fingerprint
        if not slot_name or not model_input_name or not preprocessing_fingerprint:
            raise ValueError("test-only source-bound tensors require slot and preprocessing identity")
        return cls(
            source_modality_id=source_modality_id,
            source_modality_fingerprint=source_modality_fingerprint,
            slot_name=slot_name,
            model_input_name=model_input_name,
            preprocessing_fingerprint=preprocessing_fingerprint,
            tensor=tensor,
            tensor_fingerprint=tensor_fingerprint(tensor),
            source_kind=source_kind,
            provenance=provenance,
        )
