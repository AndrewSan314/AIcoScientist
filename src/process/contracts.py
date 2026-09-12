from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .modalities import ModalityObservation, ModalityType
from .stages import ProcessStage, STAGE_ORDER


@dataclass(frozen=True)
class ProvenanceRecord:
    evidence_kind: str
    source_url: str | None = None
    source_doi: str | None = None
    source_version: str | None = None
    raw_hashes: Mapping[str, str] = field(default_factory=dict)
    adapter_version: str | None = None
    adapter_git_sha: str | None = None
    processing_parameters: Mapping[str, Any] = field(default_factory=dict)
    decoded_values_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.evidence_kind.strip():
            raise ValueError("evidence_kind is required")
        if any(not str(name).strip() or not str(digest).strip() for name, digest in self.raw_hashes.items()):
            raise ValueError("raw_hashes must map non-empty paths to non-empty digests")
        if self.decoded_values_fingerprint is not None and not self.decoded_values_fingerprint.strip():
            raise ValueError("decoded_values_fingerprint must be non-empty when supplied")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_kind": self.evidence_kind,
            "source_url": self.source_url,
            "source_doi": self.source_doi,
            "source_version": self.source_version,
            "raw_hashes": dict(sorted(self.raw_hashes.items())),
            "adapter_version": self.adapter_version,
            "adapter_git_sha": self.adapter_git_sha,
            "processing_parameters": dict(self.processing_parameters),
            "decoded_values_fingerprint": self.decoded_values_fingerprint,
        }


@dataclass(frozen=True)
class ParameterValue:
    value: Any
    units: str | None = None
    source_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "units": self.units, "source_name": self.source_name}


@dataclass(frozen=True)
class MeasurementValue:
    value: Any
    units: str | None = None
    uncertainty: float | None = None
    source_name: str | None = None

    def __post_init__(self) -> None:
        if self.uncertainty is not None and float(self.uncertainty) < 0:
            raise ValueError("measurement uncertainty cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "units": self.units,
            "uncertainty": self.uncertainty,
            "source_name": self.source_name,
        }


@dataclass(frozen=True)
class StageTimeRange:
    started_at: str | None = None
    ended_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"started_at": self.started_at, "ended_at": self.ended_at}


@dataclass
class StageRecord:
    stage_id: str
    stage_type: ProcessStage
    sequence_index: int
    controls: dict[str, ParameterValue]
    intermediate_properties: dict[str, MeasurementValue]
    modalities: list[ModalityObservation]
    upstream_stage_id: str | None = None
    timestamps: StageTimeRange | None = None
    quality_flags: list[str] = field(default_factory=list)
    provenance: ProvenanceRecord | None = None

    def __post_init__(self) -> None:
        if not self.stage_id.strip() or self.sequence_index < 0:
            raise ValueError("stage_id and non-negative sequence_index are required")
        overlap = set(self.controls) & set(self.intermediate_properties)
        if overlap:
            raise ValueError(f"controls and intermediate properties must remain distinct: {sorted(overlap)}")
        if any(modality.observed_at_stage != self.stage_type for modality in self.modalities):
            raise ValueError("a modality must be attached to the stage at which it was observed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_type": self.stage_type.value,
            "sequence_index": self.sequence_index,
            "controls": {name: value.to_dict() for name, value in self.controls.items()},
            "intermediate_properties": {name: value.to_dict() for name, value in self.intermediate_properties.items()},
            "modalities": [modality.to_dict() for modality in self.modalities],
            "upstream_stage_id": self.upstream_stage_id,
            "timestamps": self.timestamps.to_dict() if self.timestamps else None,
            "quality_flags": list(self.quality_flags),
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


@dataclass(frozen=True)
class BatteryProcessRun:
    run_id: str
    cell_id: str | None
    batch_id: str | None
    chemistry_id: str
    equipment_context: Mapping[str, Any]
    environment_context: Mapping[str, Any]
    stages: list[StageRecord]
    final_kpis: Mapping[str, MeasurementValue]
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.chemistry_id.strip():
            raise ValueError("run_id and chemistry_id are required")
        ids = [stage.stage_id for stage in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError("stage IDs must be unique within a process run")
        if [stage.sequence_index for stage in self.stages] != sorted(stage.sequence_index for stage in self.stages):
            raise ValueError("stages must be supplied in sequence order")
        if any(stage.sequence_index != STAGE_ORDER[stage.stage_type] for stage in self.stages):
            raise ValueError("stage sequence_index must match canonical ProcessStage order")
        stage_ids = set(ids)
        if any(stage.upstream_stage_id not in stage_ids for stage in self.stages if stage.upstream_stage_id):
            raise ValueError("upstream_stage_id must reference a stage in this process run")

    @property
    def identity_fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "batch_id": self.batch_id,
            "chemistry_id": self.chemistry_id,
            "equipment_context": dict(self.equipment_context),
            "environment_context": dict(self.environment_context),
            "stages": [stage.to_dict() for stage in self.stages],
            "final_kpis": {name: value.to_dict() for name, value in self.final_kpis.items()},
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BatteryProcessRun":
        def provenance(raw: Mapping[str, Any] | None) -> ProvenanceRecord | None:
            return ProvenanceRecord(**dict(raw)) if raw else None

        def parameter(raw: Mapping[str, Any]) -> ParameterValue:
            return ParameterValue(**dict(raw))

        def measurement(raw: Mapping[str, Any]) -> MeasurementValue:
            return MeasurementValue(**dict(raw))

        stages = []
        for raw_stage in data.get("stages", []):
            raw_stage = dict(raw_stage)
            modalities = [
                ModalityObservation(
                    modality_id=raw["modality_id"],
                    modality_type=ModalityType(raw["modality_type"]),
                    observed_at_stage=ProcessStage(raw["observed_at_stage"]),
                    source_path=raw.get("source_path"),
                    values=raw.get("values"),
                    units=raw.get("units"),
                    shape=tuple(raw["shape"]) if raw.get("shape") else None,
                    missing_reason=raw.get("missing_reason"),
                    quality_score=raw.get("quality_score"),
                    provenance=provenance(raw.get("provenance")),
                )
                for raw in raw_stage.get("modalities", [])
            ]
            timestamps = raw_stage.get("timestamps")
            stages.append(
                StageRecord(
                    stage_id=raw_stage["stage_id"],
                    stage_type=ProcessStage(raw_stage["stage_type"]),
                    sequence_index=int(raw_stage["sequence_index"]),
                    controls={key: parameter(value) for key, value in raw_stage.get("controls", {}).items()},
                    intermediate_properties={key: measurement(value) for key, value in raw_stage.get("intermediate_properties", {}).items()},
                    modalities=modalities,
                    upstream_stage_id=raw_stage.get("upstream_stage_id"),
                    timestamps=StageTimeRange(**timestamps) if timestamps else None,
                    quality_flags=list(raw_stage.get("quality_flags", [])),
                    provenance=provenance(raw_stage.get("provenance")),
                )
            )
        return cls(
            run_id=str(data["run_id"]),
            cell_id=data.get("cell_id"),
            batch_id=data.get("batch_id"),
            chemistry_id=str(data["chemistry_id"]),
            equipment_context=dict(data.get("equipment_context", {})),
            environment_context=dict(data.get("environment_context", {})),
            stages=stages,
            final_kpis={key: measurement(value) for key, value in data.get("final_kpis", {}).items()},
            provenance=provenance(data.get("provenance")) or ProvenanceRecord(evidence_kind="UNSPECIFIED"),
        )
