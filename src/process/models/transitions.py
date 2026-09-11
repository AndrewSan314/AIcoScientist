from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from ..contracts import StageRecord
from ..modalities import ModalityObservation
from ..stages import ProcessStage


def source_stage_fingerprint(record: StageRecord) -> str:
    payload = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LegalStageTransition:
    """A model-ready transition bound to one audited source StageRecord."""

    source_stage_id: str
    stage: ProcessStage
    controls: torch.Tensor
    scalar_observations: torch.Tensor
    modality_observations: tuple[ModalityObservation, ...]
    provenance: Mapping[str, Any]
    modality_inputs: Mapping[str, torch.Tensor] = field(default_factory=dict)
    availability: Mapping[str, bool | torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_stage_id.strip() or not isinstance(self.stage, ProcessStage):
            raise ValueError("legal transition needs a source stage identity and stage")
        if self.provenance.get("source_stage_id") != self.source_stage_id or self.provenance.get("stage") not in {self.stage, self.stage.value}:
            raise ValueError("legal transition provenance does not bind to its source stage")
        if not self.provenance.get("source_stage_fingerprint"):
            raise ValueError("legal transition requires a source stage fingerprint")
        if any(key in self.provenance for key in ("final_kpi", "final_kpis", "final_outputs", "future_kpi", "future_metadata")):
            raise ValueError("final KPIs cannot be transition provenance")
        for modality in self.modality_observations:
            if modality.observed_at_stage != self.stage:
                raise ValueError("transition modalities must come from the bound source stage")
        if set(self.modality_inputs) - set(self.availability):
            raise ValueError("every encoded modality must have an explicit availability mask")
        for name, available in self.availability.items():
            if bool(torch.as_tensor(available).any()) and name not in self.modality_inputs:
                raise ValueError(f"available modality {name!r} requires an encoded token")
            if not bool(torch.as_tensor(available).any()) and name in self.modality_inputs:
                raise ValueError(f"unavailable modality {name!r} must not have an encoded token")

    @classmethod
    def from_source_stage(
        cls,
        record: StageRecord,
        *,
        controls: torch.Tensor,
        scalar_observations: torch.Tensor,
        modality_inputs: Mapping[str, torch.Tensor],
        modality_bindings: Mapping[str, str],
        provenance: Mapping[str, Any] | None = None,
    ) -> "LegalStageTransition":
        tokens: dict[str, torch.Tensor] = {}
        availability: dict[str, bool] = {}
        for modality in record.modalities:
            model_name = modality_bindings.get(modality.modality_id)
            if model_name is None or model_name in availability:
                raise ValueError(f"source modality {modality.modality_id!r} lacks a unique model binding")
            availability[model_name] = not modality.is_missing
            if not modality.is_missing:
                if modality.modality_id not in modality_inputs:
                    raise ValueError(f"source modality {modality.modality_id!r} lacks encoder input")
                tokens[model_name] = modality_inputs[modality.modality_id]
        supplied_provenance = dict(provenance or {})
        reserved = {"source_stage_id", "stage", "source_stage_fingerprint", "control_names", "scalar_observation_names", "modality_bindings"}
        if reserved & set(supplied_provenance):
            raise ValueError("source-bound transition provenance fields cannot be overridden")
        if {"final_kpi", "final_kpis", "final_outputs", "future_kpi", "future_metadata"} & set(supplied_provenance):
            raise ValueError("final KPIs cannot be transition provenance")
        bound_provenance = {
            **supplied_provenance,
            "source_stage_id": record.stage_id,
            "stage": record.stage_type.value,
            "source_stage_fingerprint": source_stage_fingerprint(record),
            "control_names": tuple(record.controls),
            "scalar_observation_names": tuple(record.intermediate_properties),
            "modality_bindings": {modality.modality_id: modality_bindings[modality.modality_id] for modality in record.modalities},
        }
        return cls(
            source_stage_id=record.stage_id, stage=record.stage_type, controls=controls,
            scalar_observations=scalar_observations, modality_observations=tuple(record.modalities),
            provenance=bound_provenance, modality_inputs=tokens, availability=availability,
        )
