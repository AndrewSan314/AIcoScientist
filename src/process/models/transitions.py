from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping, Sequence

import torch
import pandas as pd

from ..contracts import StageRecord
from ..modalities import ModalityObservation
from ..stages import ProcessStage


def source_stage_fingerprint(record: StageRecord) -> str:
    payload = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, Real) and not isinstance(value, bool):
        return math.isnan(float(value))
    return False


def _category(value: Any) -> tuple[str, str]:
    return ("missing", "") if _missing(value) else ("value", str(value))


def _field_value(value: Any) -> Any:
    return value.value


@dataclass(frozen=True)
class EncodedStageFeatures:
    tensor: torch.Tensor
    feature_names: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class _FieldSpec:
    name: str
    kind: str
    categories: tuple[tuple[str, str], ...] = ()

    @property
    def width(self) -> int:
        return 2 if self.kind == "numeric" else len(self.categories)


@dataclass(frozen=True)
class _StageSpec:
    stage_type: ProcessStage
    controls: tuple[_FieldSpec, ...]
    observations: tuple[_FieldSpec, ...]


@dataclass(frozen=True)
class StageFeatureEncoder:
    """Deterministic, source-schema-bound encoder for legal stage records."""

    stages: tuple[_StageSpec, ...]
    control_dim: int
    observation_dim: int
    fingerprint: str

    @classmethod
    def from_horizon(
        cls,
        horizon: Any,
        *,
        category_vocabularies: Mapping[str, Sequence[object]] | None = None,
        control_dim: int | None = None,
        observation_dim: int | None = None,
    ) -> "StageFeatureEncoder":
        return cls.fit(
            horizon,
            category_vocabularies=category_vocabularies,
            control_dim=control_dim,
            observation_dim=observation_dim,
        )

    @classmethod
    def fit(
        cls,
        training_data: Any,
        *,
        category_vocabularies: Mapping[str, Sequence[object]] | None = None,
        control_dim: int | None = None,
        observation_dim: int | None = None,
    ) -> "StageFeatureEncoder":
        records = tuple(getattr(training_data, "source_stages", training_data))
        source_ids = tuple(getattr(training_data, "source_stage_ids", (record.stage_id for record in records)))
        if not records or tuple(record.stage_id for record in records) != source_ids:
            raise ValueError("StageFeatureEncoder requires source stages in HorizonView order")
        if len({record.stage_type for record in records}) != len(records):
            raise ValueError("StageFeatureEncoder requires one training schema per stage type")
        vocabularies = category_vocabularies or {}
        stages = tuple(
            _StageSpec(
                record.stage_type,
                cls._fields(record, "control", record.controls, vocabularies),
                cls._fields(record, "observation", record.intermediate_properties, vocabularies),
            )
            for record in records
        )
        inferred_control_dim = max(sum(field.width for field in stage.controls) for stage in stages)
        inferred_observation_dim = max(sum(field.width for field in stage.observations) for stage in stages)
        if control_dim is not None and control_dim < inferred_control_dim or observation_dim is not None and observation_dim < inferred_observation_dim:
            raise ValueError("explicit StageFeatureEncoder dimensions cannot truncate source features")
        control_dim = control_dim or inferred_control_dim
        observation_dim = observation_dim or inferred_observation_dim
        schema = {
            "stages": [
                {
                    "stage": stage.stage_type.value,
                    "controls": [field.__dict__ for field in stage.controls],
                    "observations": [field.__dict__ for field in stage.observations],
                }
                for stage in stages
            ],
            "control_dim": control_dim,
            "observation_dim": observation_dim,
        }
        fingerprint = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return cls(stages, control_dim, observation_dim, fingerprint)

    @classmethod
    def from_training_data(cls, training_data: Any, **kwargs: Any) -> "StageFeatureEncoder":
        return cls.fit(training_data, **kwargs)

    @classmethod
    def from_training(cls, training_data: Any, **kwargs: Any) -> "StageFeatureEncoder":
        return cls.fit(training_data, **kwargs)

    @property
    def feature_schema(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                stage.stage_type.value,
                tuple((field.name, field.kind, field.categories) for field in stage.controls),
                tuple((field.name, field.kind, field.categories) for field in stage.observations),
            )
            for stage in self.stages
        )

    @property
    def category_vocabularies(self) -> dict[str, tuple[tuple[str, str], ...]]:
        return {
            field.name: field.categories
            for stage in self.stages
            for field in (*stage.controls, *stage.observations)
            if field.kind == "categorical"
        }

    @staticmethod
    def _fields(
        record: StageRecord,
        scope: str,
        fields: Mapping[str, Any],
        vocabularies: Mapping[str, Sequence[object]],
    ) -> tuple[_FieldSpec, ...]:
        result: list[_FieldSpec] = []
        stage_name = record.stage_type.value.lower()
        for name in sorted(fields):
            raw = _field_value(fields[name])
            key = f"{stage_name}.{scope}.{name}"
            short_key = f"{stage_name}.{name}"
            vocabulary = vocabularies.get(key, vocabularies.get(short_key, vocabularies.get(name)))
            if _missing(raw) and vocabulary is None:
                result.append(_FieldSpec(key, "numeric"))
                continue
            if not isinstance(raw, Real) or isinstance(raw, bool):
                if vocabulary is None:
                    raise ValueError(f"categorical source field {key!r} requires a source-supported vocabulary")
                categories = tuple(sorted({_category(value) for value in vocabulary}))
                if not categories or _category(raw) not in categories:
                    raise ValueError(f"categorical source field {key!r} is unknown to its vocabulary")
                result.append(_FieldSpec(key, "categorical", categories))
            else:
                if not math.isfinite(float(raw)):
                    raise ValueError(f"source field {key!r} contains a non-finite number")
                result.append(_FieldSpec(key, "numeric"))
        return tuple(result)

    def _stage(self, record: StageRecord) -> _StageSpec:
        matches = [stage for stage in self.stages if stage.stage_type == record.stage_type]
        if len(matches) != 1:
            raise ValueError(f"stage type {record.stage_type.value!r} is not uniquely represented by the encoder")
        stage = matches[0]
        control_names = tuple(f"{record.stage_type.value.lower()}.control.{name}" for name in sorted(record.controls))
        observation_names = tuple(f"{record.stage_type.value.lower()}.observation.{name}" for name in sorted(record.intermediate_properties))
        if tuple(field.name for field in stage.controls) != control_names or tuple(field.name for field in stage.observations) != observation_names:
            raise ValueError("source stage schema does not match the StageFeatureEncoder")
        return stage

    def _encode_scope(self, record: StageRecord, scope: str) -> EncodedStageFeatures:
        stage = self._stage(record)
        specs = stage.controls if scope == "control" else stage.observations
        source = record.controls if scope == "control" else record.intermediate_properties
        values: list[float] = []
        names: list[str] = []
        for spec in specs:
            raw = _field_value(source[spec.name.rsplit(".", 1)[-1]])
            if spec.kind == "numeric":
                values.extend([0.0, 0.0] if _missing(raw) else [float(raw), 1.0])
                names.extend((f"{spec.name}.value", f"{spec.name}.observed"))
            else:
                token = _category(raw)
                if token not in spec.categories:
                    raise ValueError(f"categorical source field {spec.name!r} is unknown to its vocabulary")
                values.extend(float(token == category) for category in spec.categories)
                names.extend(f"{spec.name}.category.{index}" for index in range(len(spec.categories)))
        target_dim = self.control_dim if scope == "control" else self.observation_dim
        for index in range(len(values), target_dim):
            values.append(0.0)
            names.append(f"{record.stage_type.value.lower()}.{scope}.__padding.{index}")
        tensor = torch.tensor(values, dtype=torch.float32)
        if not torch.isfinite(tensor).all():
            raise ValueError("encoded source stage features must be finite")
        digest_payload = {"encoder": self.fingerprint, "stage": record.stage_type.value, "scope": scope, "values": values, "names": names}
        digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return EncodedStageFeatures(tensor, tuple(names), digest)

    def encode_controls(self, record: StageRecord) -> EncodedStageFeatures:
        return self._encode_scope(record, "control")

    def encode_observations(self, record: StageRecord) -> EncodedStageFeatures:
        return self._encode_scope(record, "observation")

    def validate_transition(self, record: StageRecord, transition: "LegalStageTransition") -> None:
        controls = self.encode_controls(record)
        observations = self.encode_observations(record)
        if transition.provenance.get("encoder_fingerprint") != self.fingerprint:
            raise ValueError("transition encoder fingerprint is not source-bound")
        if transition.provenance.get("control_feature_names") != controls.feature_names or transition.provenance.get("observation_feature_names") != observations.feature_names:
            raise ValueError("transition feature names are not source-bound")
        if transition.provenance.get("encoded_control_fingerprint") != controls.fingerprint or transition.provenance.get("encoded_observation_fingerprint") != observations.fingerprint:
            raise ValueError("transition encoded features are not source-bound")
        if not torch.equal(transition.controls, controls.tensor) or not torch.equal(transition.scalar_observations, observations.tensor):
            raise ValueError("transition tensors do not equal deterministic source-record encodings")


@dataclass(frozen=True)
class SourceBackedInitialState:
    tensor: torch.Tensor
    source_stage_ids: tuple[str, ...]
    model_fingerprint: str
    encoder_fingerprint: str
    provenance_fingerprint: str

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        source_stage_ids: Sequence[str],
        model_fingerprint: str,
        encoder_fingerprint: str,
        test_only: bool = False,
    ) -> "SourceBackedInitialState":
        if not test_only:
            raise ValueError("wrapping an arbitrary initial tensor is test-only")
        if not isinstance(tensor, torch.Tensor) or tensor.ndim not in (1, 2) or not torch.isfinite(tensor).all():
            raise ValueError("source-backed initial state must be a finite rank-1 or rank-2 tensor")
        ids = tuple(str(item) for item in source_stage_ids)
        if not ids or not model_fingerprint.strip() or not encoder_fingerprint.strip():
            raise ValueError("source-backed initial state requires source IDs, model, and encoder fingerprints")
        payload = {"source_stage_ids": ids, "model_fingerprint": model_fingerprint, "encoder_fingerprint": encoder_fingerprint, "tensor": tensor.detach().cpu().tolist()}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return cls(tensor, ids, model_fingerprint, encoder_fingerprint, fingerprint)


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

    @staticmethod
    def _modality_payload(record: StageRecord, modality_inputs: Mapping[str, torch.Tensor], modality_bindings: Mapping[str, str]) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
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
        return tokens, availability

    @classmethod
    def from_encoded_source_stage(
        cls,
        record: StageRecord,
        *,
        encoder: StageFeatureEncoder,
        modality_inputs: Mapping[str, torch.Tensor],
        modality_bindings: Mapping[str, str],
        provenance: Mapping[str, Any] | None = None,
    ) -> "LegalStageTransition":
        controls = encoder.encode_controls(record)
        observations = encoder.encode_observations(record)
        tokens, availability = cls._modality_payload(record, modality_inputs, modality_bindings)
        supplied = dict(provenance or {})
        reserved = {
            "source_stage_id", "stage", "source_stage_fingerprint", "control_names", "scalar_observation_names", "modality_bindings",
            "encoder_fingerprint", "control_feature_names", "observation_feature_names", "encoded_control_fingerprint", "encoded_observation_fingerprint",
        }
        if reserved & set(supplied):
            raise ValueError("source-bound transition provenance fields cannot be overridden")
        if {"final_kpi", "final_kpis", "final_outputs", "future_kpi", "future_metadata"} & set(supplied):
            raise ValueError("final KPIs cannot be transition provenance")
        bound = {
            **supplied,
            "source_stage_id": record.stage_id,
            "stage": record.stage_type.value,
            "source_stage_fingerprint": source_stage_fingerprint(record),
            "control_names": tuple(record.controls),
            "scalar_observation_names": tuple(record.intermediate_properties),
            "modality_bindings": {modality.modality_id: modality_bindings[modality.modality_id] for modality in record.modalities},
            "encoder_fingerprint": encoder.fingerprint,
            "control_feature_names": controls.feature_names,
            "observation_feature_names": observations.feature_names,
            "encoded_control_fingerprint": controls.fingerprint,
            "encoded_observation_fingerprint": observations.fingerprint,
        }
        return cls(record.stage_id, record.stage_type, controls.tensor, observations.tensor, tuple(record.modalities), bound, tokens, availability)

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
        test_only: bool = False,
    ) -> "LegalStageTransition":
        if not test_only:
            raise ValueError("arbitrary stage tensors are test-only; use a frozen training StageFeatureEncoder")
        tokens, availability = cls._modality_payload(record, modality_inputs, modality_bindings)
        supplied = dict(provenance or {})
        reserved = {"source_stage_id", "stage", "source_stage_fingerprint", "control_names", "scalar_observation_names", "modality_bindings"}
        if reserved & set(supplied):
            raise ValueError("source-bound transition provenance fields cannot be overridden")
        if {"final_kpi", "final_kpis", "final_outputs", "future_kpi", "future_metadata"} & set(supplied):
            raise ValueError("final KPIs cannot be transition provenance")
        bound = {
            **supplied,
            "source_stage_id": record.stage_id,
            "stage": record.stage_type.value,
            "source_stage_fingerprint": source_stage_fingerprint(record),
            "control_names": tuple(record.controls),
            "scalar_observation_names": tuple(record.intermediate_properties),
            "modality_bindings": {modality.modality_id: modality_bindings[modality.modality_id] for modality in record.modalities},
            "unsafe_test_only": True,
        }
        return cls(record.stage_id, record.stage_type, controls, scalar_observations, tuple(record.modalities), bound, tokens, availability)
