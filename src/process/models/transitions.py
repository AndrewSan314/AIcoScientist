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
from ..modalities import (
    ModalityObservation,
    ModalitySlotSpec,
    SourceBoundModalityInput,
    source_modality_fingerprint,
    tensor_fingerprint,
)
from ..stages import ProcessStage, STAGE_ORDER


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
    allow_missing: bool = False

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
        optional_fields: Mapping[str, bool] | Sequence[str] | None = None,
    ) -> "StageFeatureEncoder":
        return cls.fit(
            horizon,
            category_vocabularies=category_vocabularies,
            control_dim=control_dim,
            observation_dim=observation_dim,
            optional_fields=optional_fields,
        )

    @classmethod
    def fit(
        cls,
        training_data: Any,
        *,
        category_vocabularies: Mapping[str, Sequence[object]] | None = None,
        control_dim: int | None = None,
        observation_dim: int | None = None,
        optional_fields: Mapping[str, bool] | Sequence[str] | None = None,
    ) -> "StageFeatureEncoder":
        records = cls._training_records(training_data)
        vocabularies = category_vocabularies or {}
        optional = cls._optional_fields(optional_fields)
        stage_types = tuple(sorted({record.stage_type for record in records}, key=STAGE_ORDER.__getitem__))
        stages = tuple(
            _StageSpec(
                stage_type,
                cls._field_specs(records, stage_type, "control", vocabularies, optional),
                cls._field_specs(records, stage_type, "observation", vocabularies, optional),
            )
            for stage_type in stage_types
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
                tuple((field.name, field.kind, field.categories, field.allow_missing) for field in stage.controls),
                tuple((field.name, field.kind, field.categories, field.allow_missing) for field in stage.observations),
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
    def _training_records(training_data: Any) -> tuple[StageRecord, ...]:
        if hasattr(training_data, "source_stages"):
            datasets = (training_data,)
        else:
            try:
                items = tuple(training_data)
            except TypeError:
                items = (training_data,)
            datasets = items if items and all(hasattr(item, "source_stages") for item in items) else ()
        if datasets:
            records = []
            for horizon in datasets:
                source_stages = tuple(horizon.source_stages)
                if tuple(record.stage_id for record in source_stages) != tuple(horizon.source_stage_ids):
                    raise ValueError("StageFeatureEncoder requires source stages in HorizonView order")
                records.extend(source_stages)
        else:
            records = list(items if "items" in locals() else (training_data,))
        if not records:
            raise ValueError("StageFeatureEncoder requires at least one training stage record")
        return tuple(records)

    @staticmethod
    def _optional_fields(optional_fields: Mapping[str, bool] | Sequence[str] | None) -> set[str]:
        if optional_fields is None:
            return set()
        if isinstance(optional_fields, Mapping):
            return {str(name) for name, allowed in optional_fields.items() if allowed}
        return {str(name) for name in optional_fields}

    @staticmethod
    def _field_specs(
        records: Sequence[StageRecord], stage_type: ProcessStage, scope: str,
        vocabularies: Mapping[str, Sequence[object]], optional_fields: set[str],
    ) -> tuple[_FieldSpec, ...]:
        sources = tuple(
            record.controls if scope == "control" else record.intermediate_properties
            for record in records if record.stage_type == stage_type
        )
        names = tuple(sorted({name for source in sources for name in source}))
        result: list[_FieldSpec] = []
        stage_name = stage_type.value.lower()
        for name in names:
            key = f"{stage_name}.{scope}.{name}"
            short_key = f"{stage_name}.{name}"
            vocabulary = vocabularies.get(key, vocabularies.get(short_key, vocabularies.get(name)))
            values = tuple(_field_value(source[name]) for source in sources if name in source)
            absent = len(values) != len(sources)
            explicitly_missing = any(_missing(value) for value in values)
            allow_missing = key in optional_fields or short_key in optional_fields or name in optional_fields or explicitly_missing
            if absent and not allow_missing:
                raise ValueError(f"inconsistent {stage_type.value} {scope} schema for {key!r}; declare it optional")
            observed = tuple(value for value in values if not _missing(value))
            if not observed and vocabulary is None:
                raise ValueError(f"cannot infer the kind of source field {key!r} from missing-only training data")
            kinds = {"numeric" if isinstance(value, Real) and not isinstance(value, bool) else "categorical" for value in observed}
            if len(kinds) > 1:
                raise ValueError(f"inconsistent {stage_type.value} {scope} schema for {key!r}: numeric and categorical values disagree")
            kind = next(iter(kinds), "categorical" if vocabulary is not None else "numeric")
            if kind == "categorical":
                category_values = ((*vocabulary,) if vocabulary is not None else ()) + observed
                if allow_missing:
                    category_values += (None,)
                categories = tuple(sorted({
                    _category(value) for value in category_values
                    if allow_missing or not _missing(value)
                }))
                if not categories or any(_category(value) not in categories for value in values):
                    raise ValueError(f"categorical source field {key!r} is unknown to its vocabulary")
                result.append(_FieldSpec(key, kind, categories, allow_missing))
            else:
                if any(not math.isfinite(float(value)) for value in observed):
                    raise ValueError(f"source field {key!r} contains a non-finite number")
                result.append(_FieldSpec(key, kind, (), allow_missing))
        return tuple(result)

    def _stage(self, record: StageRecord) -> _StageSpec:
        matches = [stage for stage in self.stages if stage.stage_type == record.stage_type]
        if len(matches) != 1:
            raise ValueError(f"stage type {record.stage_type.value!r} is not uniquely represented by the encoder")
        stage = matches[0]
        control_names = tuple(f"{record.stage_type.value.lower()}.control.{name}" for name in sorted(record.controls))
        observation_names = tuple(f"{record.stage_type.value.lower()}.observation.{name}" for name in sorted(record.intermediate_properties))
        for specs, names, scope in ((stage.controls, control_names, "control"), (stage.observations, observation_names, "observation")):
            expected = tuple(field.name for field in specs)
            extra = tuple(name for name in names if name not in expected)
            missing = tuple(field for field in specs if field.name not in names and not field.allow_missing)
            if extra or missing:
                raise ValueError("source stage schema does not match the StageFeatureEncoder")
        return stage

    def _encode_scope(self, record: StageRecord, scope: str) -> EncodedStageFeatures:
        stage = self._stage(record)
        specs = stage.controls if scope == "control" else stage.observations
        source = record.controls if scope == "control" else record.intermediate_properties
        values: list[float] = []
        names: list[str] = []
        for spec in specs:
            field_name = spec.name.rsplit(".", 1)[-1]
            raw = _field_value(source[field_name]) if field_name in source else None
            if _missing(raw) and not spec.allow_missing:
                raise ValueError(f"source stage field {spec.name!r} is missing but not optional")
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
    def _resolve_modality_slots(
        record: StageRecord, modality_slots: Sequence[ModalitySlotSpec],
    ) -> tuple[dict[str, ModalitySlotSpec], tuple[ModalitySlotSpec, ...]]:
        relevant = tuple(sorted((slot for slot in modality_slots if slot.stage == record.stage_type), key=ModalitySlotSpec.as_tuple))
        resolved: dict[str, ModalitySlotSpec] = {}
        for modality in record.modalities:
            matches = tuple(slot for slot in relevant if slot.matches(modality))
            if len(matches) != 1:
                reason = "no semantic slot" if not matches else "ambiguous semantic slots"
                raise ValueError(f"source modality {modality.modality_id!r} has {reason}")
            if modality.modality_id in resolved or matches[0].model_input_name in {slot.model_input_name for slot in resolved.values()}:
                raise ValueError(f"source modality {modality.modality_id!r} does not resolve uniquely")
            resolved[modality.modality_id] = matches[0]
        for slot in relevant:
            if slot.model_input_name not in {item.model_input_name for item in resolved.values()} and slot.required:
                raise ValueError(f"required semantic modality slot {slot.slot_name!r} is missing")
        return resolved, relevant

    @staticmethod
    def _validate_modality_input(value: torch.Tensor, slot: ModalitySlotSpec) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"modality input for slot {slot.slot_name!r} must be a torch.Tensor")
        if value.ndim < 1 or value.shape[-1] != slot.expected_input_dim:
            raise ValueError(
                f"modality input for slot {slot.slot_name!r} has dimension {tuple(value.shape)}; "
                f"expected final dimension {slot.expected_input_dim}"
            )
        if slot.expected_shape is not None and tuple(value.shape) != slot.expected_shape:
            raise ValueError(f"modality input for slot {slot.slot_name!r} does not match expected shape {slot.expected_shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"modality input for slot {slot.slot_name!r} contains non-finite values")

    @staticmethod
    def _modality_payload(
        record: StageRecord, modality_inputs: Mapping[str, SourceBoundModalityInput | torch.Tensor] | None,
        modality_bindings: Mapping[str, str] | None = None,
        modality_slots: Sequence[ModalitySlotSpec] | None = None,
        *, test_only: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, bool], dict[str, str], tuple[ModalitySlotSpec, ...], dict[str, dict[str, Any]]]:
        tokens: dict[str, torch.Tensor] = {}
        availability: dict[str, bool] = {}
        modality_provenance: dict[str, dict[str, Any]] = {}
        if modality_slots is not None:
            resolved, relevant = LegalStageTransition._resolve_modality_slots(record, tuple(modality_slots))
            bindings = {modality_id: slot.model_input_name for modality_id, slot in resolved.items()}
            if modality_inputs is None:
                modality_inputs = {
                    modality.modality_id: SourceBoundModalityInput.from_observation(modality, resolved[modality.modality_id])
                    for modality in record.modalities if not modality.is_missing
                }
            extra = set(modality_inputs) - {modality.modality_id for modality in record.modalities}
            if extra:
                raise ValueError(f"modality inputs contain unknown source IDs: {sorted(extra)}")
            for modality in record.modalities:
                slot = resolved[modality.modality_id]
                if modality.is_missing and not slot.allowed_missing:
                    raise ValueError(f"required semantic modality slot {slot.slot_name!r} is explicitly missing")
                availability[slot.model_input_name] = not modality.is_missing
                if not modality.is_missing:
                    if modality.modality_id not in modality_inputs:
                        raise ValueError(f"source modality {modality.modality_id!r} lacks encoder input")
                    bound = modality_inputs[modality.modality_id]
                    if not test_only:
                        if not isinstance(bound, SourceBoundModalityInput):
                            raise ValueError("production modality inputs must be SourceBoundModalityInput values")
                        expected_source_fingerprint = source_modality_fingerprint(modality)
                        if bound.source_modality_id != modality.modality_id or bound.source_modality_fingerprint != expected_source_fingerprint:
                            raise ValueError("encoded modality is not bound to the exact source observation")
                        if bound.slot_name != slot.slot_name or bound.model_input_name != slot.model_input_name:
                            raise ValueError("encoded modality semantic slot binding is invalid")
                        if bound.preprocessing_fingerprint != slot.effective_preprocessing_fingerprint:
                            raise ValueError("encoded modality preprocessing fingerprint is invalid")
                        if bound.tensor_fingerprint != tensor_fingerprint(bound.tensor):
                            raise ValueError("encoded modality tensor was tampered after binding")
                        value = bound.tensor
                        modality_provenance[modality.modality_id] = {
                            "source_modality_id": bound.source_modality_id,
                            "source_modality_fingerprint": bound.source_modality_fingerprint,
                            "semantic_slot": bound.slot_name,
                            "model_input_name": bound.model_input_name,
                            "preprocessing_fingerprint": bound.preprocessing_fingerprint,
                            "encoded_tensor_fingerprint": bound.tensor_fingerprint,
                            "source_kind": bound.source_kind,
                        }
                    else:
                        value = bound.tensor if isinstance(bound, SourceBoundModalityInput) else bound
                    LegalStageTransition._validate_modality_input(value, slot)
                    tokens[slot.model_input_name] = value
            for slot in relevant:
                availability.setdefault(slot.model_input_name, False)
            return tokens, availability, bindings, relevant, modality_provenance
        if modality_bindings is None:
            raise ValueError("source modality encoding requires semantic modality slots")
        if modality_inputs is None:
            raise ValueError("test-only modality encoding requires explicit tensors")
        bindings = {}
        for modality in record.modalities:
            model_name = modality_bindings.get(modality.modality_id)
            if model_name is None or model_name in availability:
                raise ValueError(f"source modality {modality.modality_id!r} lacks a unique model binding")
            availability[model_name] = not modality.is_missing
            bindings[modality.modality_id] = model_name
            if not modality.is_missing:
                if modality.modality_id not in modality_inputs:
                    raise ValueError(f"source modality {modality.modality_id!r} lacks encoder input")
                value = modality_inputs[modality.modality_id]
                tokens[model_name] = value.tensor if isinstance(value, SourceBoundModalityInput) else value
        return tokens, availability, bindings, (), modality_provenance

    @classmethod
    def from_encoded_source_stage(
        cls,
        record: StageRecord,
        *,
        encoder: StageFeatureEncoder,
        modality_inputs: Mapping[str, SourceBoundModalityInput | torch.Tensor] | None = None,
        modality_bindings: Mapping[str, str] | None = None,
        modality_slots: Sequence[ModalitySlotSpec] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "LegalStageTransition":
        controls = encoder.encode_controls(record)
        observations = encoder.encode_observations(record)
        tokens, availability, bindings, resolved_slots, modality_provenance = cls._modality_payload(
            record, modality_inputs, modality_bindings, modality_slots,
            test_only=False,
        )
        supplied = dict(provenance or {})
        reserved = {
            "source_stage_id", "stage", "source_stage_fingerprint", "control_names", "scalar_observation_names", "modality_bindings", "semantic_modality_slots",
            "encoder_fingerprint", "control_feature_names", "observation_feature_names", "encoded_control_fingerprint", "encoded_observation_fingerprint", "modality_provenance",
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
            "modality_bindings": bindings,
            "semantic_modality_slots": tuple(slot.as_tuple() for slot in resolved_slots),
            "modality_provenance": modality_provenance,
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
        modality_inputs: Mapping[str, SourceBoundModalityInput | torch.Tensor],
        modality_bindings: Mapping[str, str] | None = None,
        modality_slots: Sequence[ModalitySlotSpec] | None = None,
        provenance: Mapping[str, Any] | None = None,
        test_only: bool = False,
    ) -> "LegalStageTransition":
        if not test_only:
            raise ValueError("arbitrary stage tensors are test-only; use a frozen training StageFeatureEncoder")
        tokens, availability, bindings, resolved_slots, _ = cls._modality_payload(
            record, modality_inputs, modality_bindings, modality_slots,
            test_only=True,
        )
        supplied = dict(provenance or {})
        reserved = {"source_stage_id", "stage", "source_stage_fingerprint", "control_names", "scalar_observation_names", "modality_bindings", "semantic_modality_slots"}
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
            "modality_bindings": bindings,
            "semantic_modality_slots": tuple(slot.as_tuple() for slot in resolved_slots),
            "unsafe_test_only": True,
        }
        return cls(record.stage_id, record.stage_type, controls, scalar_observations, tuple(record.modalities), bound, tokens, availability)
