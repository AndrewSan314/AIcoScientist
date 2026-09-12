from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from ..modalities import ModalitySlotSpec
from ..optimization.state import ModelValidationStatus
from .maspo import MASPOProcessStateModel
from .transitions import StageFeatureEncoder


@dataclass(frozen=True)
class MASPOModelArtifact:
    """Model, encoder, and semantic modality contract consumed by production inference."""

    model: MASPOProcessStateModel
    model_version: str
    model_weight_fingerprint: str
    stage_feature_encoder: StageFeatureEncoder
    encoder_fingerprint: str
    encoder_schema: tuple[Any, ...]
    semantic_modality_slots: tuple[ModalitySlotSpec, ...]
    category_vocabularies: Mapping[str, tuple[tuple[str, str], ...]]
    control_dim: int
    observation_dim: int
    state_dim: int
    embedding_dim: int
    initial_state_spec: str
    initial_state_fingerprint: str
    training_evidence_id: str | None = None
    dataset_fingerprint: str | None = None
    validation_status: ModelValidationStatus = ModelValidationStatus.TRAINED_UNVALIDATED
    modality_schema_fingerprint: str = field(init=False)
    artifact_fingerprint: str = field(init=False)

    @classmethod
    def from_training(
        cls,
        model: MASPOProcessStateModel,
        *,
        feature_encoder: StageFeatureEncoder,
        model_version: str,
        semantic_modality_slots: Sequence[ModalitySlotSpec] | None = None,
        expected_model_fingerprint: str | None = None,
        modality_schema: Sequence[ModalitySlotSpec] | None = None,
        category_vocabularies: Mapping[str, Sequence[object]] | None = None,
        training_evidence_id: str | None = None,
        dataset_fingerprint: str | None = None,
        validation_status: ModelValidationStatus = ModelValidationStatus.TRAINED_UNVALIDATED,
    ) -> "MASPOModelArtifact":
        if not model_version.strip():
            raise ValueError("model artifact requires an explicit model version")
        slots = tuple(semantic_modality_slots if semantic_modality_slots is not None else (modality_schema or ()))
        if not slots:
            raise ValueError("model artifact requires semantic modality slots")
        if not all(isinstance(slot, ModalitySlotSpec) for slot in slots):
            raise TypeError("model artifact semantic_modality_slots must contain ModalitySlotSpec values")
        slots = tuple(sorted(slots, key=ModalitySlotSpec.as_tuple))
        actual_model_fingerprint = model_weight_fingerprint(model)
        if expected_model_fingerprint is not None:
            if not expected_model_fingerprint.strip() or expected_model_fingerprint != actual_model_fingerprint:
                raise ValueError("expected_model_fingerprint does not match the model state_dict")
        status = ModelValidationStatus(validation_status)
        if status == ModelValidationStatus.SOURCE_BACKED_VALIDATED and (not training_evidence_id or not dataset_fingerprint):
            raise ValueError("SOURCE_BACKED_VALIDATED requires training evidence and dataset fingerprint")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        initial_fingerprint = _tensor_fingerprint(model.initial_state.detach())
        vocabularies = feature_encoder.category_vocabularies if category_vocabularies is None else category_vocabularies
        normalized_vocabularies = {
            str(name): tuple(
                item if isinstance(item, tuple) and len(item) == 2 else _category_token(item)
                for item in values
            )
            for name, values in vocabularies.items()
        }
        return cls(
            model=model, model_version=model_version, model_weight_fingerprint=actual_model_fingerprint,
            stage_feature_encoder=feature_encoder, encoder_fingerprint=feature_encoder.fingerprint,
            encoder_schema=feature_encoder.feature_schema, semantic_modality_slots=slots,
            category_vocabularies=normalized_vocabularies, control_dim=feature_encoder.control_dim,
            observation_dim=feature_encoder.observation_dim,
            state_dim=model.stage_model.stage_embedding.embedding_dim, embedding_dim=model.embedding_dim,
            initial_state_spec="model_owned:MASPOProcessStateModel.initial_state",
            initial_state_fingerprint=initial_fingerprint, training_evidence_id=training_evidence_id,
            dataset_fingerprint=dataset_fingerprint, validation_status=status,
        )

    def __post_init__(self) -> None:
        if not self.model_version.strip() or not self.model_weight_fingerprint.strip() or not self.semantic_modality_slots:
            raise ValueError("model artifact requires model identity and semantic modality slots")
        if not all(isinstance(slot, ModalitySlotSpec) for slot in self.semantic_modality_slots):
            raise TypeError("model artifact semantic_modality_slots must contain ModalitySlotSpec values")
        slots = tuple(sorted(self.semantic_modality_slots, key=ModalitySlotSpec.as_tuple))
        object.__setattr__(self, "semantic_modality_slots", slots)
        if self.encoder_fingerprint != self.stage_feature_encoder.fingerprint:
            raise ValueError("model artifact encoder fingerprint is invalid")
        if self.encoder_schema != self.stage_feature_encoder.feature_schema:
            raise ValueError("model artifact encoder schema is invalid")
        if dict(self.category_vocabularies) != self.stage_feature_encoder.category_vocabularies:
            raise ValueError("model artifact category vocabulary is not the frozen encoder vocabulary")
        if len({slot.slot_name for slot in slots}) != len(slots):
            raise ValueError("model artifact semantic modality slot names must be unique")
        if len({(slot.stage, slot.model_input_name) for slot in slots}) != len(slots):
            raise ValueError("model artifact cannot bind two slots to one model input at one stage")
        for slot in slots:
            if slot.model_input_name not in self.model.modality_encoders:
                raise ValueError(f"model artifact slot references unknown model input {slot.model_input_name!r}")
            expected_dim = _model_input_dim(self.model, slot.model_input_name)
            if expected_dim != slot.expected_input_dim:
                raise ValueError(
                    f"model artifact slot {slot.slot_name!r} expects input dimension {slot.expected_input_dim}, "
                    f"but model input {slot.model_input_name!r} expects {expected_dim}"
                )
        if self.control_dim != self.stage_feature_encoder.control_dim or self.observation_dim != self.stage_feature_encoder.observation_dim:
            raise ValueError("model artifact dimensions do not match its frozen encoder")
        if self.control_dim != self.model.stage_model.control_dim or self.observation_dim + self.embedding_dim != self.model.stage_model.observation_dim:
            raise ValueError("model artifact dimensions do not match its model")
        if self.state_dim != self.model.stage_model.stage_embedding.embedding_dim or self.embedding_dim != self.model.embedding_dim:
            raise ValueError("model artifact state dimensions do not match its model")
        if self.initial_state_spec != "model_owned:MASPOProcessStateModel.initial_state":
            raise ValueError("model artifact must use its model-owned initial state")
        if self.initial_state_fingerprint != _tensor_fingerprint(self.model.initial_state.detach()):
            raise ValueError("model artifact initial state fingerprint does not match its model")
        actual_model_fingerprint = model_weight_fingerprint(self.model)
        if self.model_weight_fingerprint != actual_model_fingerprint:
            raise ValueError("model artifact model_weight_fingerprint does not match its model state_dict")
        modality_schema = tuple(slot.as_dict() for slot in slots)
        object.__setattr__(self, "modality_schema_fingerprint", _sha256_json(modality_schema))
        payload = {
            "model_version": self.model_version, "model_weight_fingerprint": self.model_weight_fingerprint,
            "encoder_fingerprint": self.encoder_fingerprint, "encoder_schema": self.encoder_schema,
            "semantic_modality_slots": modality_schema,
            "modality_schema_fingerprint": self.modality_schema_fingerprint,
            "category_vocabularies": self.category_vocabularies,
            "dims": (self.control_dim, self.observation_dim, self.state_dim, self.embedding_dim),
            "initial_state_spec": self.initial_state_spec, "initial_state_fingerprint": self.initial_state_fingerprint,
            "training_evidence_id": self.training_evidence_id, "dataset_fingerprint": self.dataset_fingerprint,
            "validation_status": ModelValidationStatus(self.validation_status).value,
        }
        object.__setattr__(self, "artifact_fingerprint", _sha256_json(payload))

    def verify_model_integrity(self) -> None:
        """Reject inference if the live model no longer matches this artifact."""
        actual = model_weight_fingerprint(self.model)
        if actual != self.model_weight_fingerprint:
            raise RuntimeError("MASPO model artifact integrity check failed: model weights changed after artifact construction")

    def initial_state(self) -> torch.Tensor:
        """Return the model-owned initial state; callers cannot replace it in production."""
        return self.model.initial_state.detach().clone()

    @property
    def model_fingerprint(self) -> str:
        """Backward-compatible audit alias for the computed weight fingerprint."""
        return self.model_weight_fingerprint

    @property
    def computed_model_weight_fingerprint(self) -> str:
        return self.model_weight_fingerprint

    @property
    def modality_schema(self) -> tuple[ModalitySlotSpec, ...]:
        """Backward-compatible name for the frozen semantic slot schema."""
        return self.semantic_modality_slots

    @property
    def encoder(self) -> StageFeatureEncoder:
        return self.stage_feature_encoder

    @property
    def fingerprint(self) -> str:
        return self.artifact_fingerprint

    @property
    def initial_state_semantics(self) -> str:
        return self.initial_state_spec

    @property
    def dimensions(self) -> dict[str, int]:
        return {
            "control": self.control_dim, "observation": self.observation_dim,
            "state": self.state_dim, "embedding": self.embedding_dim,
        }


def model_weight_fingerprint(model: torch.nn.Module) -> str:
    """Hash every dense state_dict entry without relying on Python hash randomization."""
    try:
        state = model.state_dict()
    except Exception as exc:  # pragma: no cover - defensive boundary for custom modules
        raise TypeError("model state_dict could not be read for deterministic fingerprinting") from exc
    digest = hashlib.sha256()
    digest.update(b"MASPO_MODEL_STATE_DICT_SHA256\0")
    _update_bytes(digest, str(len(state)).encode("utf-8"))
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state_dict entry {key!r} is not a tensor")
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError(f"model state_dict entry {key!r} uses an unsupported tensor layout or quantization")
        try:
            cpu_value = value.detach().cpu().contiguous()
            raw = cpu_value.reshape(-1).view(torch.uint8).numpy().tobytes()
        except Exception as exc:
            raise TypeError(f"model state_dict entry {key!r} cannot be converted to canonical CPU bytes") from exc
        _update_bytes(digest, key.encode("utf-8"))
        _update_bytes(digest, str(cpu_value.dtype).encode("utf-8"))
        _update_bytes(digest, _canonical_json(tuple(cpu_value.shape)).encode("utf-8"))
        _update_bytes(digest, raw)
    return digest.hexdigest()


def _model_input_dim(model: MASPOProcessStateModel, name: str) -> int:
    for module in model.modality_encoders[name].modules():
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)
    raise TypeError(f"model modality encoder {name!r} has no Linear input contract")


def _tensor_fingerprint(value: torch.Tensor) -> str:
    return _sha256_json({"dtype": str(value.dtype), "shape": tuple(value.shape), "values": value.cpu().tolist()})


def _category_token(value: object) -> tuple[str, str]:
    return ("missing", "") if value is None else ("value", str(value))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _update_bytes(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
