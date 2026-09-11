from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from ..optimization.state import ModelValidationStatus
from .maspo import MASPOProcessStateModel
from .transitions import StageFeatureEncoder


@dataclass(frozen=True)
class MASPOModelArtifact:
    """Frozen model/encoder contract consumed by production MASPO inference."""

    model: MASPOProcessStateModel
    model_version: str
    model_fingerprint: str
    stage_feature_encoder: StageFeatureEncoder
    encoder_fingerprint: str
    encoder_schema: tuple[Any, ...]
    modality_bindings: Mapping[str, str]
    modality_schema: tuple[Any, ...]
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
    artifact_fingerprint: str = field(init=False)

    @classmethod
    def from_training(
        cls,
        model: MASPOProcessStateModel,
        *,
        feature_encoder: StageFeatureEncoder,
        model_version: str,
        model_fingerprint: str,
        modality_bindings: Mapping[str, str],
        modality_schema: Sequence[Any] = (),
        category_vocabularies: Mapping[str, Sequence[object]] | None = None,
        training_evidence_id: str | None = None,
        dataset_fingerprint: str | None = None,
        validation_status: ModelValidationStatus = ModelValidationStatus.TRAINED_UNVALIDATED,
    ) -> "MASPOModelArtifact":
        if not model_version.strip() or not model_fingerprint.strip():
            raise ValueError("model artifact requires explicit model version and fingerprint")
        if not modality_bindings:
            raise ValueError("model artifact requires modality bindings")
        status = ModelValidationStatus(validation_status)
        if status == ModelValidationStatus.SOURCE_BACKED_VALIDATED and (not training_evidence_id or not dataset_fingerprint):
            raise ValueError("SOURCE_BACKED_VALIDATED requires training evidence and dataset fingerprint")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        initial = model.initial_state.detach()
        initial_fingerprint = _tensor_fingerprint(initial)
        vocabularies = category_vocabularies or feature_encoder.category_vocabularies
        normalized_vocabularies = {
            str(name): tuple(
                item if isinstance(item, tuple) and len(item) == 2 else _category_token(item)
                for item in values
            )
            for name, values in vocabularies.items()
        }
        return cls(
            model=model, model_version=model_version, model_fingerprint=model_fingerprint,
            stage_feature_encoder=feature_encoder, encoder_fingerprint=feature_encoder.fingerprint,
            encoder_schema=feature_encoder.feature_schema,
            modality_bindings=dict(modality_bindings),
            modality_schema=tuple(modality_schema), category_vocabularies=normalized_vocabularies,
            control_dim=feature_encoder.control_dim, observation_dim=feature_encoder.observation_dim,
            state_dim=model.stage_model.stage_embedding.embedding_dim, embedding_dim=model.embedding_dim,
            initial_state_spec="model_owned:MASPOProcessStateModel.initial_state",
            initial_state_fingerprint=initial_fingerprint, training_evidence_id=training_evidence_id,
            dataset_fingerprint=dataset_fingerprint, validation_status=status,
        )

    def __post_init__(self) -> None:
        if not self.model_version.strip() or not self.model_fingerprint.strip() or not self.modality_bindings:
            raise ValueError("model artifact requires model identity and modality bindings")
        if self.encoder_fingerprint != self.stage_feature_encoder.fingerprint:
            raise ValueError("model artifact encoder fingerprint is invalid")
        if self.encoder_schema != self.stage_feature_encoder.feature_schema:
            raise ValueError("model artifact encoder schema is invalid")
        if dict(self.category_vocabularies) != self.stage_feature_encoder.category_vocabularies:
            raise ValueError("model artifact category vocabulary is not the frozen encoder vocabulary")
        if set(self.modality_bindings.values()) - set(self.model.modality_encoders):
            raise ValueError("model artifact modality binding references an unknown model modality")
        if self.control_dim != self.stage_feature_encoder.control_dim or self.observation_dim != self.stage_feature_encoder.observation_dim:
            raise ValueError("model artifact dimensions do not match its frozen encoder")
        if self.control_dim != self.model.stage_model.control_dim or self.observation_dim + self.embedding_dim != self.model.stage_model.observation_dim:
            raise ValueError("model artifact dimensions do not match its model")
        if self.state_dim != self.model.stage_model.stage_embedding.embedding_dim or self.embedding_dim != self.model.embedding_dim:
            raise ValueError("model artifact state dimensions do not match its model")
        if self.initial_state_spec != "model_owned:MASPOProcessStateModel.initial_state":
            raise ValueError("model artifact must use its model-owned initial state")
        actual_initial_fingerprint = _tensor_fingerprint(self.model.initial_state.detach())
        if self.initial_state_fingerprint != actual_initial_fingerprint:
            raise ValueError("model artifact initial state fingerprint does not match its model")
        payload = {
            "model_version": self.model_version, "model_fingerprint": self.model_fingerprint,
            "encoder_fingerprint": self.encoder_fingerprint,
            "encoder_schema": self.encoder_schema,
            "modality_bindings": dict(sorted(self.modality_bindings.items())),
            "modality_schema": self.modality_schema,
            "category_vocabularies": self.category_vocabularies,
            "dims": (self.control_dim, self.observation_dim, self.state_dim, self.embedding_dim),
            "initial_state_spec": self.initial_state_spec,
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "training_evidence_id": self.training_evidence_id, "dataset_fingerprint": self.dataset_fingerprint,
            "validation_status": ModelValidationStatus(self.validation_status).value,
        }
        object.__setattr__(self, "artifact_fingerprint", hashlib.sha256(_canonical_json(payload).encode()).hexdigest())

    def initial_state(self) -> torch.Tensor:
        """Return the model-owned initial state; callers cannot replace it in production."""
        return self.model.initial_state.detach().clone()

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


def _tensor_fingerprint(value: torch.Tensor) -> str:
    payload = {"dtype": str(value.dtype), "shape": tuple(value.shape), "values": value.cpu().tolist()}
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _category_token(value: object) -> tuple[str, str]:
    return ("missing", "") if value is None else ("value", str(value))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
