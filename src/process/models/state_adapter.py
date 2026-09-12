from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Sequence

import numpy as np
import torch

from ..information_horizon import HorizonView
from ..modalities import ModalitySlotSpec
from ..optimization.state import ModelValidationStatus, OptimizationState, context_provenance_fingerprint
from ..stages import stage_precedes
from .artifact import MASPOModelArtifact
from .maspo import MASPOProcessStateModel
from .transitions import LegalStageTransition, SourceBackedInitialState, StageFeatureEncoder, source_stage_fingerprint


StageTransition = LegalStageTransition


def build_legal_stage_transitions(
    horizon: HorizonView,
    *,
    feature_encoder: StageFeatureEncoder | None = None,
    modality_inputs: Mapping[str, torch.Tensor],
    modality_bindings: Mapping[str, str] | None = None,
    modality_slots: Sequence[ModalitySlotSpec] | None = None,
    stage_inputs: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    test_only: bool = False,
) -> tuple[LegalStageTransition, ...]:
    """Encode source StageRecords; arbitrary tensors are retained only for explicit tests."""
    if tuple(record.stage_id for record in horizon.source_stages) != tuple(horizon.source_stage_ids):
        raise ValueError("HorizonView must carry source StageRecords in source_stage_ids order")
    if not test_only and modality_slots is None:
        raise ValueError("production stage transitions require semantic modality slots")
    if stage_inputs is not None:
        if not test_only:
            raise ValueError("arbitrary stage tensors are test-only; pass test_only=True explicitly")
        if set(stage_inputs) != set(horizon.source_stage_ids):
            raise ValueError("stage inputs must cover exactly the legal source stages")
        return tuple(
            LegalStageTransition.from_source_stage(
                record, controls=stage_inputs[record.stage_id][0],
                scalar_observations=stage_inputs[record.stage_id][1],
                modality_inputs=modality_inputs, modality_bindings=modality_bindings,
                modality_slots=modality_slots, test_only=True,
            )
            for record in horizon.source_stages
        )
    if feature_encoder is None:
        raise ValueError("production stage transitions require a StageFeatureEncoder")
    return tuple(
        LegalStageTransition.from_encoded_source_stage(
            record, encoder=feature_encoder,
            modality_inputs=modality_inputs, modality_bindings=modality_bindings,
            modality_slots=modality_slots,
        )
        for record in horizon.source_stages
    )


def _validate_legal_history(
    horizon: HorizonView, transitions: tuple[LegalStageTransition, ...],
    feature_encoder: StageFeatureEncoder | None, modality_slots: Sequence[ModalitySlotSpec] | None = None,
) -> None:
    expected_ids = tuple(horizon.source_stage_ids)
    if tuple(item.source_stage_id for item in transitions) != expected_ids:
        raise ValueError("stage history must contain the legal source stages in HorizonView order")
    source_records = {record.stage_id: record for record in horizon.source_stages}
    if set(source_records) != set(expected_ids):
        raise ValueError("HorizonView source stages are incomplete or not source-bound")
    expected_controls = {
        f"{record.stage_type.value.lower()}.{name}": value
        for record in horizon.source_stages for name, value in record.controls.items()
    }
    expected_properties = {
        f"{record.stage_type.value.lower()}.{name}": value
        for record in horizon.source_stages for name, value in record.intermediate_properties.items()
    }
    if dict(horizon.controls) != expected_controls or dict(horizon.intermediate_properties) != expected_properties:
        raise ValueError("HorizonView contains controls or metadata not present in its source stages")
    for transition in transitions:
        record = source_records[transition.source_stage_id]
        if transition.stage != record.stage_type or not stage_precedes(transition.stage, horizon.decision_stage):
            raise ValueError("legal transition stage does not match its source stage or decision horizon")
        if transition.provenance.get("source_stage_fingerprint") != source_stage_fingerprint(record):
            raise ValueError("legal transition source stage fingerprint is not bound to the HorizonView source record")
        if tuple(transition.modality_observations) != tuple(record.modalities):
            raise ValueError("transition modalities must come from the same source stage record")
        if tuple(transition.provenance.get("control_names", ())) != tuple(record.controls):
            raise ValueError("transition controls are not bound to the source stage schema")
        if tuple(transition.provenance.get("scalar_observation_names", ())) != tuple(record.intermediate_properties):
            raise ValueError("transition scalar observations are not bound to the source stage schema")
        bindings = transition.provenance.get("modality_bindings", {})
        expected_bindings = {modality.modality_id for modality in record.modalities}
        if not isinstance(bindings, Mapping) or set(bindings) != expected_bindings:
            raise ValueError("transition modality provenance is not bound to the source stage")
        if modality_slots is not None:
            resolved, relevant = LegalStageTransition._resolve_modality_slots(record, tuple(modality_slots))
            expected_slots = tuple(slot.as_tuple() for slot in relevant)
            if transition.provenance.get("semantic_modality_slots") != expected_slots:
                raise ValueError("transition semantic modality schema does not match the model artifact")
            if dict(bindings) != {modality_id: slot.model_input_name for modality_id, slot in resolved.items()}:
                raise ValueError("transition modality bindings are not resolved from the semantic slot schema")
            expected_availability = {slot.model_input_name: False for slot in relevant}
            modalities_by_id = {modality.modality_id: modality for modality in record.modalities}
            for modality_id, slot in resolved.items():
                if modalities_by_id[modality_id].is_missing and not slot.allowed_missing:
                    raise ValueError(f"required semantic modality slot {slot.slot_name!r} is explicitly missing")
                expected_availability[slot.model_input_name] = not modalities_by_id[modality_id].is_missing
            if dict(transition.availability) != expected_availability:
                raise ValueError("transition modality availability is not bound to the semantic slot schema")
            for slot_name, value in transition.modality_inputs.items():
                slot = next((item for item in relevant if item.model_input_name == slot_name), None)
                if slot is None:
                    raise ValueError("transition contains an unknown semantic model input")
                LegalStageTransition._validate_modality_input(value, slot)
        if feature_encoder is not None:
            feature_encoder.validate_transition(record, transition)
    expected_modalities = tuple(modality.modality_id for record in horizon.source_stages for modality in record.modalities)
    if tuple(modality.modality_id for modality in horizon.modalities) != expected_modalities:
        raise ValueError("HorizonView modalities are not source-stage bound")


def encode_legal_multimodal_state(
    horizon: HorizonView,
    trained_model: MASPOProcessStateModel | MASPOModelArtifact,
    *,
    stage_history: Iterable[LegalStageTransition] | None = None,
    feature_encoder: StageFeatureEncoder | None = None,
    stage_inputs: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    modality_inputs: Mapping[str, torch.Tensor] | None = None,
    modality_bindings: Mapping[str, str] | None = None,
    initial_state: SourceBackedInitialState | torch.Tensor | None = None,
    model_version: str | None = None,
    model_fingerprint: str | None = None,
    validation_status: ModelValidationStatus | None = None,
    training_evidence_id: str | None = None,
    dataset_fingerprint: str | None = None,
) -> OptimizationState:
    """Encode a source-bound legal history through a frozen MASPO model artifact."""
    artifact = trained_model if isinstance(trained_model, MASPOModelArtifact) else None
    model = artifact.model if artifact is not None else trained_model
    modality_slots: Sequence[ModalitySlotSpec] | None = None
    if artifact is None and validation_status != ModelValidationStatus.TEST_ONLY:
        raise ValueError("production multimodal inference requires a MASPOModelArtifact")
    if artifact is not None:
        artifact.verify_model_integrity()
        modality_slots = artifact.semantic_modality_slots
        if modality_bindings is not None:
            raise ValueError("production inference resolves source modalities through semantic modality slots")
        if feature_encoder is not None and feature_encoder.fingerprint != artifact.encoder_fingerprint:
            raise ValueError("inference encoder fingerprint does not match the model artifact")
        feature_encoder = artifact.stage_feature_encoder
        if model_version is not None and model_version != artifact.model_version:
            raise ValueError("inference model version does not match the model artifact")
        if model_fingerprint is not None and model_fingerprint != artifact.model_fingerprint:
            raise ValueError("inference model fingerprint does not match the model artifact")
        model_version = artifact.model_version
        model_fingerprint = artifact.model_fingerprint
        artifact_status = ModelValidationStatus(artifact.validation_status)
        if validation_status is not None and ModelValidationStatus(validation_status) != artifact_status:
            raise ValueError("inference validation status does not match the model artifact")
        status = artifact_status
        if training_evidence_id is None:
            training_evidence_id = artifact.training_evidence_id
        if dataset_fingerprint is None:
            dataset_fingerprint = artifact.dataset_fingerprint
    else:
        model_version = model_version or "test-only"
        model_fingerprint = model_fingerprint or "test-only"
        status = ModelValidationStatus(validation_status)
    if not model_version.strip() or not model_fingerprint.strip():
        raise ValueError("multimodal optimization state requires explicit model version and fingerprint")
    if status == ModelValidationStatus.SOURCE_BACKED_VALIDATED and (not training_evidence_id or not dataset_fingerprint):
        raise ValueError("SOURCE_BACKED_VALIDATED requires training evidence and dataset fingerprint")
    if stage_history is not None:
        stage_history = tuple(stage_history)
        if any(not isinstance(item, LegalStageTransition) for item in stage_history):
            raise TypeError("stage_history accepts only source-bound LegalStageTransition values")
    if artifact is not None and initial_state is not None and status != ModelValidationStatus.TEST_ONLY:
        raise ValueError("production inference uses the model artifact initial state")
    if initial_state is None:
        if artifact is None:
            raise ValueError("test-only multimodal encoding requires an explicit initial state")
        initial_tensor = artifact.initial_state()
        initial_state_fingerprint = artifact.initial_state_fingerprint
    elif isinstance(initial_state, SourceBackedInitialState):
        if initial_state.source_stage_ids != tuple(horizon.source_stage_ids):
            raise ValueError("initial state source IDs must match the HorizonView")
        if initial_state.model_fingerprint != model_fingerprint:
            raise ValueError("initial state model fingerprint does not match the requested model")
        if feature_encoder is not None and initial_state.encoder_fingerprint != feature_encoder.fingerprint:
            raise ValueError("initial state encoder fingerprint does not match the StageFeatureEncoder")
        initial_tensor = initial_state.tensor
        initial_state_fingerprint = initial_state.provenance_fingerprint
    elif isinstance(initial_state, torch.Tensor):
        if status != ModelValidationStatus.TEST_ONLY:
            raise ValueError("anonymous initial tensors are test-only; use SourceBackedInitialState")
        initial_tensor = initial_state
        initial_state_fingerprint = None
    else:
        raise TypeError("initial_state must be SourceBackedInitialState or a test-only tensor")
    if stage_history is None:
        if modality_inputs is None or (modality_bindings is None and modality_slots is None):
            raise ValueError("multimodal encoding requires source modalities")
        if feature_encoder is None and stage_inputs is None:
            raise ValueError("production multimodal encoding requires a StageFeatureEncoder")
        if stage_inputs is not None and status != ModelValidationStatus.TEST_ONLY:
            raise ValueError("arbitrary stage tensors are test-only; use a StageFeatureEncoder")
        transitions = build_legal_stage_transitions(
            horizon, feature_encoder=feature_encoder, modality_inputs=modality_inputs, modality_bindings=modality_bindings,
            modality_slots=modality_slots,
            stage_inputs=stage_inputs, test_only=status == ModelValidationStatus.TEST_ONLY,
        )
    else:
        if feature_encoder is None and status != ModelValidationStatus.TEST_ONLY:
            raise ValueError("stage history validation requires the source StageFeatureEncoder")
        transitions = tuple(stage_history)
    if any(not isinstance(item, LegalStageTransition) for item in transitions):
        raise TypeError("stage_history accepts only source-bound LegalStageTransition values")
    _validate_legal_history(horizon, transitions, feature_encoder, modality_slots)
    if initial_tensor.ndim not in (1, 2) or initial_tensor.shape[-1] != model.stage_model.stage_embedding.embedding_dim:
        raise ValueError("initial_state has the wrong shape for the MASPO StageAwareProcessModel")
    expected_observation_dim = model.stage_model.observation_dim - model.embedding_dim
    if feature_encoder is not None and (feature_encoder.control_dim != model.stage_model.control_dim or feature_encoder.observation_dim != expected_observation_dim):
        raise ValueError("StageFeatureEncoder dimensions do not match the MASPO model")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            state = model.state_from_transitions(initial_tensor, transitions) if transitions else initial_tensor
    finally:
        model.train(was_training)
    if state.ndim == 2:
        if state.shape[0] != 1:
            raise ValueError("multimodal optimization state must encode one current process state")
        state = state[0]
    values = state.detach().cpu().numpy().reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("multimodal stage model produced no finite optimizer state")

    availability = {modality.modality_id: not modality.is_missing for modality in horizon.modalities}
    bindings = {
        modality.modality_id: transition.provenance["modality_bindings"][modality.modality_id]
        for transition in transitions
        for modality in transition.modality_observations
    }
    semantic_slots = tuple(slot.as_tuple() for slot in artifact.semantic_modality_slots) if artifact is not None else tuple(sorted(
        (transition.stage.value, modality.modality_type.value, transition.provenance["modality_bindings"][modality.modality_id], not modality.is_missing)
        for transition in transitions for modality in transition.modality_observations
    ))
    semantic_metadata: dict[str, Any] = {
        "model_fingerprint": model_fingerprint,
        "model_version": model_version,
        "encoder_fingerprint": feature_encoder.fingerprint if feature_encoder is not None else "unsafe-test-only",
        "encoder_schema": feature_encoder.feature_schema if feature_encoder is not None else None,
        "encoder_dimensions": {"control": feature_encoder.control_dim, "observation": feature_encoder.observation_dim} if feature_encoder is not None else None,
        "model_artifact_fingerprint": artifact.artifact_fingerprint if artifact is not None else None,
        "initial_state_fingerprint": initial_state_fingerprint,
        "modality_schema": semantic_slots,
        "modality_schema_fingerprint": artifact.modality_schema_fingerprint if artifact is not None else None,
    }
    feature_names = tuple(f"context.state.{index}" for index in range(values.size))
    feature_values = {name: float(value) for name, value in zip(feature_names, values)}
    fingerprint = context_provenance_fingerprint(
        feature_values, horizon.decision_stage, "multimodal_stage_state", semantic_metadata=semantic_metadata,
    )
    return OptimizationState(
        feature_names=feature_names, feature_values=feature_values, decision_stage=horizon.decision_stage,
        source_stage_ids=horizon.source_stage_ids, provenance_fingerprint=fingerprint,
        representation_kind="multimodal_stage_state", provenance={
            "model_version": model_version, "model_fingerprint": model_fingerprint,
            "modality_ids": tuple(availability), "modality_availability": availability,
            "modality_bindings": bindings, "semantic_fingerprint_inputs": semantic_metadata,
            "latent_feature_schema": feature_names,
            "model_artifact_fingerprint": artifact.artifact_fingerprint if artifact is not None else None,
            "audit_provenance": {
                "source_stage_ids": horizon.source_stage_ids,
                "source_transition_fingerprints": tuple(item.provenance["source_stage_fingerprint"] for item in transitions),
                "initial_state_fingerprint": initial_state_fingerprint,
            },
            "training_evidence_id": training_evidence_id, "dataset_fingerprint": dataset_fingerprint,
            "validation_status": status.value,
        },
        validation_status=status,
    )
