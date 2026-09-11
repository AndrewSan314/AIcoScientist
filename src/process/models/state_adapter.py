from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

from ..information_horizon import HorizonView
from ..optimization.state import ModelValidationStatus, OptimizationState, context_provenance_fingerprint
from ..stages import stage_precedes
from .maspo import MASPOProcessStateModel
from .transitions import LegalStageTransition, source_stage_fingerprint


StageTransition = LegalStageTransition


def build_legal_stage_transitions(
    horizon: HorizonView,
    *,
    stage_inputs: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    modality_inputs: Mapping[str, torch.Tensor],
    modality_bindings: Mapping[str, str],
) -> tuple[LegalStageTransition, ...]:
    """Bind model tensors to the source stages already carried by a HorizonView."""
    if tuple(record.stage_id for record in horizon.source_stages) != tuple(horizon.source_stage_ids):
        raise ValueError("HorizonView must carry source StageRecords in source_stage_ids order")
    if set(stage_inputs) != set(horizon.source_stage_ids):
        raise ValueError("stage inputs must cover exactly the legal source stages")
    return tuple(
        LegalStageTransition.from_source_stage(
            record, controls=stage_inputs[record.stage_id][0],
            scalar_observations=stage_inputs[record.stage_id][1],
            modality_inputs=modality_inputs, modality_bindings=modality_bindings,
        )
        for record in horizon.source_stages
    )


def _validate_legal_history(horizon: HorizonView, transitions: tuple[LegalStageTransition, ...]) -> None:
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
    expected_modalities = tuple(modality.modality_id for record in horizon.source_stages for modality in record.modalities)
    if tuple(modality.modality_id for modality in horizon.modalities) != expected_modalities:
        raise ValueError("HorizonView modalities are not source-stage bound")


def encode_legal_multimodal_state(
    horizon: HorizonView,
    trained_model: MASPOProcessStateModel,
    *,
    stage_history: Iterable[LegalStageTransition] | None = None,
    stage_inputs: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    modality_inputs: Mapping[str, torch.Tensor] | None = None,
    modality_bindings: Mapping[str, str] | None = None,
    initial_state: torch.Tensor | None = None,
    model_version: str,
    model_fingerprint: str,
    validation_status: ModelValidationStatus = ModelValidationStatus.TRAINED_UNVALIDATED,
    training_evidence_id: str | None = None,
    dataset_fingerprint: str | None = None,
) -> OptimizationState:
    """Encode a source-bound legal history through fusion and StageAwareProcessModel."""
    if not model_version.strip() or not model_fingerprint.strip():
        raise ValueError("multimodal optimization state requires explicit model version and fingerprint")
    status = ModelValidationStatus(validation_status)
    if status == ModelValidationStatus.SOURCE_BACKED_VALIDATED and (not training_evidence_id or not dataset_fingerprint):
        raise ValueError("SOURCE_BACKED_VALIDATED requires training evidence and dataset fingerprint")
    if initial_state is None:
        raise ValueError("multimodal optimization state requires an explicit source-backed initial_state")
    if stage_history is None:
        if stage_inputs is None or modality_inputs is None or modality_bindings is None:
            raise ValueError("production multimodal encoding requires source-bound stage history or its source-stage inputs")
        transitions = build_legal_stage_transitions(
            horizon, stage_inputs=stage_inputs, modality_inputs=modality_inputs, modality_bindings=modality_bindings,
        )
    else:
        transitions = tuple(stage_history)
    if any(not isinstance(item, LegalStageTransition) for item in transitions):
        raise TypeError("stage_history accepts only source-bound LegalStageTransition values")
    _validate_legal_history(horizon, transitions)
    if initial_state.ndim not in (1, 2) or initial_state.shape[-1] != trained_model.stage_model.stage_embedding.embedding_dim:
        raise ValueError("initial_state has the wrong shape for the MASPO StageAwareProcessModel")

    was_training = trained_model.training
    trained_model.eval()
    try:
        with torch.no_grad():
            state = trained_model.state_from_transitions(initial_state, transitions) if transitions else initial_state
    finally:
        trained_model.train(was_training)
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
    semantic_metadata: dict[str, Any] = {
        "source_stage_ids": horizon.source_stage_ids,
        "model_fingerprint": model_fingerprint,
        "model_version": model_version,
        "modality_availability": availability,
        "modality_bindings": bindings,
    }
    if training_evidence_id is not None:
        semantic_metadata["training_evidence_id"] = training_evidence_id
    if dataset_fingerprint is not None:
        semantic_metadata["dataset_fingerprint"] = dataset_fingerprint
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
            "training_evidence_id": training_evidence_id, "dataset_fingerprint": dataset_fingerprint,
            "modality_ids": tuple(availability), "modality_availability": availability,
            "modality_bindings": bindings, "semantic_fingerprint_inputs": semantic_metadata,
            "source_transition_fingerprints": tuple(item.provenance["source_stage_fingerprint"] for item in transitions),
            "validation_status": status.value,
        },
        validation_status=status,
    )
