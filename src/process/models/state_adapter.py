from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

from ..information_horizon import HorizonView
from ..optimization.state import OptimizationState, context_provenance_fingerprint
from ..stages import ProcessStage, stage_precedes
from .maspo import MASPOProcessStateModel


StageTransition = tuple[
    ProcessStage,
    torch.Tensor,
    torch.Tensor,
    Mapping[str, torch.Tensor],
    Mapping[str, bool | torch.Tensor],
]


def encode_legal_multimodal_state(
    horizon: HorizonView,
    trained_model: MASPOProcessStateModel,
    *,
    modality_inputs: Mapping[str, torch.Tensor],
    modality_bindings: Mapping[str, str],
    initial_state: torch.Tensor | None = None,
    stage_history: Iterable[StageTransition] = (),
    model_version: str,
    model_fingerprint: str,
    validated: bool = False,
) -> OptimizationState:
    """Encode only a legal horizon into optimizer features using the existing MASPO model."""
    if not model_version.strip() or not model_fingerprint.strip():
        raise ValueError("multimodal optimization state requires explicit model version and fingerprint")
    tokens: dict[str, torch.Tensor] = {}
    available: dict[str, bool] = {}
    modality_ids: list[str] = []
    availability: dict[str, bool] = {}
    for modality in horizon.modalities:
        if not stage_precedes(modality.observed_at_stage, horizon.decision_stage):
            raise ValueError(f"modality {modality.modality_id!r} is unavailable before {horizon.decision_stage.value}")
        model_name = modality_bindings.get(modality.modality_id)
        if model_name is None or model_name in available:
            raise ValueError(f"legal modality {modality.modality_id!r} lacks a unique model binding")
        modality_ids.append(modality.modality_id)
        if modality.is_missing:
            available[model_name] = False
            availability[modality.modality_id] = False
            continue
        if modality.modality_id not in modality_inputs:
            raise ValueError(f"observed modality {modality.modality_id!r} lacks encoder input")
        tokens[model_name] = modality_inputs[modality.modality_id]
        available[model_name] = True
        availability[modality.modality_id] = True
    if not any(available.values()):
        raise ValueError("multimodal optimization state needs at least one observed legal modality")

    legal_history = []
    for transition in stage_history:
        if not stage_precedes(transition[0], horizon.decision_stage):
            raise ValueError(f"stage history contains information unavailable before {horizon.decision_stage.value}")
        legal_history.append(transition)
    was_training = trained_model.training
    trained_model.eval()
    try:
        with torch.no_grad():
            current_fused = trained_model.fuse_observations(tokens, available)
            if legal_history:
                if initial_state is None:
                    raise ValueError("initial_state is required when stage_history is supplied")
                history_state = trained_model.state_from_transitions(initial_state, legal_history)
                if history_state.ndim == 1 and current_fused.ndim == 2:
                    history_state = history_state.unsqueeze(0)
                if current_fused.ndim == 1 and history_state.ndim == 2:
                    current_fused = current_fused.unsqueeze(0)
                latent = torch.cat([history_state, current_fused], dim=-1)
            else:
                latent = current_fused
    finally:
        trained_model.train(was_training)
    if latent.ndim == 2:
        if latent.shape[0] != 1:
            raise ValueError("multimodal optimization state must encode one current process state")
        latent = latent[0]
    values = latent.detach().cpu().numpy().reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("multimodal model produced no finite optimizer state")
    feature_names = tuple(f"context.latent.{index}" for index in range(values.size))
    feature_values = {name: float(value) for name, value in zip(feature_names, values)}
    representation_kind = "multimodal_latent" if not legal_history else "scalar_plus_multimodal"
    fingerprint = context_provenance_fingerprint(feature_values, horizon.decision_stage, representation_kind)
    return OptimizationState(
        feature_names=feature_names, feature_values=feature_values, decision_stage=horizon.decision_stage,
        source_stage_ids=horizon.source_stage_ids, provenance_fingerprint=fingerprint,
        representation_kind=representation_kind,
        provenance={
            "model_version": model_version, "model_fingerprint": model_fingerprint,
            "modality_ids": modality_ids, "modality_availability": availability,
            "modality_bindings": dict(modality_bindings),
            "fusion_mode": "current_multimodal" if not legal_history else "history_state_plus_current_multimodal",
            "validation_status": "IMPLEMENTED_NOT_VALIDATED",
        },
    )
