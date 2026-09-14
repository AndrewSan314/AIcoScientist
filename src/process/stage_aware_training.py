"""End-to-end, source-bound training for the learned stage-aware model family."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .contracts import BatteryProcessRun
from .modalities import ModalitySlotSpec
from .models import MASPOModelArtifact, MASPOProcessStateModel, StageFeatureEncoder
from .models.transitions import LegalStageTransition
from .optimization.state import ModelValidationStatus
from .stages import ProcessStage


@dataclass(frozen=True)
class StageAwareTrainingResult:
    status: str
    reason: str | None
    artifact: MASPOModelArtifact | None
    train_run_ids: tuple[str, ...] = ()
    validation_run_ids: tuple[str, ...] = ()
    test_run_ids: tuple[str, ...] = ()
    split_fingerprint: str | None = None
    final_target_masks: dict[str, int] | None = None
    intermediate_target_masks: dict[str, int] | None = None
    test_mae: dict[str, float] | None = None


def train_stage_aware_multimodal(
    runs: Sequence[BatteryProcessRun], *, modality_slots: Sequence[ModalitySlotSpec], final_targets: Sequence[str], dataset_fingerprint: str, seed: int = 42, epochs: int = 20, state_dim: int = 8, embedding_dim: int = 4,
) -> StageAwareTrainingResult:
    """Train a compact linked-trajectory model without exposing future stages or KPIs."""
    groups = {run.batch_id or run.run_id for run in runs}
    if len(runs) < 3 or len(groups) < 3:
        return StageAwareTrainingResult("IMPLEMENTED_NOT_EVALUATABLE_ON_THIS_DATASET", "at least three source-isolated trajectories are required", None)
    if not modality_slots:
        return StageAwareTrainingResult("IMPLEMENTED_NOT_EVALUATABLE_ON_THIS_DATASET", "stage-aware multimodal training requires source-declared modality slots", None)
    if not final_targets or not dataset_fingerprint.strip() or epochs < 1:
        raise ValueError("final targets, dataset fingerprint, and positive epochs are required")
    torch.manual_seed(seed); np.random.seed(seed)
    ordered_groups = np.asarray(sorted(groups), dtype=object); np.random.default_rng(seed).shuffle(ordered_groups)
    test_group, validation_group = str(ordered_groups[0]), str(ordered_groups[1])
    train = tuple(run for run in runs if (run.batch_id or run.run_id) not in {test_group, validation_group})
    validation = tuple(run for run in runs if (run.batch_id or run.run_id) == validation_group)
    test = tuple(run for run in runs if (run.batch_id or run.run_id) == test_group)
    train_records = [record for run in train for record in run.stages]
    encoder = StageFeatureEncoder.fit(train_records)
    dimensions = {slot.model_input_name: slot.expected_input_dim for slot in modality_slots}
    if any(dimensions[slot.model_input_name] != slot.expected_input_dim for slot in modality_slots):
        raise ValueError("modality slots must bind each model input name to one dimension")
    model = MASPOProcessStateModel(state_dim, encoder.control_dim, encoder.observation_dim, dimensions, embedding_dim=embedding_dim)
    for target in final_targets:
        model.add_final_head(target)
    intermediate_targets = sorted({(record.stage_type, name) for record in train_records for name, value in record.intermediate_properties.items() if isinstance(value.value, (int, float))}, key=lambda item: (item[0].value, item[1]))
    for stage, target in intermediate_targets:
        model.add_intermediate_head(stage, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    final_masks = {target: sum(isinstance(run.final_kpis.get(target).value, (int, float)) for run in train if run.final_kpis.get(target) is not None) for target in final_targets}
    intermediate_masks = {f"{stage.value}.{target}": sum(isinstance(record.intermediate_properties.get(target).value, (int, float)) for run in train for record in run.stages if record.stage_type == stage and record.intermediate_properties.get(target) is not None) for stage, target in intermediate_targets}
    for _ in range(epochs):
        optimizer.zero_grad(); losses = []
        for run in train:
            outputs = model(model.initial_state, _transitions(run, encoder, modality_slots))
            for target in final_targets:
                value = run.final_kpis.get(target)
                if value is not None and isinstance(value.value, (int, float)):
                    losses.append(F.mse_loss(outputs[target].reshape(()), torch.tensor(float(value.value))))
            for record in run.stages:
                for target, value in record.intermediate_properties.items():
                    key = f"{record.stage_type.value}.{target}"
                    if key in outputs and isinstance(value.value, (int, float)):
                        losses.append(F.mse_loss(outputs[key].reshape(()), torch.tensor(float(value.value))))
        if not losses:
            raise ValueError("training trajectories contain no numeric final or intermediate labels")
        torch.stack(losses).mean().backward(); optimizer.step()
    artifact = MASPOModelArtifact.from_training(model, feature_encoder=encoder, model_version="stage_aware_multimodal_surrogate-v1", semantic_modality_slots=modality_slots, dataset_fingerprint=dataset_fingerprint, validation_status=ModelValidationStatus.TRAINED_UNVALIDATED)
    artifact.verify_model_integrity()
    metrics = _evaluate(artifact.model, test, encoder, modality_slots, final_targets, intermediate_targets)
    split = {"train": [run.run_id for run in train], "validation": [run.run_id for run in validation], "test": [run.run_id for run in test], "seed": seed, "dataset_fingerprint": dataset_fingerprint}
    return StageAwareTrainingResult("EVALUATED_SOURCE_BACKED", None, artifact, tuple(split["train"]), tuple(split["validation"]), tuple(split["test"]), hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(), final_masks, intermediate_masks, metrics)


def _transitions(run: BatteryProcessRun, encoder: StageFeatureEncoder, slots: Sequence[ModalitySlotSpec]) -> tuple[LegalStageTransition, ...]:
    return tuple(LegalStageTransition.from_encoded_source_stage(record, encoder=encoder, modality_slots=slots) for record in run.stages)


def _evaluate(
    model: MASPOProcessStateModel,
    runs: Sequence[BatteryProcessRun],
    encoder: StageFeatureEncoder,
    slots: Sequence[ModalitySlotSpec],
    targets: Sequence[str],
    intermediate_targets: Sequence[tuple[ProcessStage, str]] = (),
) -> dict[str, float]:
    model.eval(); errors: dict[str, list[float]] = {target: [] for target in targets}
    for stage, target in intermediate_targets:
        errors[f"{stage.value}.{target}"] = []
    with torch.no_grad():
        for run in runs:
            outputs = model(model.initial_state, _transitions(run, encoder, slots))
            for target in targets:
                value = run.final_kpis.get(target)
                if value is not None and isinstance(value.value, (int, float)):
                    errors[target].append(abs(float(outputs[target]) - float(value.value)))
            for record in run.stages:
                for target, value in record.intermediate_properties.items():
                    key = f"{record.stage_type.value}.{target}"
                    if key in errors and isinstance(value.value, (int, float)):
                        errors[key].append(abs(float(outputs[key]) - float(value.value)))
    return {target: float(np.mean(values)) for target, values in errors.items() if values}
