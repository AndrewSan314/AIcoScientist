from dataclasses import replace

from src.process.modalities import ModalitySlotSpec, ModalityType
from src.process.stage_aware_training import train_stage_aware_multimodal
from src.process.stages import ProcessStage
from .conftest import process_run


def _runs():
    return [replace(process_run(), run_id=f"run-{index}", batch_id=f"batch-{index}") for index in range(3)]


def _slots():
    return (ModalitySlotSpec("mixing.process", ProcessStage.MIXING, ModalityType.PROCESS_TABULAR, "process", 2),)


def test_stage_aware_training_masks_labels_and_freezes_an_artifact():
    result = train_stage_aware_multimodal(_runs(), modality_slots=_slots(), final_targets=("capacity",), dataset_fingerprint="dataset-v1", epochs=2)
    assert result.status == "EVALUATED_SOURCE_BACKED"
    assert result.artifact is not None and result.artifact.model_version.startswith("stage_aware")
    assert result.final_target_masks == {"capacity": 1}
    assert result.artifact.model_weight_fingerprint
    assert set(result.train_run_ids).isdisjoint(result.validation_run_ids) and set(result.train_run_ids).isdisjoint(result.test_run_ids)


def test_stage_aware_training_refuses_insufficient_linked_trajectories():
    result = train_stage_aware_multimodal(_runs()[:2], modality_slots=_slots(), final_targets=("capacity",), dataset_fingerprint="dataset-v1")
    assert result.status == "IMPLEMENTED_NOT_EVALUATABLE_ON_THIS_DATASET"
