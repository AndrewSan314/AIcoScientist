from __future__ import annotations

import json

import pytest

from scripts.run_maspo_pipeline import synthetic_frame
from src.process.surrogates.core import GenericTabularAdapter, SurrogateArtifact
from src.process.surrogates.pipeline import PipelineConfig, run_pipeline


MAPPING = {
    "dataset_name": "synthetic-process-fixture", "dataset_version": "1", "sample_id": "sample_id", "run_id": "run_id", "recipe_id": "recipe_id",
    "group_id": "group_id", "stage": "stage", "fidelity": "fidelity", "requested_horizon": "requested_horizon",
    "controls": ["cbd_fraction", "calender_pressure"], "observations": ["drying_porosity"], "previous_state": ["slurry_viscosity"], "modality_state": [], "targets": ["capacity_retention"],
}


def test_synthetic_pipeline_is_grouped_auditable_and_frozen(tmp_path) -> None:
    report = run_pipeline(
        GenericTabularAdapter(synthetic_frame(), MAPPING, source="synthetic-test"), tmp_path,
        PipelineConfig(("capacity_retention",), model_type="extra_trees", proposal_count=2),
    )
    split = json.loads((tmp_path / "split_manifest.json").read_text(encoding="utf-8"))
    assert report["status"] == "SUCCESS_SYNTHETIC_ONLY"
    assert not (set(split["train_ids"]) & set(split["validation_ids"]) or set(split["train_ids"]) & set(split["test_ids"]))
    assert (tmp_path / "revalidation_queue.json").is_file()
    assert SurrogateArtifact.load(tmp_path / "surrogate.joblib", expected_dataset_fingerprint=report["dataset_fingerprint"])


def test_target_cannot_be_exposed_as_feature() -> None:
    mapping = {**MAPPING, "controls": ["capacity_retention"]}
    report = GenericTabularAdapter(synthetic_frame(), mapping).validate()
    assert not report.valid
    assert "target is incorrectly exposed as a feature: capacity_retention" in report.errors
