from __future__ import annotations

import json
from pathlib import Path

from src.science.provenance import (
    ScientificModelProvenance,
    build_benchmark_run_manifest,
    get_environment_provenance,
    get_git_provenance,
)


def test_get_git_provenance() -> None:
    git_info = get_git_provenance()
    assert "code_head_commit" in git_info
    assert "git_dirty" in git_info
    assert "git_diff_sha256" in git_info
    assert "branch" in git_info
    assert isinstance(git_info["git_dirty"], bool)


def test_get_environment_provenance() -> None:
    env_info = get_environment_provenance()
    assert "python_version" in env_info
    assert "numpy_version" in env_info
    assert "scipy_version" in env_info
    assert "sklearn_version" in env_info
    assert "joblib_version" in env_info
    assert "platform" in env_info


def test_build_benchmark_run_manifest() -> None:
    manifest = build_benchmark_run_manifest(
        dataset_name="attia_continuous",
        comparison_baseline_commit="53a1c7241222105cdede343d5a155fdd5a97ee78",
        simulator_version="1.0.0",
        n_seeds=30,
        budgets=[10, 15, 20, 30],
        strategies=["random", "greedy", "gp_ucb", "expected_improvement", "nei", "turbo_nei", "adaptive"],
    )
    assert manifest["dataset"] == "attia_continuous"
    assert manifest["comparison_baseline_commit"] == "53a1c7241222105cdede343d5a155fdd5a97ee78"
    assert manifest["simulator_version"] == "1.0.0"
    assert "generated_at" in manifest
    assert "python_version" in manifest
    assert "numpy_version" in manifest
    assert "git_dirty" in manifest


def test_scientific_model_provenance_determinism() -> None:
    prov1 = ScientificModelProvenance.create(
        dataset_name="synthetic_science",
        dataset_fingerprint="ds_abc123",
        spec_fingerprint="spec_xyz789",
        training_experiment_ids=["EXP_001", "EXP_002"],
        feature_columns=["x1", "x2"],
        target_columns=["y"],
        random_seed=42,
        model_types={"stage_a": "GaussianProcessRegressor", "stage_b": "GaussianProcessRegressor"},
    )
    prov2 = ScientificModelProvenance.create(
        dataset_name="synthetic_science",
        dataset_fingerprint="ds_abc123",
        spec_fingerprint="spec_xyz789",
        training_experiment_ids=["EXP_002", "EXP_001"],  # Same IDs different order
        feature_columns=["x1", "x2"],
        target_columns=["y"],
        random_seed=42,
        model_types={"stage_a": "GaussianProcessRegressor", "stage_b": "GaussianProcessRegressor"},
    )
    assert prov1.model_run_id == prov2.model_run_id
    assert prov1.to_dict()["model_run_id"] == prov2.to_dict()["model_run_id"]

    # Changing training IDs or features changes the fingerprint
    prov3 = ScientificModelProvenance.create(
        dataset_name="synthetic_science",
        dataset_fingerprint="ds_abc123",
        spec_fingerprint="spec_xyz789",
        training_experiment_ids=["EXP_001", "EXP_003"],
        feature_columns=["x1", "x2"],
        target_columns=["y"],
        random_seed=42,
        model_types={"stage_a": "GaussianProcessRegressor", "stage_b": "GaussianProcessRegressor"},
    )
    assert prov1.model_run_id != prov3.model_run_id
