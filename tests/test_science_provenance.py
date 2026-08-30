from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.datasets.base import DatasetSpec, TwoStageModelSpec
from src.science.provenance import (
    ScientificModelProvenance,
    build_benchmark_run_manifest,
    compute_dataset_fingerprint,
    compute_spec_fingerprint,
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


def test_dataset_and_spec_fingerprints() -> None:
    df1 = pd.DataFrame({
        "exp_id": ["EXP_01", "EXP_02"],
        "x1": [1.0, 2.0],
        "y": [100.0, 200.0],
    })
    df2 = pd.DataFrame({
        "exp_id": ["EXP_02", "EXP_01"],
        "x1": [2.0, 1.0],
        "y": [200.0, 100.0],
    })
    # Identical content in different row order produces same fingerprint
    fp1 = compute_dataset_fingerprint(df1, feature_cols=["x1"], target_cols=["y"], id_col="exp_id")
    fp2 = compute_dataset_fingerprint(df2, feature_cols=["x1"], target_cols=["y"], id_col="exp_id")
    assert fp1 == fp2

    # Modified value changes fingerprint
    df3 = df1.copy()
    df3.loc[0, "y"] = 105.0
    fp3 = compute_dataset_fingerprint(df3, feature_cols=["x1"], target_cols=["y"], id_col="exp_id")
    assert fp1 != fp3

    spec1 = DatasetSpec(
        name="test_spec",
        id_column="exp_id",
        candidate_id_column="cand_id",
        feature_columns=["x1"],
        target_column="y",
        pre_experiment_features=["x1"],
        candidate_variables=["x1"],
        targets=["y"],
    )
    spec_fp1 = compute_spec_fingerprint(spec1)
    assert isinstance(spec_fp1, str) and len(spec_fp1) > 0


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
