from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datasets.base import DatasetSpec, TwoStageModelSpec
from src.science.direct_baseline import DirectPerformanceModel
from src.science.evaluation import evaluate_two_stage_model
from src.science.model_bundle import ScientificModelBundle
from src.science.provenance import ScientificModelProvenance
from src.science.two_stage import TwoStageScientificModel


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame, TwoStageModelSpec, DatasetSpec]:
    rng = np.random.default_rng(42)
    n_train = 30
    n_test = 15

    # Process variables
    x1_tr = rng.uniform(1.0, 5.0, size=n_train)
    x2_tr = rng.uniform(10.0, 50.0, size=n_train)

    # Characterization channels: z1 = sin(x1) + 0.1*x2, z2 = cos(x2/10)
    z1_tr = np.sin(x1_tr) + 0.05 * x2_tr + rng.normal(0.0, 0.05, size=n_train)
    z2_tr = np.cos(x2_tr / 10.0) + rng.normal(0.0, 0.05, size=n_train)

    # Performance target: y = 500 + 50*x1 + 100*z1 - 40*z2 + noise
    y_tr = 500.0 + 50.0 * x1_tr + 100.0 * z1_tr - 40.0 * z2_tr + rng.normal(0.0, 5.0, size=n_train)

    train_df = pd.DataFrame({
        "sample_id": [f"TR_{i:03d}" for i in range(n_train)],
        "x1": x1_tr,
        "x2": x2_tr,
        "z1": z1_tr,
        "z2": z2_tr,
        "y": y_tr,
    })

    # Test set
    x1_te = rng.uniform(1.0, 5.0, size=n_test)
    x2_te = rng.uniform(10.0, 50.0, size=n_test)
    z1_te = np.sin(x1_te) + 0.05 * x2_te + rng.normal(0.0, 0.05, size=n_test)
    z2_te = np.cos(x2_te / 10.0) + rng.normal(0.0, 0.05, size=n_test)
    y_te = 500.0 + 50.0 * x1_te + 100.0 * z1_te - 40.0 * z2_te + rng.normal(0.0, 5.0, size=n_test)

    test_df = pd.DataFrame({
        "sample_id": [f"TE_{i:03d}" for i in range(n_test)],
        "x1": x1_te,
        "x2": x2_te,
        "z1": z1_te,
        "z2": z2_te,
        "y": y_te,
    })

    two_stage_spec = TwoStageModelSpec(
        dataset_name="synthetic_science_test",
        process_features=["x1", "x2"],
        characterization_targets=["z1", "z2"],
        performance_targets=["y"],
    )

    dataset_spec = DatasetSpec(
        name="synthetic_science_test",
        id_column="sample_id",
        feature_columns=["x1", "x2", "z1", "z2"],
        target_column="y",
        candidate_columns=["x1", "x2"],
        pre_experiment_features=["x1", "x2"],
        post_experiment_characterization=["z1", "z2"],
        targets=["y"],
        candidate_variables=["x1", "x2"],
    )

    return train_df, test_df, two_stage_spec, dataset_spec


def test_two_stage_scientific_model_predictions_and_variance_decomposition(synthetic_data) -> None:
    train_df, test_df, two_stage_spec, dataset_spec = synthetic_data

    model = TwoStageScientificModel(two_stage_spec, random_state=42)
    model.fit(train_df)
    assert model.is_fitted

    # 1. Stage A Predictions
    char_preds = model.predict_characterization(test_df[["x1", "x2"]])
    assert "z1" in char_preds and "z2" in char_preds
    assert len(char_preds["z1"]["mean"]) == len(test_df)
    assert np.all(char_preds["z1"]["latent_std"] > 0.0)
    assert np.all(char_preds["z1"]["observation_std"] >= char_preds["z1"]["latent_std"])

    # 2. Stage B Diagnostic with Observed Characterization
    b_mean, b_std = model.predict_performance_with_observed_characterization(
        test_df[["x1", "x2"]],
        test_df[["z1", "z2"]],
    )
    assert len(b_mean) == len(test_df)
    assert np.all(b_std > 0.0)

    # 3. End-to-End Monte Carlo Propagation
    e2e_pred = model.predict_end_to_end(
        test_df[["x1", "x2"]],
        n_mc_samples=64,
        seed=101,
    )
    assert len(e2e_pred.performance_mean) == len(test_df)
    assert len(e2e_pred.performance_latent_std) == len(test_df)
    assert np.all(e2e_pred.performance_latent_std > 0.0)

    # Law of Total Variance verification
    assert np.allclose(
        e2e_pred.total_variance,
        e2e_pred.performance_model_variance + e2e_pred.characterization_propagation_variance,
        atol=1e-6,
    )
    assert np.all(e2e_pred.characterization_propagation_variance >= 0.0)
    assert np.all(e2e_pred.performance_model_variance > 0.0)

    # Determinism with same seed
    e2e_pred_same = model.predict_end_to_end(test_df[["x1", "x2"]], n_mc_samples=64, seed=101)
    assert np.allclose(e2e_pred.performance_mean, e2e_pred_same.performance_mean, atol=1e-9)
    assert np.allclose(e2e_pred.performance_latent_std, e2e_pred_same.performance_latent_std, atol=1e-9)


def test_direct_baseline_and_honest_evaluation(synthetic_data) -> None:
    train_df, test_df, two_stage_spec, dataset_spec = synthetic_data

    two_stage_model = TwoStageScientificModel(two_stage_spec, random_state=42)
    two_stage_model.fit(train_df)

    direct_model = DirectPerformanceModel(
        process_features=two_stage_spec.process_features,
        target_column="y",
        random_state=42,
    )
    direct_model.fit(train_df)
    assert direct_model.is_fitted

    dir_mean, dir_std = direct_model.predict(test_df[["x1", "x2"]])
    assert len(dir_mean) == len(test_df)
    assert np.all(dir_std > 0.0)

    report = evaluate_two_stage_model(
        two_stage_model=two_stage_model,
        direct_model=direct_model,
        test_df=test_df,
        spec=dataset_spec,
        two_stage_spec=two_stage_spec,
        n_mc_samples=32,
        seed=42,
    )

    assert "direct_baseline" in report
    assert "stage_a_characterization" in report
    assert "stage_b_diagnostic_upper_bound" in report
    assert "two_stage_end_to_end" in report
    assert "model_disagreement_summary" in report
    assert "coverage_95_pct" in report["two_stage_end_to_end"]["calibration"]


def test_scientific_model_bundle_serialization(synthetic_data, tmp_path: Path) -> None:
    train_df, test_df, two_stage_spec, dataset_spec = synthetic_data

    two_stage_model = TwoStageScientificModel(two_stage_spec, random_state=42)
    two_stage_model.fit(train_df)

    direct_model = DirectPerformanceModel(
        process_features=two_stage_spec.process_features,
        target_column="y",
        random_state=42,
    )
    direct_model.fit(train_df)

    prov = ScientificModelProvenance.create(
        dataset_name=dataset_spec.name,
        dataset_fingerprint="synth_fp",
        spec_fingerprint="spec_fp",
        training_experiment_ids=list(train_df["sample_id"]),
        feature_columns=dataset_spec.feature_columns,
        target_columns=dataset_spec.targets,
        random_seed=42,
        model_types={"direct": "GPR", "stage_a": "GPR", "stage_b": "GPR"},
    )

    bundle = ScientificModelBundle(
        direct_model=direct_model,
        two_stage_model=two_stage_model,
        spec=dataset_spec,
        two_stage_spec=two_stage_spec,
        provenance=prov,
        evaluation_report={"test_mae": 1.23},
    )

    bundle_path = tmp_path / "model_bundle.pkl"
    bundle.save(bundle_path)
    assert bundle_path.is_file()

    loaded_bundle = ScientificModelBundle.load(bundle_path)
    assert loaded_bundle.provenance.model_run_id == prov.model_run_id
    assert loaded_bundle.two_stage_model.is_fitted
    assert loaded_bundle.direct_model.is_fitted

    # Predictions from loaded bundle match original
    m_orig, _ = bundle.direct_model.predict(test_df[["x1", "x2"]])
    m_load, _ = loaded_bundle.direct_model.predict(test_df[["x1", "x2"]])
    assert np.allclose(m_orig, m_load)
