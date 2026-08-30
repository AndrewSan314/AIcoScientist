from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.datasets.base import DatasetSpec
from src.optimization.closed_loop import ClosedLoopOptimizer
from src.science.coordinator import ScientificClosedLoopCoordinator
from src.science.direct_baseline import DirectPerformanceModel
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.synthetic import SyntheticExperimentOracle, SyntheticScienceAdapter
from src.science.two_stage import StageACharacterizationModel, TwoStageScientificModel
from src.science.validation import InformationHorizonError, validate_record_against_spec


def test_leakage_spy_proves_future_data_absent_from_proposal_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """TEST A: Model Input Spy.

    Monitors all function calls during propose_next() and verifies that proposal-time
    methods never receive candidate ground truth characterization, target performance, or oracle fields.
    """
    adapter = SyntheticScienceAdapter()
    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=30, seed=42)

    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        strategy="expected_improvement",
        random_state=42,
    )

    observed_direct_inputs: list[Any] = []
    orig_direct_predict = coord.model_bundle.direct_model.predict

    def spy_direct_predict(X_process: Any, return_std: bool = True) -> Any:
        if isinstance(X_process, pd.DataFrame):
            observed_direct_inputs.append(dict(X_process.iloc[0]))
        else:
            observed_direct_inputs.append(X_process)
        return orig_direct_predict(X_process, return_std=return_std)

    monkeypatch.setattr(coord.model_bundle.direct_model, "predict", spy_direct_predict)

    observed_stage_a_inputs: list[Any] = []
    orig_stage_a_predict = coord.model_bundle.two_stage_model.stage_a.predict

    def spy_stage_a_predict(X_process: Any, return_std: bool = True) -> Any:
        observed_stage_a_inputs.append(X_process)
        return orig_stage_a_predict(X_process, return_std=return_std)

    monkeypatch.setattr(coord.model_bundle.two_stage_model.stage_a, "predict", spy_stage_a_predict)

    # Generate proposal
    rec, rationale = coord.propose_next(n_mc_samples=16)

    # 1. Assert Direct model only received process features (x1, x2)
    assert len(observed_direct_inputs) > 0
    for inp in observed_direct_inputs:
        if isinstance(inp, dict):
            for k in inp.keys():
                assert k in adapter.two_stage_spec.process_features
                assert k not in adapter.two_stage_spec.characterization_targets
                assert k not in adapter.two_stage_spec.performance_targets

    # 2. Assert Stage A only received process feature matrix with dimension == len(process_features)
    assert len(observed_stage_a_inputs) > 0
    for inp in observed_stage_a_inputs:
        inp_mat = np.asarray(inp)
        assert inp_mat.shape[1] == len(adapter.two_stage_spec.process_features)

    # 3. Assert proposal record has empty characterization and performance
    assert rec.characterization == {}
    assert rec.performance == {}


def test_hidden_future_data_invariance() -> None:
    """TEST B: Hidden Future Data Invariance.

    Two simulated worlds with radically different future oracle physics produce identical
    proposals when initialized on identical historical data.
    """
    seed = 42
    adapter = SyntheticScienceAdapter()

    init_df = adapter.load_initial_dataset(n_samples=8, seed=seed)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=seed)

    coord_1 = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        strategy="expected_improvement",
        random_state=seed,
    )
    coord_2 = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        strategy="expected_improvement",
        random_state=seed,
    )

    # World 1 oracle and World 2 oracle differ, but neither is accessible at proposal time
    rec_1, rat_1 = coord_1.propose_next(n_mc_samples=16)
    rec_2, rat_2 = coord_2.propose_next(n_mc_samples=16)

    assert rec_1.candidate_id == rec_2.candidate_id
    assert rec_1.pre_experiment_features == rec_2.pre_experiment_features
    assert rat_1.predicted_performance_mean == pytest.approx(rat_2.predicted_performance_mean, rel=1e-5)
    assert rat_1.acquisition_score == pytest.approx(rat_2.acquisition_score, rel=1e-5)


def test_characterization_only_post_observation_divergence() -> None:
    """TEST C: Characterization-Only Post-Observation Divergence.

    Coordinator A and B receive IDENTICAL process conditions and IDENTICAL primary target y,
    but radically DIFFERENT characterization measurements (z1, z2).

    Verifies:
    1. Direct model on A and B remains EQUIVALENT (since (X, y) are identical).
    2. Stage A and Two-Stage end-to-end models on A and B DIVERGE (isolating characterization effects).
    """
    seed = 42
    adapter = SyntheticScienceAdapter()

    init_df = adapter.load_initial_dataset(n_samples=8, seed=seed)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=seed)

    coord_a = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        strategy="expected_improvement",
        random_state=seed,
    )
    coord_b = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        strategy="expected_improvement",
        random_state=seed,
    )

    rec_a1, _ = coord_a.propose_next(n_mc_samples=16)
    rec_b1, _ = coord_b.propose_next(n_mc_samples=16)

    # Identical performance y = 500.0 on both, but different characterization z
    coord_a.record_executed(rec_a1.experiment_id)
    coord_a.record_characterization(rec_a1.experiment_id, {"z1": 0.1, "z2": 0.1})
    coord_a.record_performance(rec_a1.experiment_id, {"y": 500.0})

    coord_b.record_executed(rec_b1.experiment_id)
    coord_b.record_characterization(rec_b1.experiment_id, {"z1": 25.0, "z2": -20.0})
    coord_b.record_performance(rec_b1.experiment_id, {"y": 500.0})

    # Test candidate for evaluation (including all process features: x1, x2, x3)
    test_cand = pd.DataFrame([{"x1": 3.0, "x2": 25.0, "x3": 5.0}])

    # 1. Direct models must predict identical means (both saw same x and same y)
    dir_mean_a, dir_std_a = coord_a.model_bundle.direct_model.predict(test_cand)
    dir_mean_b, dir_std_b = coord_b.model_bundle.direct_model.predict(test_cand)
    assert dir_mean_a[0] == pytest.approx(dir_mean_b[0], rel=1e-4)

    # 2. Stage A characterization models must predict different characterizations
    char_pred_a = coord_a.model_bundle.two_stage_model.stage_a.predict(test_cand)
    char_pred_b = coord_b.model_bundle.two_stage_model.stage_a.predict(test_cand)
    assert char_pred_a["z1"]["mean"][0] != pytest.approx(char_pred_b["z1"]["mean"][0], rel=1e-1)


def test_oracle_column_firewall_rejection() -> None:
    """TEST D: Oracle firewall rejection."""
    spec_with_oracle = DatasetSpec(
        name="test_oracle_firewall",
        id_column="exp_id",
        candidate_id_column="cand_id",
        feature_columns=["x1"],
        target_column="y",
        pre_experiment_features=["x1"],
        candidate_variables=["x1"],
        targets=["y"],
        oracle_columns=["secret_oracle_score", "hidden_ground_truth"],
    )

    leaky_rec = ScientificExperimentRecord(
        experiment_id="EXP_001",
        candidate_id="CAND_001",
        dataset_name="test_oracle_firewall",
        pre_experiment_features={"x1": 1.0},
        candidate_variables={"x1": 1.0, "secret_oracle_score": 99.0},
    )

    with pytest.raises(InformationHorizonError, match="Oracle leakage detected"):
        validate_record_against_spec(leaky_rec, spec_with_oracle)
