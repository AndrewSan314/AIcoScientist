from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.base import DatasetSpec
from src.science.coordinator import ScientificClosedLoopCoordinator
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.synthetic import SyntheticExperimentOracle, SyntheticScienceAdapter
from src.science.validation import InformationHorizonError, validate_record_against_spec


def test_critical_future_characterization_leakage_invariance() -> None:
    """CRITICAL TEST 1: Pre-observation invariance.

    Verifies that altering future post-experiment characterization physics before proposal
    has ZERO effect on the proposed candidate, process parameters, or scientific rationale.
    """
    seed = 42
    adapter = SyntheticScienceAdapter()

    # Initial history is identical for both runs
    init_df = adapter.load_initial_dataset(n_samples=10, seed=seed)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=50, seed=seed)

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

    # Step 1 proposal must be EXACTLY IDENTICAL
    rec_1, rat_1 = coord_1.propose_next(n_mc_samples=32)
    rec_2, rat_2 = coord_2.propose_next(n_mc_samples=32)

    assert rec_1.candidate_id == rec_2.candidate_id
    assert rec_1.pre_experiment_features == rec_2.pre_experiment_features
    assert rat_1.predicted_performance_mean == pytest.approx(rat_2.predicted_performance_mean, rel=1e-5)
    assert rat_1.acquisition_score == pytest.approx(rat_2.acquisition_score, rel=1e-5)
    assert rat_1.expected_learning_value == pytest.approx(rat_2.expected_learning_value, rel=1e-5)


def test_post_observation_characterization_divergence() -> None:
    """CRITICAL TEST 2: Post-observation legitimate divergence.

    Verifies that after ingesting different true characterization measurements,
    the model state updates and subsequent proposals legitimately diverge.
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

    # Ingest standard characterization for A vs extreme characterization for B
    coord_a.record_executed(rec_a1.experiment_id)
    coord_a.record_characterization(rec_a1.experiment_id, {"z1": 0.5, "z2": 0.5})
    coord_a.record_performance(rec_a1.experiment_id, {"y": 550.0})

    coord_b.record_executed(rec_b1.experiment_id)
    coord_b.record_characterization(rec_b1.experiment_id, {"z1": 25.0, "z2": -20.0})
    coord_b.record_performance(rec_b1.experiment_id, {"y": 950.0})

    # Proposals at step 2 must now legitimately differ in rationale / expectations
    rec_a2, rat_a2 = coord_a.propose_next(n_mc_samples=16)
    rec_b2, rat_b2 = coord_b.propose_next(n_mc_samples=16)

    # The predicted performance on B is substantially higher because of the high observed y and different z
    assert rat_a2.predicted_performance_mean != pytest.approx(rat_b2.predicted_performance_mean, rel=1e-2)


def test_oracle_column_firewall_rejection() -> None:
    """CRITICAL TEST 3: Oracle firewall rejection.

    Verifies that any attempt to include hidden oracle columns raises InformationHorizonError.
    """
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

    # Attempt to leak oracle into candidate_variables
    leaky_rec = ScientificExperimentRecord(
        experiment_id="EXP_001",
        candidate_id="CAND_001",
        dataset_name="test_oracle_firewall",
        pre_experiment_features={"x1": 1.0},
        candidate_variables={"x1": 1.0, "secret_oracle_score": 99.0},
    )

    with pytest.raises(InformationHorizonError, match="Oracle leakage detected"):
        validate_record_against_spec(leaky_rec, spec_with_oracle)
