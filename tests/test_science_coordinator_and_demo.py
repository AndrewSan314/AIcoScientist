from __future__ import annotations

from pathlib import Path

import pytest

from src.datasets.base import DatasetSpec
from src.optimization.search_space import ContinuousVariable, SearchSpace
from src.science.cli import run_synthetic_demo
from src.science.coordinator import PendingExperimentError, ScientificClosedLoopCoordinator
from src.science.records import ExperimentStage
from src.science.synthetic import SyntheticExperimentOracle, SyntheticScienceAdapter
from src.science.validation import InformationHorizonError


def test_coordinator_closed_loop_and_pending_protection(tmp_path: Path) -> None:
    db_file = tmp_path / "coordinator_test.db"
    adapter = SyntheticScienceAdapter()
    oracle = SyntheticExperimentOracle()

    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_file,
        strategy="expected_improvement",
        allow_parallel_experiments=False,
        random_state=42,
    )

    # 1. First proposal
    rec1, rat1 = coord.propose_next(n_mc_samples=32)
    assert rec1.stage == ExperimentStage.PROPOSED
    assert rat1.experiment_id == rec1.experiment_id
    assert len(rat1.render_text()) > 0
    assert 0.0 <= rat1.expected_learning_value <= 1.0

    # 2. Attempting second proposal while first is pending must RAISE PendingExperimentError
    with pytest.raises(PendingExperimentError, match="Cannot propose a new experiment"):
        coord.propose_next()

    # 3. Execute & characterize
    coord.record_executed(rec1.experiment_id)
    chars = oracle.evaluate_characterization(rec1.pre_experiment_features, seed=100)
    coord.record_characterization(rec1.experiment_id, chars)

    # Still pending performance!
    with pytest.raises(PendingExperimentError):
        coord.propose_next()

    # 4. Record performance & complete
    perf = oracle.evaluate_performance(rec1.pre_experiment_features, chars, seed=200)
    completed_rec = coord.record_performance(rec1.experiment_id, perf)
    assert completed_rec.stage == ExperimentStage.COMPLETED

    # 5. Now next proposal is allowed
    rec2, rat2 = coord.propose_next(n_mc_samples=32)
    assert rec2.stage == ExperimentStage.PROPOSED
    assert rec2.experiment_id != rec1.experiment_id

    # 6. Test failed experiment semantics
    coord.record_executed(rec2.experiment_id)
    failed_rec = coord.record_failed(rec2.experiment_id, failure_reason="Synthesis precipitate clogged filter")
    assert failed_rec.stage == ExperimentStage.FAILED
    assert failed_rec.failure_reason == "Synthesis precipitate clogged filter"
    assert not failed_rec.has_performance()  # No fake target value!

    # Verify ledger integrity
    valid, errors = coord.ledger.verify_integrity()
    assert valid
    assert len(errors) == 0


def test_coordinator_asymmetric_async_flow(tmp_path: Path) -> None:
    """Tests case where performance arrives before physical characterization."""
    db_file = tmp_path / "async_flow.db"
    adapter = SyntheticScienceAdapter()
    oracle = SyntheticExperimentOracle()

    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_file,
        strategy="expected_improvement",
        random_state=42,
    )

    rec, _ = coord.propose_next(n_mc_samples=16)
    coord.record_executed(rec.experiment_id)

    # 1. Performance arrives FIRST
    perf = {"y": 620.5}
    rec_perf = coord.record_performance(rec.experiment_id, perf)
    assert rec_perf.stage == ExperimentStage.PERFORMANCE_MEASURED
    assert rec_perf.has_performance()
    assert not rec_perf.has_characterization()

    # 2. Characterization arrives SECOND -> triggers completion
    chars = {"z1": 0.42, "z2": -0.15}
    rec_completed = coord.record_characterization(rec.experiment_id, chars)
    assert rec_completed.stage == ExperimentStage.COMPLETED
    assert rec_completed.has_characterization()
    assert rec_completed.has_performance()


def test_coordinator_exact_deterministic_resume(tmp_path: Path) -> None:
    """Verifies that an uninterrupted run and an interrupted/resumed run produce 100% identical proposals."""
    adapter = SyntheticScienceAdapter()
    oracle = SyntheticExperimentOracle()

    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    # --- Run A: Uninterrupted 3 steps ---
    db_a = tmp_path / "run_a.db"
    coord_a = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_a,
        strategy="expected_improvement",
        random_state=42,
    )

    proposals_a = []
    for step in range(3):
        rec, _ = coord_a.propose_next(n_mc_samples=16)
        proposals_a.append(rec.candidate_variables)
        coord_a.record_executed(rec.experiment_id)
        chars = oracle.evaluate_characterization(rec.pre_experiment_features, seed=500 + step)
        coord_a.record_characterization(rec.experiment_id, chars)
        perf = oracle.evaluate_performance(rec.pre_experiment_features, chars, seed=600 + step)
        coord_a.record_performance(rec.experiment_id, perf)
    coord_a.ledger.close()

    # --- Run B: 2 steps, close, resume, propose 3rd step ---
    db_b = tmp_path / "run_b.db"
    coord_b = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_b,
        strategy="expected_improvement",
        random_state=42,
    )

    proposals_b = []
    for step in range(2):
        rec, _ = coord_b.propose_next(n_mc_samples=16)
        proposals_b.append(rec.candidate_variables)
        coord_b.record_executed(rec.experiment_id)
        chars = oracle.evaluate_characterization(rec.pre_experiment_features, seed=500 + step)
        coord_b.record_characterization(rec.experiment_id, chars)
        perf = oracle.evaluate_performance(rec.pre_experiment_features, chars, seed=600 + step)
        coord_b.record_performance(rec.experiment_id, perf)
    coord_b.ledger.close()

    # Resume from ledger
    resumed_b = ScientificClosedLoopCoordinator.resume_from_ledger(
        db_path=db_b,
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        strategy="expected_improvement",
        random_state=42,
    )
    rec3, _ = resumed_b.propose_next(n_mc_samples=16)
    proposals_b.append(rec3.candidate_variables)
    resumed_b.ledger.close()

    # Verify all 3 steps match bit-for-bit
    assert len(proposals_a) == 3
    assert len(proposals_b) == 3
    for step_idx in range(3):
        assert proposals_a[step_idx] == proposals_b[step_idx]


def test_coordinator_pre_experiment_context_merging(tmp_path: Path) -> None:
    db_file = tmp_path / "context_merge.db"
    adapter = SyntheticScienceAdapter()

    # Spec where x1, x2 are controllable, but x3 is non-controllable pre-experiment context
    custom_spec = DatasetSpec(
        name="custom_science",
        id_column="experiment_id",
        candidate_id_column="candidate_id",
        feature_columns=["x1", "x2", "x3", "z1", "z2"],
        target_column="y",
        pre_experiment_features=["x1", "x2", "x3"],
        candidate_variables=["x1", "x2"],
        post_experiment_characterization=["z1", "z2"],
        targets=["y"],
    )
    custom_space = SearchSpace(
        name="custom_space",
        variables=[
            ContinuousVariable(name="x1", lower=1.0, upper=5.0),
            ContinuousVariable(name="x2", lower=10.0, upper=50.0),
        ],
    )
    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    # Give candidate pool only controllable columns
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=20, seed=42)[["candidate_id", "x1", "x2"]]

    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=custom_spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=custom_space,
        db_path=db_file,
        strategy="expected_improvement",
        random_state=42,
    )

    # 1. Valid context: x3 provided via pre_experiment_context
    rec, _ = coord.propose_next(pre_experiment_context={"x3": 1.25})
    assert rec.pre_experiment_features["x3"] == 1.25
    assert "x1" in rec.candidate_variables
    assert "x2" in rec.candidate_variables
    assert "x3" not in rec.candidate_variables

    # 2. Conflict detection: attempting to override an optimizer-chosen controllable variable with conflicting context
    cand_col = "x1"
    cand_val = rec.candidate_variables[cand_col]
    coord.record_executed(rec.experiment_id)
    coord.record_failed(rec.experiment_id, "test conflict")

    with pytest.raises(InformationHorizonError, match="Conflict in pre_experiment_context"):
        coord.propose_next(pre_experiment_context={"x3": 1.25, cand_col: cand_val + 10.0})


def test_cli_demo_execution(tmp_path: Path) -> None:
    out_dir = tmp_path / "demo_output"
    res = run_synthetic_demo(seed=42, steps=3, output_dir=out_dir)

    assert res["status"] == "SUCCESS"
    assert res["ledger_valid"] is True
    assert (out_dir / "proposal_history.jsonl").is_file()
    assert (out_dir / "model_report.json").is_file()
    assert (out_dir / "run_provenance.json").is_file()
