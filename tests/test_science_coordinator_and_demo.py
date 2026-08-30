from __future__ import annotations

from pathlib import Path

import pytest

from src.science.cli import run_synthetic_demo
from src.science.coordinator import PendingExperimentError, ScientificClosedLoopCoordinator
from src.science.records import ExperimentStage
from src.science.synthetic import SyntheticExperimentOracle, SyntheticScienceAdapter


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


def test_coordinator_resume_from_ledger(tmp_path: Path) -> None:
    db_file = tmp_path / "resume_test.db"
    adapter = SyntheticScienceAdapter()
    oracle = SyntheticExperimentOracle()

    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        db_path=db_file,
        strategy="expected_improvement",
        random_state=42,
    )

    # Run 2 complete cycles
    for i in range(2):
        rec, _ = coord.propose_next(n_mc_samples=16)
        coord.record_executed(rec.experiment_id)
        chars = oracle.evaluate_characterization(rec.pre_experiment_features, seed=300 + i)
        coord.record_characterization(rec.experiment_id, chars)
        perf = oracle.evaluate_performance(rec.pre_experiment_features, chars, seed=400 + i)
        coord.record_performance(rec.experiment_id, perf)

    coord.ledger.close()

    # Resume from ledger in a new coordinator instance
    resumed_coord = ScientificClosedLoopCoordinator.resume_from_ledger(
        db_path=db_file,
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        candidate_pool=cand_pool,
        strategy="expected_improvement",
        random_state=42,
    )

    assert resumed_coord.step_counter == len(init_df) + 2
    assert len(resumed_coord.ledger.list_completed_records()) == len(init_df) + 2

    # Resumed coordinator can seamlessly continue proposing
    rec3, rat3 = resumed_coord.propose_next(n_mc_samples=16)
    assert rec3.stage == ExperimentStage.PROPOSED
    assert len(rat3.render_text()) > 0


def test_cli_demo_execution(tmp_path: Path) -> None:
    out_dir = tmp_path / "demo_output"
    res = run_synthetic_demo(seed=42, steps=3, output_dir=out_dir)

    assert res["status"] == "SUCCESS"
    assert res["ledger_valid"] is True
    assert (out_dir / "proposal_history.jsonl").is_file()
    assert (out_dir / "model_report.json").is_file()
    assert (out_dir / "run_provenance.json").is_file()
    assert (out_dir / "experiment_ledger.db").is_file()
