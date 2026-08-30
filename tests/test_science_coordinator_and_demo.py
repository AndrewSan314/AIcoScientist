import numpy as np
import pandas as pd
import pytest

from src.datasets.base import DatasetSpec, TwoStageModelSpec
from src.optimization.search_space import ContinuousVariable, SearchSpace
from src.science.cli import run_synthetic_demo
from src.science.coordinator import (
    PendingExperimentError,
    ResumeStateMismatchError,
    ScientificClosedLoopCoordinator,
)
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


def test_coordinator_failed_proposal_transactional_rollback(tmp_path: Path) -> None:
    """Verifies that a failed proposal request leaves active optimizer state 100% unchanged."""
    adapter = SyntheticScienceAdapter()
    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    db_test = tmp_path / "trans_test.db"
    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_test,
        strategy="expected_improvement",
        random_state=42,
    )

    step_before = coord.optimizer_state.step
    best_before = coord.optimizer_state.current_best
    history_len_before = len(coord.optimizer_state.history)

    # Trigger proposal failure via conflicting pre_experiment_context
    with pytest.raises(InformationHorizonError):
        coord.propose_next(pre_experiment_context={"x1": 9999.0})

    # Assert optimizer state is untouched
    assert coord.optimizer_state.step == step_before
    assert coord.optimizer_state.current_best == best_before
    assert len(coord.optimizer_state.history) == history_len_before

    # Now make valid proposal and compare with clean control coordinator
    rec_actual, rat_actual = coord.propose_next(n_mc_samples=16)

    db_control = tmp_path / "trans_control.db"
    coord_control = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_control,
        strategy="expected_improvement",
        random_state=42,
    )
    rec_control, rat_control = coord_control.propose_next(n_mc_samples=16)

    assert rec_actual.candidate_id == rec_control.candidate_id
    assert rec_actual.candidate_variables == rec_control.candidate_variables
    assert rat_actual.acquisition_score == pytest.approx(rat_control.acquisition_score, rel=1e-5)


def test_coordinator_pending_proposal_crash_and_exact_resume(tmp_path: Path) -> None:
    """CRITICAL TEST: Verifies that crashing with a pending proposal and resuming produces bit-for-bit identical next step."""
    adapter = SyntheticScienceAdapter()
    oracle = SyntheticExperimentOracle()

    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    # --- Run A: Continuous Uninterrupted Run ---
    db_a = tmp_path / "pending_a.db"
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

    # Step 1
    rec_a1, _ = coord_a.propose_next(n_mc_samples=16)
    coord_a.record_executed(rec_a1.experiment_id)
    c1 = oracle.evaluate_characterization(rec_a1.pre_experiment_features, seed=101)
    coord_a.record_characterization(rec_a1.experiment_id, c1)
    p1 = oracle.evaluate_performance(rec_a1.pre_experiment_features, c1, seed=201)
    coord_a.record_performance(rec_a1.experiment_id, p1)

    # Step 2
    rec_a2, _ = coord_a.propose_next(n_mc_samples=16)
    coord_a.record_executed(rec_a2.experiment_id)
    c2 = oracle.evaluate_characterization(rec_a2.pre_experiment_features, seed=102)
    coord_a.record_characterization(rec_a2.experiment_id, c2)
    p2 = oracle.evaluate_performance(rec_a2.pre_experiment_features, c2, seed=202)
    coord_a.record_performance(rec_a2.experiment_id, p2)

    # Step 3 proposed
    rec_a3, rat_a3 = coord_a.propose_next(n_mc_samples=16)

    # Step 3 executed & completed in run A
    coord_a.record_executed(rec_a3.experiment_id)
    c3 = oracle.evaluate_characterization(rec_a3.pre_experiment_features, seed=103)
    coord_a.record_characterization(rec_a3.experiment_id, c3)
    p3 = oracle.evaluate_performance(rec_a3.pre_experiment_features, c3, seed=203)
    coord_a.record_performance(rec_a3.experiment_id, p3)

    # Step 4 proposed in run A
    rec_a4, rat_a4 = coord_a.propose_next(n_mc_samples=16)
    coord_a.ledger.close()

    # --- Run B: Process Crashes after Step 3 Proposal ---
    db_b = tmp_path / "pending_b.db"
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

    # Step 1
    rec_b1, _ = coord_b.propose_next(n_mc_samples=16)
    coord_b.record_executed(rec_b1.experiment_id)
    coord_b.record_characterization(rec_b1.experiment_id, c1)
    coord_b.record_performance(rec_b1.experiment_id, p1)

    # Step 2
    rec_b2, _ = coord_b.propose_next(n_mc_samples=16)
    coord_b.record_executed(rec_b2.experiment_id)
    coord_b.record_characterization(rec_b2.experiment_id, c2)
    coord_b.record_performance(rec_b2.experiment_id, p2)

    # Step 3 proposed
    rec_b3, rat_b3 = coord_b.propose_next(n_mc_samples=16)
    assert rec_a3.candidate_id == rec_b3.candidate_id
    assert rec_a3.candidate_variables == rec_b3.candidate_variables

    # SIMULATE CRASH: close coordinator B while step 3 is pending in ledger
    coord_b.ledger.close()

    # Resume from ledger with pending experiment
    resumed_b = ScientificClosedLoopCoordinator.resume_from_ledger(
        db_path=db_b,
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        strategy="expected_improvement",
        random_state=42,
    )

    # Assert pending record exists and restored proposal matches exactly
    pending = resumed_b.ledger.list_pending_records()
    assert len(pending) == 1
    assert pending[0].experiment_id == rec_b3.experiment_id
    assert resumed_b._last_proposal is not None
    assert resumed_b._last_proposal.candidate_id == rec_b3.candidate_id

    # Execute & complete the pending step 3 in resumed session
    resumed_b.record_executed(pending[0].experiment_id)
    resumed_b.record_characterization(pending[0].experiment_id, c3)
    resumed_b.record_performance(pending[0].experiment_id, p3)

    # Propose Step 4 in resumed session
    rec_b4, rat_b4 = resumed_b.propose_next(n_mc_samples=16)
    resumed_b.ledger.close()

    # Verify Step 4 candidate and rationale match Run A
    assert rec_a4.candidate_id == rec_b4.candidate_id
    assert rec_a4.candidate_variables == rec_b4.candidate_variables
    assert rat_a4.acquisition_score == pytest.approx(rat_b4.acquisition_score, rel=1e-5)


def test_turbo_proposal_reason_code_survives_resume(tmp_path: Path) -> None:
    adapter = SyntheticScienceAdapter()
    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    db_path = tmp_path / "turbo_test.db"
    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_path,
        strategy="turbo_nei",
        random_state=42,
    )

    rec, rat = coord.propose_next(n_mc_samples=16)
    assert "reason_code" in rec.proposal_metadata
    orig_reason = rec.proposal_metadata["reason_code"]
    coord.ledger.close()

    # Resume from ledger
    resumed = ScientificClosedLoopCoordinator.resume_from_ledger(
        db_path=db_path,
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        strategy="turbo_nei",
        random_state=42,
    )
    assert resumed._last_proposal is not None
    assert resumed._last_proposal.reason_code == orig_reason
    resumed.ledger.close()


def test_component_training_horizons_and_asynchronous_refits(tmp_path: Path) -> None:
    adapter = SyntheticScienceAdapter()
    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    db_path = tmp_path / "horizons.db"
    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_path,
        strategy="expected_improvement",
        random_state=42,
    )

    rec, _ = coord.propose_next(n_mc_samples=16)
    coord.record_executed(rec.experiment_id)

    # 1. Record characterization ONLY
    coord.record_characterization(rec.experiment_id, {"z1": 0.35, "z2": -0.2})

    # Stage A training frame must immediately contain this experiment
    stage_a_df = coord.build_stage_a_training_frame()
    assert rec.experiment_id in list(stage_a_df["experiment_id"])

    # Direct training frame must NOT contain this experiment yet (performance pending)
    direct_df = coord.build_direct_training_frame()
    assert rec.experiment_id not in list(direct_df["experiment_id"])

    # Provenance Stage A tracking reflects the new experiment
    assert rec.experiment_id in coord.model_bundle.provenance.stage_a_training_experiment_ids_per_channel["z1"]

    coord.ledger.close()


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


def test_resume_rebuilds_stage_a_from_characterized_partial_records(tmp_path: Path) -> None:
    adapter = SyntheticScienceAdapter()
    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    db_path = tmp_path / "stage_a_resume.db"
    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_path,
        strategy="expected_improvement",
        random_state=42,
    )

    rec, _ = coord.propose_next(n_mc_samples=16)
    coord.record_executed(rec.experiment_id)
    coord.record_characterization(rec.experiment_id, {"z1": 0.45, "z2": -0.1})

    test_cand = pd.DataFrame([{"x1": 2.5, "x2": 20.0, "x3": 3.0}])
    pred_before = coord.model_bundle.two_stage_model.stage_a.predict(test_cand)
    prov_before_ids = list(coord.model_bundle.provenance.stage_a_training_experiment_ids_per_channel["z1"])
    assert rec.experiment_id in prov_before_ids
    coord.ledger.close()

    # Resume
    resumed = ScientificClosedLoopCoordinator.resume_from_ledger(
        db_path=db_path,
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        strategy="expected_improvement",
        random_state=42,
    )
    prov_after_ids = list(resumed.model_bundle.provenance.stage_a_training_experiment_ids_per_channel["z1"])
    assert rec.experiment_id in prov_after_ids
    assert prov_before_ids == prov_after_ids

    pred_after = resumed.model_bundle.two_stage_model.stage_a.predict(test_cand)
    assert np.allclose(pred_before["z1"]["mean"], pred_after["z1"]["mean"])
    resumed.ledger.close()


def test_resume_rebuilds_direct_from_performance_measured_partial_records(tmp_path: Path) -> None:
    adapter = SyntheticScienceAdapter()
    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    db_path = tmp_path / "direct_resume.db"
    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_path,
        strategy="expected_improvement",
        random_state=42,
    )

    rec, _ = coord.propose_next(n_mc_samples=16)
    coord.record_executed(rec.experiment_id)
    coord.record_performance(rec.experiment_id, {"y": 720.0})

    test_cand = pd.DataFrame([{"x1": 2.5, "x2": 20.0, "x3": 3.0}])
    pred_before, _ = coord.model_bundle.direct_model.predict(test_cand)
    prov_before_ids = list(coord.model_bundle.provenance.direct_training_experiment_ids)
    assert rec.experiment_id in prov_before_ids
    coord.ledger.close()

    # Resume
    resumed = ScientificClosedLoopCoordinator.resume_from_ledger(
        db_path=db_path,
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        strategy="expected_improvement",
        random_state=42,
    )
    prov_after_ids = list(resumed.model_bundle.provenance.direct_training_experiment_ids)
    assert rec.experiment_id in prov_after_ids
    assert prov_before_ids == prov_after_ids

    pred_after, _ = resumed.model_bundle.direct_model.predict(test_cand)
    assert np.allclose(pred_before, pred_after)
    resumed.ledger.close()


def test_historical_ingestion_with_missing_data_semantics(tmp_path: Path) -> None:
    spec = DatasetSpec(
        name="missing_hist",
        id_column="exp_id",
        candidate_id_column="cand_id",
        feature_columns=["x1", "x2", "z1", "z2"],
        target_column="y",
        pre_experiment_features=["x1", "x2"],
        candidate_variables=["x1", "x2"],
        post_experiment_characterization=["z1", "z2"],
        targets=["y"],
    )
    two_stage_spec = TwoStageModelSpec(
        dataset_name="missing_hist",
        process_features=["x1", "x2"],
        characterization_targets=["z1", "z2"],
        performance_targets=["y"],
    )
    space = SearchSpace(
        name="space",
        variables=[
            ContinuousVariable(name="x1", lower=1.0, upper=10.0),
            ContinuousVariable(name="x2", lower=10.0, upper=50.0),
        ],
    )
    initial_df = pd.DataFrame([
        {"exp_id": "EXP_00", "cand_id": "C0", "x1": 1.0, "x2": 10.0, "z1": 0.1, "z2": 0.2, "y": 500.0},  # COMPLETED
        {"exp_id": "EXP_01", "cand_id": "C1", "x1": 2.0, "x2": 20.0, "z1": 0.2, "z2": np.nan, "y": 550.0},  # PERFORMANCE_MEASURED
        {"exp_id": "EXP_02", "cand_id": "C2", "x1": 3.0, "x2": 30.0, "z1": 0.3, "z2": 0.4, "y": np.nan},  # CHARACTERIZED
        {"exp_id": "EXP_03", "cand_id": "C3", "x1": 4.0, "x2": 40.0, "z1": np.nan, "z2": np.nan, "y": np.nan},  # EXECUTED
        {"exp_id": "EXP_04", "cand_id": "C4", "x1": 5.0, "x2": 50.0, "z1": 0.5, "z2": 0.6, "y": 600.0},  # COMPLETED
    ])
    cand_pool = pd.DataFrame([{"cand_id": f"CP_{i}", "x1": float(i), "x2": float(i * 10)} for i in range(1, 10)])

    db_file = tmp_path / "missing_hist.db"
    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=spec,
        two_stage_spec=two_stage_spec,
        initial_data=initial_df,
        candidate_pool=cand_pool,
        search_space=space,
        db_path=db_file,
        strategy="expected_improvement",
        random_state=42,
    )

    # 1. Assert stages in ledger
    assert coord.ledger.get_record("EXP_00").stage == ExperimentStage.COMPLETED
    assert coord.ledger.get_record("EXP_01").stage == ExperimentStage.PERFORMANCE_MEASURED
    assert coord.ledger.get_record("EXP_02").stage == ExperimentStage.CHARACTERIZED
    assert coord.ledger.get_record("EXP_03").stage == ExperimentStage.EXECUTED
    assert coord.ledger.get_record("EXP_04").stage == ExperimentStage.COMPLETED

    # 2. Check training views
    stage_a_df = coord.build_stage_a_training_frame()
    assert set(stage_a_df["experiment_id"]) == {"EXP_00", "EXP_01", "EXP_02", "EXP_04"}

    direct_df = coord.build_direct_training_frame()
    assert set(direct_df["experiment_id"]) == {"EXP_00", "EXP_01", "EXP_04"}

    opt_df = coord.build_optimizer_training_frame()
    assert set(opt_df["experiment_id"]) == {"EXP_00", "EXP_04"}

    coord.ledger.close()


def test_stage_b_per_target_independent_missingness(tmp_path: Path) -> None:
    spec = DatasetSpec(
        name="multi_target_hist",
        id_column="exp_id",
        candidate_id_column="cand_id",
        feature_columns=["x1", "z1"],
        target_column="y1",
        pre_experiment_features=["x1"],
        candidate_variables=["x1"],
        post_experiment_characterization=["z1"],
        targets=["y1", "y2"],
    )
    two_stage_spec = TwoStageModelSpec(
        dataset_name="multi_target_hist",
        process_features=["x1"],
        characterization_targets=["z1"],
        performance_targets=["y1", "y2"],
    )
    space = SearchSpace(name="space", variables=[ContinuousVariable(name="x1", lower=1.0, upper=10.0)])

    # Row 0 & 1 have both y1 & y2; Row 2 has only y2
    initial_df = pd.DataFrame([
        {"exp_id": "EXP_00", "cand_id": "C0", "x1": 1.0, "z1": 0.1, "y1": 500.0, "y2": 100.0},
        {"exp_id": "EXP_01", "cand_id": "C1", "x1": 2.0, "z1": 0.2, "y1": 550.0, "y2": 110.0},
        {"exp_id": "EXP_02", "cand_id": "C2", "x1": 3.0, "z1": 0.3, "y1": np.nan, "y2": 120.0},
    ])
    cand_pool = pd.DataFrame([{"cand_id": f"CP_{i}", "x1": float(i)} for i in range(1, 10)])

    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=spec,
        two_stage_spec=two_stage_spec,
        initial_data=initial_df,
        candidate_pool=cand_pool,
        search_space=space,
        db_path=tmp_path / "stage_b_test.db",
        strategy="expected_improvement",
        random_state=42,
    )

    # Assert Stage B per-target sample counts
    assert coord.model_bundle.two_stage_model.stage_b.training_sample_counts["y1"] == 2
    assert coord.model_bundle.two_stage_model.stage_b.training_sample_counts["y2"] == 3

    # Assert provenance records per target
    prov = coord.model_bundle.provenance
    assert prov.stage_b_training_experiment_ids_per_target["y1"] == ["EXP_00", "EXP_01"]
    assert prov.stage_b_training_experiment_ids_per_target["y2"] == ["EXP_00", "EXP_01", "EXP_02"]
    coord.ledger.close()


def test_failed_and_cancelled_record_excluded_from_model_training(tmp_path: Path) -> None:
    adapter = SyntheticScienceAdapter()
    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    db_path = tmp_path / "failed_excl.db"
    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_path,
        strategy="expected_improvement",
        random_state=42,
    )

    rec, _ = coord.propose_next(n_mc_samples=16)
    coord.record_executed(rec.experiment_id)
    coord.record_performance(rec.experiment_id, {"y": 750.0})

    # Initially in direct training frame
    assert rec.experiment_id in list(coord.build_direct_training_frame()["experiment_id"])

    # Now mark failed
    coord.record_failed(rec.experiment_id, "Filter damaged during testing")

    # Excluded from direct training frame
    assert rec.experiment_id not in list(coord.build_direct_training_frame()["experiment_id"])
    assert rec.experiment_id not in coord.model_bundle.provenance.direct_training_experiment_ids
    coord.ledger.close()


def test_record_additional_performance_on_completed_experiment(tmp_path: Path) -> None:
    spec = DatasetSpec(
        name="sec_perf_test",
        id_column="exp_id",
        candidate_id_column="cand_id",
        feature_columns=["x1", "z1"],
        target_column="y1",
        pre_experiment_features=["x1"],
        candidate_variables=["x1"],
        post_experiment_characterization=["z1"],
        targets=["y1", "y2"],
    )
    two_stage_spec = TwoStageModelSpec(
        dataset_name="sec_perf_test",
        process_features=["x1"],
        characterization_targets=["z1"],
        performance_targets=["y1", "y2"],
    )
    space = SearchSpace(name="space", variables=[ContinuousVariable(name="x1", lower=1.0, upper=10.0)])
    initial_df = pd.DataFrame([
        {"exp_id": "EXP_00", "cand_id": "C0", "x1": 1.0, "z1": 0.1, "y1": 500.0, "y2": 100.0},
        {"exp_id": "EXP_01", "cand_id": "C1", "x1": 2.0, "z1": 0.2, "y1": 550.0, "y2": 110.0},
    ])
    cand_pool = pd.DataFrame([{"cand_id": f"CP_{i}", "x1": float(i)} for i in range(1, 10)])

    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=spec,
        two_stage_spec=two_stage_spec,
        initial_data=initial_df,
        candidate_pool=cand_pool,
        search_space=space,
        db_path=tmp_path / "sec_perf.db",
        strategy="expected_improvement",
        random_state=42,
    )

    rec, _ = coord.propose_next(n_mc_samples=16)
    coord.record_executed(rec.experiment_id)
    coord.record_characterization(rec.experiment_id, {"z1": 0.25})
    # Complete with primary target y1
    rec_completed = coord.record_performance(rec.experiment_id, {"y1": 600.0})
    assert rec_completed.stage == ExperimentStage.COMPLETED

    # Add secondary target y2 after completion
    rec_updated = coord.record_additional_performance(rec.experiment_id, {"y2": 130.0})
    assert rec_updated.stage == ExperimentStage.COMPLETED
    assert rec_updated.performance["y2"] == 130.0
    assert rec.experiment_id in coord.model_bundle.provenance.stage_b_training_experiment_ids_per_target["y2"]

    coord.ledger.close()


def test_resume_schema_mismatch_raises_error(tmp_path: Path) -> None:
    adapter = SyntheticScienceAdapter()
    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    db_path = tmp_path / "mismatch.db"
    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_path,
        strategy="expected_improvement",
        random_state=42,
    )
    coord.ledger.close()

    # Altered spec with different target column
    mismatched_spec = DatasetSpec(
        name=adapter.spec.name,
        id_column=adapter.spec.id_column,
        candidate_id_column=adapter.spec.candidate_id_column,
        feature_columns=adapter.spec.feature_columns,
        target_column="completely_different_target",
        pre_experiment_features=adapter.spec.pre_experiment_features,
        candidate_variables=adapter.spec.candidate_variables,
        targets=["completely_different_target"],
    )

    with pytest.raises(ResumeStateMismatchError, match="Snapshot target column"):
        ScientificClosedLoopCoordinator.resume_from_ledger(
            db_path=db_path,
            spec=mismatched_spec,
            two_stage_spec=adapter.two_stage_spec,
            candidate_pool=cand_pool,
            search_space=adapter.search_space,
            strategy="expected_improvement",
            random_state=42,
        )


def test_proposal_metadata_includes_optimizer_state_hash(tmp_path: Path) -> None:
    adapter = SyntheticScienceAdapter()
    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=40, seed=42)

    coord = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=tmp_path / "opt_hash.db",
        strategy="expected_improvement",
        random_state=42,
    )

    rec, _ = coord.propose_next(n_mc_samples=16)
    assert "optimizer_state_hash" in rec.proposal_metadata
    assert len(rec.proposal_metadata["optimizer_state_hash"]) == 64
    assert "search_space_fingerprint" in rec.proposal_metadata
    coord.ledger.close()

