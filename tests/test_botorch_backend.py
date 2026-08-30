from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.datasets.base import DatasetSpec, TwoStageModelSpec
from src.optimization.backend import OptimizerBackend
from src.optimization.botorch_backend import BoTorchBackend
from src.optimization.finite_pool import FiniteCandidatePool
from src.optimization.objective import OptimizationObjective
from src.optimization.proposal import CandidateProposal
from src.science.coordinator import (
    PendingExperimentError,
    ScientificClosedLoopCoordinator,
)
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.synthetic import SyntheticExperimentOracle, SyntheticScienceAdapter


@pytest.fixture
def sample_candidate_pool() -> pd.DataFrame:
    """Generates a synthetic 2D candidate pool with 50 discrete points."""
    rng = np.random.default_rng(42)
    n = 50
    x1 = rng.uniform(0.0, 10.0, size=n)
    x2 = rng.uniform(0.0, 10.0, size=n)
    # Ground truth function: Branin-like synthetic target
    y = np.sin(x1) + np.cos(x2) + 0.1 * (x1 - 5.0) ** 2
    return pd.DataFrame(
        {
            "candidate_id": [f"CAND_{i:03d}" for i in range(n)],
            "x1": x1,
            "x2": x2,
            "target": y,
        }
    )


# ---------------------------------------------------------------------------
# Requirement 1: BoTorchBackend implements OptimizerBackend protocol
# ---------------------------------------------------------------------------
def test_botorch_backend_implements_protocol() -> None:
    backend = BoTorchBackend()
    assert isinstance(backend, OptimizerBackend)
    assert backend.name == "botorch"
    assert isinstance(backend.version, str) and len(backend.version) > 0


# ---------------------------------------------------------------------------
# Requirement 2 & 3: Initial observation ingestion & Random strategy
# ---------------------------------------------------------------------------
def test_random_strategy_selects_unseen_candidates(sample_candidate_pool: pd.DataFrame) -> None:
    backend = BoTorchBackend(default_strategy="random")
    cand_pool = sample_candidate_pool[["candidate_id", "x1", "x2"]]
    obs_df = sample_candidate_pool.iloc[:5].copy()

    proposals = backend.propose(
        observations=obs_df,
        candidate_pool=cand_pool,
        objective="target",
        feature_columns=["x1", "x2"],
        candidate_id_column="candidate_id",
        n=3,
        seed=123,
    )
    assert len(proposals) == 3
    observed_cids = set(obs_df["candidate_id"])
    for p in proposals:
        assert isinstance(p, CandidateProposal)
        assert p.candidate_id not in observed_cids
        assert p.acquisition_name == "random"
        assert p.backend_name == "botorch"


# ---------------------------------------------------------------------------
# Requirement 4: Greedy / Posterior Mean strategy
# ---------------------------------------------------------------------------
def test_greedy_strategy_selects_valid_unseen(sample_candidate_pool: pd.DataFrame) -> None:
    backend = BoTorchBackend()
    cand_pool = sample_candidate_pool[["candidate_id", "x1", "x2"]]
    obs_df = sample_candidate_pool.iloc[:8].copy()

    proposals = backend.propose(
        observations=obs_df,
        candidate_pool=cand_pool,
        objective="target",
        feature_columns=["x1", "x2"],
        candidate_id_column="candidate_id",
        strategy="greedy",
        n=1,
        seed=42,
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.candidate_id not in set(obs_df["candidate_id"])
    assert p.acquisition_name == "greedy"
    assert isinstance(p.predicted_mean, float)
    assert isinstance(p.predicted_std, float)
    assert p.predicted_std >= 0.0


# ---------------------------------------------------------------------------
# Requirement 5: GP-UCB strategy with analytic UCB
# ---------------------------------------------------------------------------
def test_gp_ucb_strategy(sample_candidate_pool: pd.DataFrame) -> None:
    backend = BoTorchBackend()
    cand_pool = sample_candidate_pool[["candidate_id", "x1", "x2"]]
    obs_df = sample_candidate_pool.iloc[:8].copy()

    proposals = backend.propose(
        observations=obs_df,
        candidate_pool=cand_pool,
        objective="target",
        feature_columns=["x1", "x2"],
        candidate_id_column="candidate_id",
        strategy="gp_ucb",
        beta=2.0,
        n=1,
        seed=42,
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.candidate_id not in set(obs_df["candidate_id"])
    assert p.acquisition_name == "gp_ucb"
    assert p.acquisition_score > -1e9


# ---------------------------------------------------------------------------
# Requirement 6: Expected Improvement (EI / LogEI)
# ---------------------------------------------------------------------------
def test_expected_improvement_strategy(sample_candidate_pool: pd.DataFrame) -> None:
    backend = BoTorchBackend()
    cand_pool = sample_candidate_pool[["candidate_id", "x1", "x2"]]
    obs_df = sample_candidate_pool.iloc[:8].copy()

    proposals = backend.propose(
        observations=obs_df,
        candidate_pool=cand_pool,
        objective="target",
        feature_columns=["x1", "x2"],
        candidate_id_column="candidate_id",
        strategy="ei",
        n=1,
        seed=42,
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.candidate_id not in set(obs_df["candidate_id"])
    assert p.acquisition_name == "ei"


# ---------------------------------------------------------------------------
# Requirement 7: Noisy Expected Improvement (NEI / qLogNEI)
# ---------------------------------------------------------------------------
def test_noisy_expected_improvement_strategy(sample_candidate_pool: pd.DataFrame) -> None:
    backend = BoTorchBackend()
    cand_pool = sample_candidate_pool[["candidate_id", "x1", "x2"]]
    obs_df = sample_candidate_pool.iloc[:8].copy()

    proposals = backend.propose(
        observations=obs_df,
        candidate_pool=cand_pool,
        objective="target",
        feature_columns=["x1", "x2"],
        candidate_id_column="candidate_id",
        strategy="nei",
        n=1,
        seed=42,
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.candidate_id not in set(obs_df["candidate_id"])
    assert p.acquisition_name == "nei"


# ---------------------------------------------------------------------------
# Requirement 8: Joint Thompson Sampling
# ---------------------------------------------------------------------------
def test_joint_thompson_sampling_strategy(sample_candidate_pool: pd.DataFrame) -> None:
    backend = BoTorchBackend()
    cand_pool = sample_candidate_pool[["candidate_id", "x1", "x2"]]
    obs_df = sample_candidate_pool.iloc[:8].copy()

    p1 = backend.propose(
        observations=obs_df,
        candidate_pool=cand_pool,
        objective="target",
        strategy="thompson",
        seed=42,
    )
    p2 = backend.propose(
        observations=obs_df,
        candidate_pool=cand_pool,
        objective="target",
        strategy="thompson",
        seed=42,
    )
    assert len(p1) == 1
    assert len(p2) == 1
    # Deterministic with fixed seed
    assert p1[0].candidate_id == p2[0].candidate_id
    assert p1[0].acquisition_name == "thompson"


# ---------------------------------------------------------------------------
# Requirement 9: NEVER returns already observed candidate IDs
# ---------------------------------------------------------------------------
def test_never_returns_observed_candidates(sample_candidate_pool: pd.DataFrame) -> None:
    backend = BoTorchBackend()
    cand_pool = sample_candidate_pool[["candidate_id", "x1", "x2"]]
    observed_cids = set(sample_candidate_pool["candidate_id"].iloc[:20])
    obs_df = sample_candidate_pool[sample_candidate_pool["candidate_id"].isin(observed_cids)].copy()

    for strat in ["random", "greedy", "gp_ucb", "ei", "nei", "thompson"]:
        props = backend.propose(
            observations=obs_df,
            candidate_pool=cand_pool,
            objective="target",
            strategy=strat,
            n=5,
            seed=42,
        )
        for p in props:
            assert p.candidate_id not in observed_cids, f"Strategy {strat} returned observed candidate {p.candidate_id}"


# ---------------------------------------------------------------------------
# Requirement 10: Custom beta changes UCB trade-off
# ---------------------------------------------------------------------------
def test_custom_beta_alters_ucb_scores(sample_candidate_pool: pd.DataFrame) -> None:
    backend = BoTorchBackend()
    cand_pool = sample_candidate_pool[["candidate_id", "x1", "x2"]]
    obs_df = sample_candidate_pool.iloc[:8].copy()

    p_low = backend.propose(
        observations=obs_df,
        candidate_pool=cand_pool,
        objective="target",
        strategy="gp_ucb",
        beta=0.1,
        n=len(cand_pool) - 8,
        seed=42,
    )
    p_high = backend.propose(
        observations=obs_df,
        candidate_pool=cand_pool,
        objective="target",
        strategy="gp_ucb",
        beta=10.0,
        n=len(cand_pool) - 8,
        seed=42,
    )
    scores_low = [p.acquisition_value for p in p_low]
    scores_high = [p.acquisition_value for p in p_high]
    assert scores_low != scores_high


# ---------------------------------------------------------------------------
# Requirement 11: Scale Invariance Test (y' = a * y + b, a > 0)
# ---------------------------------------------------------------------------
def test_scale_invariance_under_positive_affine_transformation(sample_candidate_pool: pd.DataFrame) -> None:
    backend = BoTorchBackend()
    cand_pool = sample_candidate_pool[["candidate_id", "x1", "x2"]]

    obs_orig = sample_candidate_pool.iloc[:10].copy()
    a = 15.3
    b = -42.7
    obs_scaled = obs_orig.copy()
    obs_scaled["target"] = a * obs_orig["target"] + b

    # Greedy ranking
    p_orig = backend.propose(
        observations=obs_orig,
        candidate_pool=cand_pool,
        objective="target",
        strategy="greedy",
        n=5,
        seed=42,
    )
    p_scaled = backend.propose(
        observations=obs_scaled,
        candidate_pool=cand_pool,
        objective="target",
        strategy="greedy",
        n=5,
        seed=42,
    )
    cids_orig = [p.candidate_id for p in p_orig]
    cids_scaled = [p.candidate_id for p in p_scaled]
    assert cids_orig == cids_scaled, f"Greedy ranking changed under affine scaling: {cids_orig} vs {cids_scaled}"


# ---------------------------------------------------------------------------
# Requirement 12: Batch proposal returns top n distinct candidates in sorted order
# ---------------------------------------------------------------------------
def test_batch_proposal_returns_top_n_sorted(sample_candidate_pool: pd.DataFrame) -> None:
    backend = BoTorchBackend()
    cand_pool = sample_candidate_pool[["candidate_id", "x1", "x2"]]
    obs_df = sample_candidate_pool.iloc[:6].copy()

    props = backend.propose(
        observations=obs_df,
        candidate_pool=cand_pool,
        objective="target",
        strategy="gp_ucb",
        n=4,
        seed=42,
    )
    assert len(props) == 4
    cids = [p.candidate_id for p in props]
    assert len(cids) == len(set(cids))
    scores = [p.acquisition_value for p in props]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Requirement 13: Minimization sense (minimize=True) properly inverts target
# ---------------------------------------------------------------------------
def test_minimization_sense(sample_candidate_pool: pd.DataFrame) -> None:
    backend = BoTorchBackend()
    cand_pool = sample_candidate_pool[["candidate_id", "x1", "x2"]]
    obs_df = sample_candidate_pool.iloc[:10].copy()

    obj_max = OptimizationObjective(target_name="target", minimize=False)
    obj_min = OptimizationObjective(target_name="target", minimize=True)

    p_max = backend.propose(
        observations=obs_df,
        candidate_pool=cand_pool,
        objective=obj_max,
        strategy="greedy",
        seed=42,
    )
    p_min = backend.propose(
        observations=obs_df,
        candidate_pool=cand_pool,
        objective=obj_min,
        strategy="greedy",
        seed=42,
    )
    assert p_max[0].candidate_id != p_min[0].candidate_id
    assert p_min[0].predicted_mean < p_max[0].predicted_mean


# ---------------------------------------------------------------------------
# Requirement 14 & 15: Science Coordinator & Ledger Snapshots with BoTorch
# ---------------------------------------------------------------------------
def test_science_coordinator_with_botorch_backend(tmp_path: Path) -> None:
    db_file = tmp_path / "botorch_coord.db"
    adapter = SyntheticScienceAdapter()
    oracle = SyntheticExperimentOracle()

    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=30, seed=42)

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

    rec, rat = coord.propose_next(n_mc_samples=32)
    assert rec.stage == ExperimentStage.PROPOSED
    assert rat.experiment_id == rec.experiment_id
    assert rat.candidate_id == rec.candidate_id

    # Verify ledger snapshot exists
    snap = coord.ledger.get_latest_verified_optimizer_snapshot()
    assert snap is not None


# ---------------------------------------------------------------------------
# Requirement 16: Two-stage scientific modeling runs end-to-end
# ---------------------------------------------------------------------------
def test_two_stage_modeling_end_to_end(tmp_path: Path) -> None:
    db_file = tmp_path / "two_stage.db"
    adapter = SyntheticScienceAdapter()
    oracle = SyntheticExperimentOracle()

    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=20, seed=42)

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

    rec, rat = coord.propose_next(n_mc_samples=32)
    coord.record_executed(rec.experiment_id)
    chars = oracle.evaluate_characterization(rec.pre_experiment_features, seed=10)
    coord.record_characterization(rec.experiment_id, chars)
    perf = oracle.evaluate_performance(rec.pre_experiment_features, chars, seed=20)
    completed = coord.record_performance(rec.experiment_id, perf)
    assert completed.stage == ExperimentStage.COMPLETED


# ---------------------------------------------------------------------------
# Requirement 17: Replay and state reconstruction from ledger
# ---------------------------------------------------------------------------
def test_resume_and_rebuild_from_ledger(tmp_path: Path) -> None:
    db_file = tmp_path / "resume_test.db"
    adapter = SyntheticScienceAdapter()
    oracle = SyntheticExperimentOracle()

    init_df = adapter.load_initial_dataset(n_samples=8, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=20, seed=42)

    coord1 = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        db_path=db_file,
        strategy="expected_improvement",
        random_state=42,
    )
    rec1, _ = coord1.propose_next()
    coord1.record_executed(rec1.experiment_id)
    chars = oracle.evaluate_characterization(rec1.pre_experiment_features, seed=11)
    coord1.record_characterization(rec1.experiment_id, chars)
    perf = oracle.evaluate_performance(rec1.pre_experiment_features, chars, seed=22)
    coord1.record_performance(rec1.experiment_id, perf)

    # Resume in a second coordinator instance from same ledger
    coord2 = ScientificClosedLoopCoordinator.resume_from_ledger(
        db_path=db_file,
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        candidate_pool=cand_pool,
        search_space=adapter.search_space,
        strategy="expected_improvement",
        random_state=42,
    )
    assert len(coord2.ledger.list_completed_records()) == len(coord1.ledger.list_completed_records())


# ---------------------------------------------------------------------------
# Requirement 18: Information horizon firewall
# ---------------------------------------------------------------------------
def test_information_horizon_firewall(tmp_path: Path) -> None:
    db_file = tmp_path / "firewall_test.db"
    adapter = SyntheticScienceAdapter()

    init_df = adapter.load_initial_dataset(n_samples=6, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=20, seed=42)

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
    # Attempting to propose next with conflicting pre-experiment controllable feature raises error
    from src.science.validation import InformationHorizonError
    with pytest.raises(InformationHorizonError):
        coord.propose_next(pre_experiment_context={"temp": 999.0})


# ---------------------------------------------------------------------------
# Requirement 19: Asynchronous lifecycle functions seamlessly
# ---------------------------------------------------------------------------
def test_asynchronous_lifecycle_stages(tmp_path: Path) -> None:
    db_file = tmp_path / "async_stages.db"
    adapter = SyntheticScienceAdapter()
    oracle = SyntheticExperimentOracle()

    init_df = adapter.load_initial_dataset(n_samples=6, seed=42)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=20, seed=42)

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

    rec, _ = coord.propose_next()
    assert rec.stage == ExperimentStage.PROPOSED
    rec = coord.record_executed(rec.experiment_id)
    assert rec.stage == ExperimentStage.EXECUTED
    chars = oracle.evaluate_characterization(rec.pre_experiment_features, seed=1)
    rec = coord.record_characterization(rec.experiment_id, chars)
    assert rec.stage == ExperimentStage.CHARACTERIZED
    perf = oracle.evaluate_performance(rec.pre_experiment_features, chars, seed=2)
    rec = coord.record_performance(rec.experiment_id, perf)
    assert rec.stage == ExperimentStage.COMPLETED


# ---------------------------------------------------------------------------
# Requirement 20: Finite Candidate Pool identity preservation
# ---------------------------------------------------------------------------
def test_finite_candidate_pool_identity_preservation() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2", "C3"],
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [10.0, 20.0, 30.0],
            "extra_meta": ["alpha", "beta", "gamma"],
        }
    )
    pool = FiniteCandidatePool(df, feature_columns=["feat_a", "feat_b"], id_column="candidate_id")
    assert len(pool) == 3
    assert pool.get_candidate_id(0) == "C1"
    assert pool.get_candidate_id(2) == "C3"
    assert pool.get_metadata(1)["extra_meta"] == "beta"

    unseen = pool.filter_unseen([{"candidate_id": "C2"}])
    assert len(unseen) == 2
    assert unseen.get_candidate_id(0) == "C1"
    assert unseen.get_candidate_id(1) == "C3"
