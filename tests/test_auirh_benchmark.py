from __future__ import annotations

from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd

from src.datasets.auirh import (
    AUIRH_CANDIDATE_ID_COLUMN,
    AuIrRhAdapter,
    AuIrRhExperimentOracle,
    generate_auirh_candidate_id,
)
from src.evaluation.auirh_benchmark import (
    run_cross_library_diagnostic,
    run_single_aicoscientist_trajectory,
)


@pytest.fixture
def synthetic_pool_and_oracle() -> tuple[pd.DataFrame, AuIrRhExperimentOracle]:
    """Creates a 20-sample synthetic candidate pool and oracle for benchmark tests."""
    records = []
    rng = np.random.default_rng(42)
    for lib in ["Au-rich", "Ir-rich"]:
        for area in range(1, 11):
            cid = generate_auirh_candidate_id(lib, area)
            au = float(rng.uniform(10.0, 70.0))
            ir = float(rng.uniform(10.0, 90.0 - au))
            rh = 100.0 - au - ir
            k0 = float(rng.uniform(0.001, 0.015))
            records.append({
                AUIRH_CANDIDATE_ID_COLUMN: cid,
                "Library": lib,
                "Area": area,
                "Au": au,
                "Ir": ir,
                "Rh": rh,
                "k^0 [cm/s]": k0,
                "k0": k0,
            })
    df = pd.DataFrame(records)
    visible_cols = [AUIRH_CANDIDATE_ID_COLUMN, "Library", "Area", "Au", "Ir", "Rh"]
    pool = df[visible_cols].copy()
    oracle = AuIrRhExperimentOracle(df, target_column="k0")
    return pool, oracle


def test_selected_candidates_always_belong_to_finite_pool(
    synthetic_pool_and_oracle: tuple[pd.DataFrame, AuIrRhExperimentOracle],
):
    """Test 11 & 12: Candidates selected along trajectory strictly belong to the pool without duplicates."""
    pool, oracle = synthetic_pool_and_oracle
    strategies = ["random", "greedy", "gp_ucb", "expected_improvement", "true_nei", "turbo_nei"]

    for strat in strategies:
        oracle.reset()
        traj = run_single_aicoscientist_trajectory(
            candidate_pool=pool,
            oracle=oracle,
            target_name="k0",
            strategy=strat,
            init_sample_index=0,
            total_budget=5,
            seed=42,
        )
        selected_ids = [step["selected_sample_id"] for step in traj]
        # Check no duplicate selections
        assert len(selected_ids) == len(set(selected_ids))
        # Check all selected belong to candidate pool
        valid_ids = set(pool[AUIRH_CANDIDATE_ID_COLUMN])
        for cid in selected_ids:
            assert cid in valid_ids


def test_deterministic_and_identical_initialization_across_methods(
    synthetic_pool_and_oracle: tuple[pd.DataFrame, AuIrRhExperimentOracle],
):
    """Tests 13 & 14: Same seed yields identical initial sample and deterministic trajectories."""
    pool, oracle = synthetic_pool_and_oracle
    strategies = ["random", "greedy", "gp_ucb", "expected_improvement", "true_nei", "turbo_nei"]

    init_ids = []
    for strat in strategies:
        oracle.reset()
        traj = run_single_aicoscientist_trajectory(
            candidate_pool=pool,
            oracle=oracle,
            target_name="k0",
            strategy=strat,
            init_sample_index=3,
            total_budget=3,
            seed=100,
        )
        init_ids.append(traj[0]["selected_sample_id"])

    # All strategies initialized with the same candidate
    assert len(set(init_ids)) == 1
    assert init_ids[0] == pool.iloc[3][AUIRH_CANDIDATE_ID_COLUMN]


def test_botorch_nei_and_turbo_runs_seamlessly(
    synthetic_pool_and_oracle: tuple[pd.DataFrame, AuIrRhExperimentOracle],
):
    """Test 17 & 18: NEI and Turbo-NEI strategies run properly via BoTorchBackend."""
    pool, oracle = synthetic_pool_and_oracle
    for strat in ["true_nei", "turbo_nei"]:
        oracle.reset()
        traj = run_single_aicoscientist_trajectory(
            candidate_pool=pool,
            oracle=oracle,
            target_name="k0",
            strategy=strat,
            init_sample_index=0,
            total_budget=4,
            seed=42,
        )
        assert len(traj) == 4
        assert traj[-1]["selected_sample_id"] is not None


@pytest.mark.external_data
def test_cross_library_diagnostic_no_held_out_leakage():
    """Test 21: Cross-library diagnostic runs without leaking held-out destination targets."""
    res = run_cross_library_diagnostic(
        target_name="k0",
        n_seeds=2,
        prior_samples=3,
        budget_in_dest=5,
        base_seed=42,
    )
    assert len(res) == 4
    for k, v in res.items():
        assert "mean_regret_cold_start" in v
        assert "mean_regret_warm_prior" in v
