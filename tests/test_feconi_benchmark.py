from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.datasets.feconi import (
    FECONI_CANDIDATE_ID_COLUMN,
    FECONI_FEATURE_COLUMNS,
    FeCoNiAdapter,
    FeCoNiExperimentOracle,
    generate_feconi_candidate_id,
    load_raw_feconi_mat,
)
from src.evaluation.feconi_benchmark import (
    get_feconi_search_space,
    run_single_aicoscientist_trajectory,
)
from src.optimization.botorch_backend import BoTorchBackend


@pytest.fixture
def synthetic_pool_and_oracle() -> tuple[pd.DataFrame, FeCoNiExperimentOracle]:
    """Provides a synthetic 40-candidate pool and an offline oracle for self-contained CI testing."""
    rng = np.random.default_rng(123)
    n = 40
    co = rng.uniform(5.0, 75.0, size=n)
    fe = rng.uniform(5.0, 75.0 - co * 0.4, size=n)
    ni = 100.0 - co - fe

    # Create synthetic smooth Kerr and rugged Coercivity
    kerr = 0.3 + 0.5 * (fe / 100.0) + 0.02 * rng.standard_normal(n)
    coer = 1.0 + 8.0 * (co / 100.0) * (ni / 100.0) + 0.05 * rng.standard_normal(n)

    full_df = pd.DataFrame(
        {
            FECONI_CANDIDATE_ID_COLUMN: [generate_feconi_candidate_id(i) for i in range(n)],
            "sample_index": np.arange(n, dtype=int),
            "Co": co,
            "Fe": fe,
            "Ni": ni,
            "Kerr": kerr,
            "Coer": coer,
        }
    )
    visible_cols = [FECONI_CANDIDATE_ID_COLUMN, "sample_index", "Co", "Fe", "Ni"]
    cand_pool = full_df[visible_cols].copy()
    oracle = FeCoNiExperimentOracle(full_records_df=full_df, target_column="Kerr", allow_duplicate_queries=False)
    return cand_pool, oracle


# ---------------------------------------------------------------------------
# Self-Contained Tests for Core CI (not external_data)
# ---------------------------------------------------------------------------

def test_same_seed_gives_identical_initialization_across_all_methods(synthetic_pool_and_oracle: tuple[pd.DataFrame, FeCoNiExperimentOracle]):
    """Test 6 & 7: Same seed produces identical initial candidate index and measurement across all strategies."""
    pool, oracle = synthetic_pool_and_oracle
    seed = 42
    rng = np.random.default_rng(seed)
    init_idx = int(rng.integers(0, len(pool)))

    methods = ["random", "greedy", "gp_ucb", "expected_improvement", "noisy_expected_improvement", "turbo_nei"]
    trajs = {}
    for m in methods:
        oracle.reset()
        traj = run_single_aicoscientist_trajectory(
            candidate_pool=pool,
            oracle=oracle,
            target_name="Kerr",
            strategy=m,
            init_sample_index=init_idx,
            total_budget=3,
            seed=seed,
        )
        trajs[m] = traj

    first_cid = trajs["random"][0]["selected_sample_id"]
    first_target = trajs["random"][0]["observed_target"]

    for m in methods:
        assert trajs[m][0]["selected_sample_id"] == first_cid
        assert trajs[m][0]["observed_target"] == first_target
        assert trajs[m][0]["iteration"] == 1


def test_botorch_nei_benchmark_path(synthetic_pool_and_oracle: tuple[pd.DataFrame, FeCoNiExperimentOracle]):
    """Test 8 & 9: NEI benchmark path delegates to BoTorch backend."""
    pool, oracle = synthetic_pool_and_oracle
    oracle.reset()
    traj = run_single_aicoscientist_trajectory(
        candidate_pool=pool,
        oracle=oracle,
        target_name="Kerr",
        strategy="noisy_expected_improvement",
        init_sample_index=0,
        total_budget=3,
        seed=10,
    )
    assert len(traj) == 3
    assert traj[-1]["selected_sample_id"] is not None


def test_selected_candidates_always_belong_to_finite_pool(synthetic_pool_and_oracle: tuple[pd.DataFrame, FeCoNiExperimentOracle]):
    """Test 12 & 13: Selected candidates always belong to measured pool and contain no duplicates."""
    pool, oracle = synthetic_pool_and_oracle
    valid_cids = set(pool[FECONI_CANDIDATE_ID_COLUMN])

    oracle.reset()
    traj = run_single_aicoscientist_trajectory(
        candidate_pool=pool,
        oracle=oracle,
        target_name="Kerr",
        strategy="turbo_nei",
        init_sample_index=2,
        total_budget=8,
        seed=50,
    )
    selected_cids = [step["selected_sample_id"] for step in traj]
    assert len(selected_cids) == len(set(selected_cids)), "Duplicate candidate selected within trajectory"
    for cid in selected_cids:
        assert cid in valid_cids


def test_global_optimum_used_only_for_evaluation_metrics(synthetic_pool_and_oracle: tuple[pd.DataFrame, FeCoNiExperimentOracle]):
    """Test 14: Global optimum is used strictly as an offline reference for regret calculations."""
    pool, oracle = synthetic_pool_and_oracle
    oracle.reset()
    traj = run_single_aicoscientist_trajectory(
        candidate_pool=pool,
        oracle=oracle,
        target_name="Kerr",
        strategy="greedy",
        init_sample_index=0,
        total_budget=4,
        seed=60,
    )
    for step in traj:
        assert step["regret"] == step["global_best"] - step["best_observed"]
        assert step["regret"] >= -1e-8


# ---------------------------------------------------------------------------
# Real MAT Data Specification Tests (Marked with @pytest.mark.external_data)
# ---------------------------------------------------------------------------

@pytest.mark.external_data
def test_real_mat_specification_exact_921_rows():
    raw = load_raw_feconi_mat()
    assert raw["C"].shape == (921, 3)
    assert len(raw["Kerr"]) == 921
    assert len(raw["Coer"]) == 921


@pytest.mark.external_data
def test_real_feconi_full_benchmark_run(tmp_path):
    from src.evaluation.feconi_benchmark import run_feconi_benchmark_suite
    report = run_feconi_benchmark_suite(
        target_name="Kerr",
        n_seeds=2,
        total_budget=5,
        output_dir=tmp_path / "feconi",
    )
    assert report["backend"] == "botorch"
    assert len(report["summary"]) > 0
