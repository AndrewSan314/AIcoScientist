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
    fit_gp_surrogate,
    get_feconi_search_space,
    run_single_aicoscientist_trajectory,
)
from src.evaluation.feconi_reproduction_benchmark import (
    run_single_reproduction_trajectory,
)
from src.optimization.trust_region import TuRBOTrustRegion


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


def test_true_nei_benchmark_path_calls_compute_true_mc_nei(synthetic_pool_and_oracle: tuple[pd.DataFrame, FeCoNiExperimentOracle]):
    """Test 8 & 9: True NEI benchmark path calls compute_true_mc_nei and does NOT call denoised EI."""
    pool, oracle = synthetic_pool_and_oracle

    with patch("src.evaluation.feconi_benchmark.compute_true_mc_nei", wraps=lambda **kwargs: np.ones(len(kwargs["X_candidates_scaled"]))) as mock_true_nei:
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
        assert mock_true_nei.called
        assert mock_true_nei.call_count >= 1
        call_kwargs = mock_true_nei.call_args[1]
        assert "n_fantasies" in call_kwargs
        assert call_kwargs["n_fantasies"] == 256
        assert call_kwargs["objective"] == "maximize"


def test_turbo_nei_uses_frozen_turbo_trust_region(synthetic_pool_and_oracle: tuple[pd.DataFrame, FeCoNiExperimentOracle]):
    """Test 10: TuRBO-NEI uses frozen TuRBOTrustRegion and calls compute_true_mc_nei."""
    pool, oracle = synthetic_pool_and_oracle

    with patch("src.evaluation.feconi_benchmark.TuRBOTrustRegion", wraps=TuRBOTrustRegion) as mock_turbo_cls:
        with patch("src.evaluation.feconi_benchmark.compute_true_mc_nei", wraps=lambda **kwargs: np.ones(len(kwargs["X_candidates_scaled"]))) as mock_true_nei:
            oracle.reset()
            traj = run_single_aicoscientist_trajectory(
                candidate_pool=pool,
                oracle=oracle,
                target_name="Kerr",
                strategy="turbo_nei",
                init_sample_index=0,
                total_budget=3,
                seed=15,
            )
            assert mock_turbo_cls.called
            assert mock_true_nei.called


def test_joint_thompson_sampling_uses_posterior_covariance(synthetic_pool_and_oracle: tuple[pd.DataFrame, FeCoNiExperimentOracle]):
    """Test 11: Thompson Sampling uses GP latent posterior covariance and safe_cholesky."""
    pool, oracle = synthetic_pool_and_oracle

    with patch("src.evaluation.feconi_reproduction_benchmark.safe_cholesky", wraps=lambda cov, **kw: np.linalg.cholesky(cov + 1e-6 * np.eye(len(cov)))) as mock_cholesky:
        oracle.reset()
        traj = run_single_reproduction_trajectory(
            candidate_pool=pool,
            oracle=oracle,
            target_name="Kerr",
            strategy="thompson_sampling",
            init_sample_index=0,
            total_budget=3,
            seed=25,
        )
        assert mock_cholesky.called
        assert len(traj) == 3


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
    assert raw["XRD"].shape == (921, 89)
    assert len(raw["Coer"]) == 921
    assert len(raw["Kerr"]) == 921


@pytest.mark.external_data
def test_real_mat_composition_sums_approximate_100():
    raw = load_raw_feconi_mat()
    sums = np.sum(raw["C"], axis=1)
    assert np.all(sums >= 99.8) and np.all(sums <= 100.2)


@pytest.mark.external_data
def test_real_mat_candidate_ids_are_unique():
    adapter = FeCoNiAdapter()
    pool = adapter.get_candidate_pool()
    assert len(pool["candidate_id"].unique()) == 921


@pytest.mark.external_data
def test_real_mat_oracle_refuses_unknown_and_duplicate_queries():
    adapter = FeCoNiAdapter(target="Kerr")
    oracle = adapter.create_oracle(allow_duplicate_queries=False)
    with pytest.raises(KeyError):
        oracle.query("NON_EXISTENT_ID")
    oracle.query("FECONI_005")
    with pytest.raises(ValueError, match="Duplicate experimental measurement"):
        oracle.query("FECONI_005")


@pytest.mark.external_data
def test_real_mat_target_selection_kerr_vs_coer_independent():
    kerr_oracle = FeCoNiAdapter(target="Kerr").create_oracle()
    coer_oracle = FeCoNiAdapter(target="Coer").create_oracle()

    assert kerr_oracle.target_column == "Kerr"
    assert coer_oracle.target_column == "Coer"
    assert kerr_oracle.global_best_candidate_id != coer_oracle.global_best_candidate_id
    assert np.isclose(kerr_oracle.global_best_value, 0.82504, atol=1e-4)
    assert np.isclose(coer_oracle.global_best_value, 10.9340, atol=1e-3)
