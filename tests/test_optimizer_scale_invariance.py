from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from src.legacy.native_optimizer.acquisition import (
    compute_acquisition,
    compute_true_mc_nei,
    expected_improvement_acquisition,
    greedy_acquisition,
    predict_latent_gp,
    ucb_acquisition,
)
from src.legacy.native_optimizer.closed_loop import (
    ClosedLoopOptimizer,
    ExperimentProposal,
    ExperimentResult,
)
from src.optimization.search_space import (
    ContinuousVariable,
    SearchSpace,
)
from src.legacy.native_optimizer.trust_region import TuRBOTrustRegion


@pytest.fixture
def synthetic_scale_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates a small deterministic 2D dataset with known target values."""
    rng = np.random.default_rng(42)
    X_obs = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.9, 0.1],
    ])
    # Target in a small scale (e.g. 0.001 to 0.015, like Au-Ir-Rh k0)
    y_obs = np.array([0.002, 0.005, 0.014, 0.008, 0.003])

    X_cand = np.array([
        [0.2, 0.3],
        [0.4, 0.5],
        [0.6, 0.7],
        [0.8, 0.9],
        [0.5, 0.2],
    ])
    return X_obs, y_obs, X_cand, rng


def test_greedy_acquisition_scale_invariance() -> None:
    """Greedy acquisition (posterior mean) ranking must be strictly invariant under positive affine scaling."""
    means = np.array([0.002, 0.008, 0.014, 0.005])
    a, b = 1000.0, 7.0
    means_scaled = a * means + b

    scores_orig = greedy_acquisition(means, objective="maximize")
    scores_scaled = greedy_acquisition(means_scaled, objective="maximize")

    assert np.allclose(scores_scaled, a * scores_orig + b)
    assert np.array_equal(np.argsort(scores_orig), np.argsort(scores_scaled))


def test_ucb_acquisition_scale_invariance() -> None:
    """GP-UCB acquisition ranking must be strictly invariant under positive affine scaling."""
    means = np.array([0.002, 0.008, 0.014, 0.005])
    stds = np.array([0.001, 0.003, 0.002, 0.004])
    beta = 2.0
    a, b = 1000.0, 7.0

    means_scaled = a * means + b
    stds_scaled = a * stds

    scores_orig = ucb_acquisition(means, stds, beta=beta, objective="maximize")
    scores_scaled = ucb_acquisition(means_scaled, stds_scaled, beta=beta, objective="maximize")

    assert np.allclose(scores_scaled, a * scores_orig + b)
    assert np.array_equal(np.argsort(scores_orig), np.argsort(scores_scaled))


def test_expected_improvement_scale_invariance_canonical_xi_zero() -> None:
    """Analytic EI with canonical xi=0 must scale linearly by 'a' and preserve ranking under y' = a*y + b."""
    means = np.array([0.005, 0.008, 0.014, 0.010, 0.002])
    stds = np.array([0.001, 0.002, 0.001, 0.003, 0.002])
    best_obs = 0.012

    a, b = 1000.0, 7.0
    means_scaled = a * means + b
    stds_scaled = a * stds
    best_obs_scaled = a * best_obs + b

    # Under canonical scale-invariant default xi=0.0
    ei_orig = expected_improvement_acquisition(means, stds, best_obs, xi=0.0)
    ei_scaled = expected_improvement_acquisition(means_scaled, stds_scaled, best_obs_scaled, xi=0.0)

    # EI'(x) = a * EI(x)
    assert np.allclose(ei_scaled, a * ei_orig, rtol=1e-5)
    assert np.array_equal(np.argsort(ei_orig), np.argsort(ei_scaled))


def test_expected_improvement_scale_dependence_under_nonzero_xi() -> None:
    """Demonstrates that non-zero absolute xi (e.g. xi=0.01) creates artificial target-scale dependence."""
    means = np.array([0.005, 0.008, 0.012, 0.010])
    stds = np.array([0.001, 0.002, 0.001, 0.003])
    best_obs = 0.010

    a, b = 1000.0, 7.0
    means_scaled = a * means + b
    stds_scaled = a * stds
    best_obs_scaled = a * best_obs + b

    # Absolute threshold xi=0.01 is huge on range ~0.01 but negligible on range ~17.0
    ei_orig = expected_improvement_acquisition(means, stds, best_obs, xi=0.01)
    ei_scaled = expected_improvement_acquisition(means_scaled, stds_scaled, best_obs_scaled, xi=0.01)

    # With non-zero absolute xi, the rankings do NOT match
    assert not np.array_equal(np.argsort(ei_orig), np.argsort(ei_scaled))


def test_true_mc_nei_scale_invariance_canonical_xi_zero(
    synthetic_scale_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """True MC NEI with canonical xi=0 must scale linearly by 'a' and preserve ranking under y' = a*y + b."""
    X_obs, y_obs, X_cand, _ = synthetic_scale_dataset
    a, b = 1000.0, 7.0
    y_obs_scaled = a * y_obs + b

    # Fit GP on original data
    kernel1 = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(noise_level=0.01)
    gp_orig = GaussianProcessRegressor(kernel=kernel1, normalize_y=True, random_state=42)
    gp_orig.fit(X_obs, y_obs)

    # Fit GP on scaled data
    kernel2 = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(noise_level=0.01)
    gp_scaled = GaussianProcessRegressor(kernel=kernel2, normalize_y=True, random_state=42)
    gp_scaled.fit(X_obs, y_obs_scaled)

    # Compute True MC NEI with xi=0.0 and identical fantasy seed
    scores_orig = compute_true_mc_nei(gp_orig, X_obs, X_cand, n_fantasies=256, xi=0.0, seed=123)
    scores_scaled = compute_true_mc_nei(gp_scaled, X_obs, X_cand, n_fantasies=256, xi=0.0, seed=123)

    # Scores must scale by 'a' (within numerical GP optimization tolerance)
    assert np.allclose(scores_scaled, a * scores_orig, rtol=1e-2)
    assert np.array_equal(np.argsort(scores_orig), np.argsort(scores_scaled))


def test_turbo_posterior_success_decision_scale_invariance() -> None:
    """TuRBO posterior success probability and update decision must be invariant under positive affine scaling with success_delta=0."""
    space = SearchSpace(variables=[ContinuousVariable("x1", lower=0, upper=1), ContinuousVariable("x2", lower=0, upper=1)])
    a, b = 1000.0, 7.0

    cand_mean, inc_mean = 0.011, 0.010
    cand_var, inc_var, cov = 4e-6, 4e-6, 3.5e-6
    obs_val = 0.012

    cand_mean_sc = a * cand_mean + b
    inc_mean_sc = a * inc_mean + b
    cand_var_sc = (a ** 2) * cand_var
    inc_var_sc = (a ** 2) * inc_var
    cov_sc = (a ** 2) * cov
    obs_val_sc = a * obs_val + b

    tr_orig = TuRBOTrustRegion(search_space=space, success_delta=0.0, success_probability_threshold=0.6)
    tr_orig.initialize({"x1": 0.5, "x2": 0.5}, initial_best_value=inc_mean)
    u_orig = tr_orig.update(
        {"x1": 0.6, "x2": 0.5},
        observed_value=obs_val,
        posterior_candidate_mean=cand_mean,
        posterior_incumbent_mean=inc_mean,
        posterior_candidate_variance=cand_var,
        posterior_incumbent_variance=inc_var,
        posterior_candidate_incumbent_covariance=cov,
    )

    tr_scaled = TuRBOTrustRegion(search_space=space, success_delta=0.0, success_probability_threshold=0.6)
    tr_scaled.initialize({"x1": 0.5, "x2": 0.5}, initial_best_value=inc_mean_sc)
    u_scaled = tr_scaled.update(
        {"x1": 0.6, "x2": 0.5},
        observed_value=obs_val_sc,
        posterior_candidate_mean=cand_mean_sc,
        posterior_incumbent_mean=inc_mean_sc,
        posterior_candidate_variance=cand_var_sc,
        posterior_incumbent_variance=inc_var_sc,
        posterior_candidate_incumbent_covariance=cov_sc,
    )

    assert np.isclose(u_orig["success_probability"], u_scaled["success_probability"])
    assert u_orig["success_counter"] == u_scaled["success_counter"]
    assert u_orig["failure_counter"] == u_scaled["failure_counter"]
    assert u_orig["expanded"] == u_scaled["expanded"]
    assert u_orig["contracted"] == u_scaled["contracted"]
    assert np.isclose(tr_orig.state.length, tr_scaled.state.length)


def test_turbo_trajectory_scale_invariance_multi_step() -> None:
    """Multi-step TuRBO expansion/contraction trajectory must be identical under positive affine target scaling."""
    space = SearchSpace(variables=[ContinuousVariable("x1", lower=0, upper=1), ContinuousVariable("x2", lower=0, upper=1)])
    a, b = 1000.0, 7.0

    steps_data = [
        # (cand_mean, inc_mean, cand_var, inc_var, cov, obs_val)
        (0.011, 0.010, 4e-6, 4e-6, 3.5e-6, 0.012),  # success
        (0.013, 0.012, 4e-6, 4e-6, 3.5e-6, 0.014),  # success
        (0.015, 0.014, 4e-6, 4e-6, 3.5e-6, 0.016),  # success -> triggers expansion (tolerance=3)
        (0.012, 0.016, 4e-6, 4e-6, 1e-6, 0.011),    # failure
    ]

    tr_orig = TuRBOTrustRegion(search_space=space, success_delta=0.0, success_tolerance=3, failure_tolerance=3)
    tr_orig.initialize({"x1": 0.5, "x2": 0.5}, initial_best_value=0.010)

    tr_scaled = TuRBOTrustRegion(search_space=space, success_delta=0.0, success_tolerance=3, failure_tolerance=3)
    tr_scaled.initialize({"x1": 0.5, "x2": 0.5}, initial_best_value=a * 0.010 + b)

    for cm, im, cv, iv, cov, ov in steps_data:
        u_orig = tr_orig.update(
            {"x1": 0.6, "x2": 0.6},
            observed_value=ov,
            posterior_candidate_mean=cm,
            posterior_incumbent_mean=im,
            posterior_candidate_variance=cv,
            posterior_incumbent_variance=iv,
            posterior_candidate_incumbent_covariance=cov,
        )
        u_scaled = tr_scaled.update(
            {"x1": 0.6, "x2": 0.6},
            observed_value=a * ov + b,
            posterior_candidate_mean=a * cm + b,
            posterior_incumbent_mean=a * im + b,
            posterior_candidate_variance=(a**2) * cv,
            posterior_incumbent_variance=(a**2) * iv,
            posterior_candidate_incumbent_covariance=(a**2) * cov,
        )
        assert np.isclose(u_orig["success_probability"], u_scaled["success_probability"])
        assert u_orig["expanded"] == u_scaled["expanded"]
        assert u_orig["contracted"] == u_scaled["contracted"]
        assert np.isclose(tr_orig.state.length, tr_scaled.state.length)


def test_end_to_end_closed_loop_candidate_selection_invariance() -> None:
    """ClosedLoopOptimizer on a finite pool must select the identical candidate trajectory under y' = a*y + b."""
    space = SearchSpace(variables=[ContinuousVariable("x1", lower=0, upper=10), ContinuousVariable("x2", lower=0, upper=10)])
    a, b = 1000.0, 7.0

    # Fixed pool of 10 distinct candidate points with ground truth values
    pool_data = [
        {"candidate_id": f"CAND_{i}", "x1": float(i * 1.0), "x2": float(10.0 - i * 1.0), "y_orig": float(0.001 * (i + 1) ** 1.2)}
        for i in range(10)
    ]
    for row in pool_data:
        row["y_scaled"] = a * row["y_orig"] + b

    pool_df = pd.DataFrame(pool_data)

    for strategy in ["greedy", "gp_ucb", "expected_improvement", "turbo_nei"]:
        # Run on original target with 2 initial points
        init_obs_orig = [
            {"candidate_id": "CAND_0", "x1": 0.0, "x2": 10.0, "target": pool_df.loc[0, "y_orig"]},
            {"candidate_id": "CAND_1", "x1": 1.0, "x2": 9.0, "target": pool_df.loc[1, "y_orig"]},
        ]
        opt_orig = ClosedLoopOptimizer(
            search_space=space,
            feature_cols=["x1", "x2"],
            target_col="target",
            strategy=strategy,
            objective="maximize",
            random_state=42,
        )
        state_orig = opt_orig.initialize(init_obs_orig)

        # Run on scaled target with 2 initial points
        init_obs_scaled = [
            {"candidate_id": "CAND_0", "x1": 0.0, "x2": 10.0, "target": pool_df.loc[0, "y_scaled"]},
            {"candidate_id": "CAND_1", "x1": 1.0, "x2": 9.0, "target": pool_df.loc[1, "y_scaled"]},
        ]
        opt_scaled = ClosedLoopOptimizer(
            search_space=space,
            feature_cols=["x1", "x2"],
            target_col="target",
            strategy=strategy,
            objective="maximize",
            random_state=42,
        )
        state_scaled = opt_scaled.initialize(init_obs_scaled)

        # Propose next experiment
        prop_orig = opt_orig.propose(state_orig)
        prop_scaled = opt_scaled.propose(state_scaled)

        # Both must propose the exact same candidate ID and design variables
        assert prop_orig.candidate_id == prop_scaled.candidate_id
        assert np.isclose(prop_orig.design_variables["x1"], prop_scaled.design_variables["x1"])
        assert np.isclose(prop_orig.design_variables["x2"], prop_scaled.design_variables["x2"])


def test_explicit_nonzero_xi_and_success_delta_remain_supported() -> None:
    """Explicit domain-specific xi > 0 and success_delta > 0 remain fully supported as user options."""
    means = np.array([0.005, 0.008])
    stds = np.array([0.001, 0.002])
    best_obs = 0.010

    # User passes explicit xi=0.05
    ei_custom = expected_improvement_acquisition(means, stds, best_obs, xi=0.05)
    assert np.all(np.isfinite(ei_custom))
    assert np.all(ei_custom >= 0.0)

    # User passes explicit success_delta=2.0
    space = SearchSpace(variables=[ContinuousVariable("x", lower=0, upper=1)])
    tr = TuRBOTrustRegion(space, success_delta=2.0)
    assert tr.success_delta == 2.0
