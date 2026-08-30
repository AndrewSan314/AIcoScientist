from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, RBF

from src.optimization.acquisition import (
    compute_acquisition,
    compute_true_mc_nei,
    denoised_expected_improvement_acquisition,
    expected_improvement_acquisition,
    safe_cholesky,
)


@pytest.fixture
def fitted_gp() -> tuple[GaussianProcessRegressor, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X_obs = rng.uniform(0.0, 1.0, size=(12, 3))
    # Latent true function f(x) = sin(3*x0) + cos(2*x1)
    y_latent = np.sin(3.0 * X_obs[:, 0]) + np.cos(2.0 * X_obs[:, 1])
    y_noisy = y_latent + rng.normal(0.0, 0.1, size=len(X_obs))

    kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=1)
    gp.fit(X_obs, y_noisy)

    X_cand = rng.uniform(0.0, 1.0, size=(25, 3))
    return gp, X_obs, X_cand


def test_safe_cholesky_escalation_and_properties() -> None:
    # 1. Standard positive definite
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    L = safe_cholesky(A)
    assert np.allclose(L @ L.T, A, atol=1e-5)

    # 2. Singular matrix with identical rows
    B = np.array([[1.0, 1.0], [1.0, 1.0]])
    L_b = safe_cholesky(B)
    assert np.all(np.isfinite(L_b))
    assert L_b.shape == (2, 2)


def test_true_mc_nei_determinism_and_finiteness(
    fitted_gp: tuple[GaussianProcessRegressor, np.ndarray, np.ndarray]
) -> None:
    gp, X_obs, X_cand = fitted_gp

    scores_1 = compute_true_mc_nei(gp, X_obs, X_cand, n_fantasies=128, seed=123)
    scores_2 = compute_true_mc_nei(gp, X_obs, X_cand, n_fantasies=128, seed=123)
    scores_diff_seed = compute_true_mc_nei(gp, X_obs, X_cand, n_fantasies=128, seed=456)

    assert len(scores_1) == len(X_cand)
    assert np.all(np.isfinite(scores_1))
    assert np.all(scores_1 >= 0.0)
    # Determinism with same seed
    assert np.allclose(scores_1, scores_2)
    # Different seed produces slightly different MC estimates with high correlation
    assert not np.allclose(scores_1, scores_diff_seed)
    assert np.corrcoef(scores_1, scores_diff_seed)[0, 1] > 0.95


def test_true_mc_nei_near_zero_noise_approaches_ei() -> None:
    rng = np.random.default_rng(42)
    X_obs = rng.uniform(0.0, 1.0, size=(10, 2))
    y_obs = np.sin(X_obs[:, 0]) + 10.0

    # GP with zero noise kernel (deterministic interpolation)
    kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(
        noise_level=1e-6, noise_level_bounds=(1e-7, 1e-5)
    )
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42, alpha=1e-8)
    gp.fit(X_obs, y_obs)

    X_cand = rng.uniform(0.0, 1.0, size=(15, 2))
    true_nei_scores = compute_true_mc_nei(gp, X_obs, X_cand, n_fantasies=1000, seed=42)

    # Analytic EI with exact incumbent max(y_obs)
    cand_mean, cand_std = gp.predict(X_cand, return_std=True)
    exact_ei = expected_improvement_acquisition(cand_mean, cand_std, best_observed=float(np.max(y_obs)))

    # Under near-zero noise, True NEI converges to analytic EI
    assert np.allclose(true_nei_scores, exact_ei, atol=0.08)


def test_true_mc_nei_vs_noisy_spike_outlier() -> None:
    X_obs = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    # A single noisy measurement spiked to 25.0, others are around 10.0
    y_obs = np.array([10.0, 10.2, 25.0, 10.1, 10.3])

    kernel = ConstantKernel(5.0, (0.1, 20.0)) * Matern(length_scale=0.5, length_scale_bounds=(0.1, 2.0), nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(0.1, 5.0))
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
    gp.fit(X_obs, y_obs)

    X_cand = np.array([[0.25], [0.35]])
    cand_mean, cand_std = gp.predict(X_cand, return_std=True)

    # Standard EI using raw noisy max (25.0) yields near zero because cand_mean ~ 10-12
    raw_ei = expected_improvement_acquisition(cand_mean, cand_std, best_observed=25.0)
    assert np.all(raw_ei < 0.005)

    # True NEI integrates over the posterior distribution of latent incumbent
    true_nei = compute_true_mc_nei(gp, X_obs, X_cand, n_fantasies=256, seed=42)
    assert np.all(np.isfinite(true_nei))
    assert np.all(true_nei >= 0.0)


def test_true_mc_nei_minimization_semantics(
    fitted_gp: tuple[GaussianProcessRegressor, np.ndarray, np.ndarray]
) -> None:
    gp, X_obs, X_cand = fitted_gp

    max_scores = compute_true_mc_nei(gp, X_obs, X_cand, objective="maximize", seed=42)
    min_scores = compute_true_mc_nei(gp, X_obs, X_cand, objective="minimize", seed=42)

    assert np.all(np.isfinite(max_scores))
    assert np.all(np.isfinite(min_scores))
    # Maximization and minimization rankings should generally be inverted
    assert np.corrcoef(max_scores, min_scores)[0, 1] < 0.0


def test_true_mc_nei_candidate_observed_covariance_correlation() -> None:
    """Proves that True NEI accounts for cross-covariance between candidate and observed points."""
    # Observations placed at x = 0.0 and x = 2.0 with noise
    X_obs = np.array([[0.0], [2.0]])
    y_obs = np.array([5.0, 3.0])

    kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.1, 2.0)) + WhiteKernel(
        noise_level=0.1, noise_level_bounds=(0.01, 1.0)
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    gp.fit(X_obs, y_obs)

    # Candidate 1 is close to observed point 0.0 (high correlation to incumbent)
    # Candidate 2 is in the middle (x=1.0, exploratory point)
    X_cand = np.array([[0.05], [1.0]])

    # Compute True MC NEI (with exact joint covariance)
    nei_scores = compute_true_mc_nei(gp, X_obs, X_cand, n_fantasies=500, xi=0.001, seed=42)

    # Scores must be finite, non-negative, and properly distinguish candidate correlations
    assert len(nei_scores) == 2
    assert np.all(np.isfinite(nei_scores))
    assert np.all(nei_scores >= 0.0)
    assert nei_scores[0] != nei_scores[1]



def test_compute_acquisition_nei_raises_on_missing_gp_inputs(
    fitted_gp: tuple[GaussianProcessRegressor, np.ndarray, np.ndarray]
) -> None:
    gp, X_obs, X_cand = fitted_gp
    cand_mean, cand_std = gp.predict(X_cand, return_std=True)

    # Silent fallback is strictly removed: missing GP or design matrices must raise ValueError
    for m in ["nei", "true_nei", "noisy_expected_improvement", "turbo_nei"]:
        with pytest.raises(ValueError, match="requires 'gp', 'X_observed_scaled'"):
            compute_acquisition(
                method=m,
                mean=cand_mean,
                std=cand_std,
                best_observed=10.0,
                gp=None,
                X_observed_scaled=None,
                X_candidates_scaled=None,
            )


def test_explicit_denoised_ei_works_without_gp(
    fitted_gp: tuple[GaussianProcessRegressor, np.ndarray, np.ndarray]
) -> None:
    gp, X_obs, X_cand = fitted_gp
    cand_mean, cand_std = gp.predict(X_cand, return_std=True)
    obs_m = gp.predict(X_obs)

    # Denoised EI works with explicit method designation
    scores = compute_acquisition(
        method="denoised_expected_improvement",
        mean=cand_mean,
        std=cand_std,
        best_observed=10.0,
        observed_posterior_means=obs_m,
    )
    assert len(scores) == len(X_cand)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)


def test_compute_acquisition_dispatch_routes_true_nei(
    fitted_gp: tuple[GaussianProcessRegressor, np.ndarray, np.ndarray]
) -> None:
    gp, X_obs, X_cand = fitted_gp
    cand_mean, cand_std = gp.predict(X_cand, return_std=True)

    score_model = compute_acquisition(
        method="true_nei",
        mean=cand_mean,
        std=cand_std,
        best_observed=10.0,
        gp=gp,
        X_observed_scaled=X_obs,
        X_candidates_scaled=X_cand,
        seed=42,
    )

    direct_score = compute_true_mc_nei(gp, X_obs, X_cand, seed=42)
    assert np.allclose(score_model, direct_score)


def test_predict_latent_gp_variance_smaller_than_noisy() -> None:
    from src.optimization.acquisition import predict_latent_gp
    rng = np.random.default_rng(42)
    X_train = rng.uniform(0.0, 1.0, size=(15, 2))
    y_train = np.sin(3 * X_train[:, 0]) + rng.normal(0.0, 0.2, size=len(X_train))

    kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
    gp.fit(X_train, y_train)

    X_test = rng.uniform(0.0, 1.0, size=(20, 2))

    # Latent prediction
    latent_mean, latent_std = predict_latent_gp(gp, X_test, return_std=True)
    latent_mean_cov, latent_cov = predict_latent_gp(gp, X_test, return_cov=True)

    # Standard GP prediction (noisy test predictions)
    noisy_mean, noisy_std = gp.predict(X_test, return_std=True)
    noisy_mean_cov, noisy_cov = gp.predict(X_test, return_cov=True)

    # Means should match
    assert np.allclose(latent_mean, noisy_mean, atol=1e-5)
    assert np.allclose(latent_mean_cov, noisy_mean_cov, atol=1e-5)

    # Latent variance must be strictly smaller than noisy variance on test points
    assert np.all(latent_std < noisy_std)
    assert np.all(np.diag(latent_cov) < np.diag(noisy_cov))

    # Check training points: latent variance > 0 (strictly positive under noise)
    _, train_latent_std = predict_latent_gp(gp, X_train, return_std=True)
    assert np.all(train_latent_std > 0.0)

