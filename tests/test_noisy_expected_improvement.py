from __future__ import annotations

import numpy as np
import pytest

from src.optimization.acquisition import (
    compute_acquisition,
    expected_improvement_acquisition,
    mc_noisy_expected_improvement,
    noisy_expected_improvement_acquisition,
)


def test_nei_numerical_stability_and_finite_outputs() -> None:
    mean = np.array([100.0, 110.0, 90.0, 120.0])
    std = np.array([0.0, 1e-12, 5.0, 10.0])
    obs_means = np.array([95.0, 105.0, 100.0])

    scores = noisy_expected_improvement_acquisition(
        mean=mean,
        std=std,
        observed_posterior_means=obs_means,
        xi=0.01,
        objective="maximize",
    )

    assert len(scores) == 4
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)

    # For zero variance below incumbent (105.0), improvement is 0.0
    assert scores[0] == 0.0
    # For zero variance above incumbent (110.0 > 105.0), improvement is 110 - 105 - 0.01 = 4.99
    assert np.isclose(scores[1], 4.99, atol=1e-2)


def test_nei_vs_standard_ei_incumbent_handling() -> None:
    mean = np.array([108.0, 102.0])
    std = np.array([3.0, 4.0])

    # Noisy observed data has a spurious spike at 150.0, but denoised posterior mean is 105.0
    noisy_spike_obs = 150.0
    denoised_obs_means = np.array([100.0, 102.0, 105.0])

    # Standard EI with raw noisy spike produces 0 or tiny score
    ei_score = expected_improvement_acquisition(mean, std, best_observed=noisy_spike_obs)
    assert np.all(ei_score < 1e-4)

    # NEI with denoised incumbent (105.0) recognizes candidate at 108.0 as high-value improvement
    nei_score = noisy_expected_improvement_acquisition(
        mean=mean,
        std=std,
        observed_posterior_means=denoised_obs_means,
    )
    assert nei_score[0] > 1.0


def test_mc_nei_consistency() -> None:
    mean = np.array([110.0, 95.0])
    std = np.array([5.0, 5.0])
    obs_means = np.array([100.0, 105.0])

    analytic_nei = noisy_expected_improvement_acquisition(mean, std, observed_posterior_means=obs_means)
    mc_nei = mc_noisy_expected_improvement(mean, std, observed_means=obs_means, n_mc_samples=5000, seed=42)

    # MC NEI and analytic NEI should be close
    assert np.allclose(analytic_nei, mc_nei, atol=0.2)


def test_compute_acquisition_dispatch_nei() -> None:
    mean = np.array([10.0, 20.0])
    std = np.array([1.0, 2.0])

    score1 = compute_acquisition("nei", mean, std, best_observed=15.0)
    score2 = compute_acquisition("noisy_expected_improvement", mean, std, best_observed=15.0)
    score3 = compute_acquisition("turbo_nei", mean, std, best_observed=15.0)

    assert np.allclose(score1, score2)
    assert np.allclose(score1, score3)
