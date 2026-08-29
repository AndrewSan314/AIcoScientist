from __future__ import annotations

import numpy as np
import pytest

from src.optimization.acquisition import (
    compute_acquisition,
    expected_improvement_acquisition,
    greedy_acquisition,
    probability_of_improvement_acquisition,
    ucb_acquisition,
)


def test_greedy_acquisition_semantics() -> None:
    means = np.array([100.0, 150.0, 120.0])
    scores = greedy_acquisition(means)
    assert np.allclose(scores, means)
    assert np.argmax(scores) == 1


def test_ucb_acquisition_semantics() -> None:
    means = np.array([100.0, 100.0, 100.0])
    stds = np.array([0.0, 10.0, 25.0])
    scores = ucb_acquisition(means, stds, beta=2.0)
    assert np.allclose(scores, [100.0, 120.0, 150.0])
    assert np.argmax(scores) == 2


def test_expected_improvement_zero_and_positive_uncertainty() -> None:
    best_observed = 100.0
    means = np.array([90.0, 100.0, 110.0, 90.0])
    stds = np.array([0.0, 0.0, 0.0, 20.0])

    ei = expected_improvement_acquisition(means, stds, best_observed=best_observed, xi=0.0)

    # 90 with std=0 -> 0 improvement
    assert ei[0] == 0.0
    # 100 with std=0 -> 0 improvement
    assert ei[1] == 0.0
    # 110 with std=0 -> 10.0 improvement
    assert np.isclose(ei[2], 10.0)
    # 90 with std=20 -> positive expected improvement due to tail probability
    assert ei[3] > 0.0
    assert np.all(np.isfinite(ei))
    assert np.all(ei >= 0.0)


def test_probability_of_improvement_properties() -> None:
    best_observed = 100.0
    means = np.array([80.0, 100.0, 120.0])
    stds = np.array([10.0, 10.0, 10.0])

    pi = probability_of_improvement_acquisition(means, stds, best_observed=best_observed, xi=0.0)
    assert 0.0 <= pi[0] < pi[1] < pi[2] <= 1.0
    assert np.isclose(pi[1], 0.5, atol=1e-2)  # mean == best_observed => Phi(0) = 0.5


def test_compute_acquisition_dispatch_and_errors() -> None:
    means = np.array([100.0, 120.0])
    stds = np.array([5.0, 10.0])
    best_obs = 110.0

    score_greedy = compute_acquisition("greedy", means, stds, best_observed=best_obs)
    score_ucb = compute_acquisition("gp_ucb", means, stds, best_observed=best_obs, beta=1.5)
    score_ei = compute_acquisition("expected_improvement", means, stds, best_observed=best_obs)
    score_pi = compute_acquisition("probability_of_improvement", means, stds, best_observed=best_obs)

    assert len(score_greedy) == 2
    assert len(score_ucb) == 2
    assert len(score_ei) == 2
    assert len(score_pi) == 2

    with pytest.raises(ValueError, match="Unknown acquisition method"):
        compute_acquisition("invalid_method", means, stds, best_observed=best_obs)
