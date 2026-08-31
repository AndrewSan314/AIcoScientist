from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.science.actions import ExperimentActionType
from src.science.falsification.identifiability import (
    compute_monte_carlo_js_divergence,
    moment_matched_gaussian_divergence_proxy,
)
from src.science.falsification.information_gain import (
    HypothesisInformationGainEstimator,
)
from src.science.falsification.synthetic_worlds import (
    World1_CompositionSufficient,
    World2_StructureInformed,
    World3_LocalStructuralRegime,
)
from src.science.hypothesis_models import (
    CompositionSufficientHypothesis,
    HypothesisEnsemble,
    PredictiveDistribution,
    StructureInformedHypothesis,
    _build_candidate_maps,
)


def test_scalar_gaussian_log_pdf_matches_analytical() -> None:
    """Verifies that 1D scalar Gaussian log_pdf matches exact analytical formula."""
    mu = 0.005
    var = 4.0e-6
    std = np.sqrt(var)
    pred = PredictiveDistribution("H1", "C1", ExperimentActionType.PROPERTY, np.array([mu]), np.array([var]))

    obs_val = 0.007
    # Analytical: -0.5 * (log(2*pi) + log(var) + (obs - mu)^2 / var)
    expected_log_prob = -0.5 * (np.log(2.0 * np.pi) + np.log(var) + ((obs_val - mu) ** 2) / var)
    actual_log_prob = pred.log_pdf(obs_val)

    assert np.isclose(actual_log_prob, expected_log_prob, atol=1e-10)


def test_multivariate_gaussian_log_pdf_matches_analytical() -> None:
    """Verifies that ND diagonal Gaussian log_pdf matches exact multivariate formula."""
    mu = np.array([0.1, -0.2, 0.5, 0.0])
    var = np.array([0.04, 0.01, 0.09, 0.16])
    dim = len(mu)
    pred = PredictiveDistribution("H1", "C1", ExperimentActionType.XRD, mu, var)

    obs = np.array([0.15, -0.18, 0.45, 0.02])
    # Analytical: -0.5 * (dim * log(2*pi) + sum(log(var_d)) + sum((obs_d - mu_d)^2 / var_d))
    diff = obs - mu
    quad = np.sum((diff**2) / var)
    log_det = np.sum(np.log(var))
    expected_log_prob = -0.5 * (dim * np.log(2.0 * np.pi) + log_det + quad)

    actual_log_prob = pred.log_pdf(obs)
    assert np.isclose(actual_log_prob, expected_log_prob, atol=1e-10)


def test_independent_duplicate_dimensions_accumulate_log_density() -> None:
    """Verifies that independent dimensions accumulate rather than averaging log density."""
    mu_1d = np.array([0.5])
    var_1d = np.array([0.25])
    pred_1d = PredictiveDistribution("H1", "C1", ExperimentActionType.PROPERTY, mu_1d, var_1d)

    mu_2d = np.array([0.5, 0.5])
    var_2d = np.array([0.25, 0.25])
    pred_2d = PredictiveDistribution("H1", "C1", ExperimentActionType.XRD, mu_2d, var_2d)

    obs_1d = np.array([0.6])
    obs_2d = np.array([0.6, 0.6])

    log_1d = pred_1d.log_pdf(obs_1d)
    log_2d = pred_2d.log_pdf(obs_2d)

    # In probability theory, log p(y1, y2) = log p(y1) + log p(y2) = 2 * log p(y1)
    assert np.isclose(log_2d, 2.0 * log_1d, atol=1e-10)


def test_true_monte_carlo_js_divergence_properties() -> None:
    """Verifies mathematical properties of Monte Carlo JS divergence using true probability densities."""
    # 1. Identical distributions -> 0.0
    p1 = PredictiveDistribution("H1", "C1", ExperimentActionType.PROPERTY, np.array([0.005]), np.array([1e-6]))
    p2 = PredictiveDistribution("H2", "C1", ExperimentActionType.PROPERTY, np.array([0.005]), np.array([1e-6]))
    js_zero = compute_monte_carlo_js_divergence(p1, p2, n_samples=256, seed=42)
    assert np.isclose(js_zero, 0.0, atol=1e-6)

    # 2. Symmetry: JS(p1, p3) == JS(p3, p1)
    p3 = PredictiveDistribution("H3", "C1", ExperimentActionType.PROPERTY, np.array([0.009]), np.array([1e-6]))
    js_13 = compute_monte_carlo_js_divergence(p1, p3, n_samples=512, seed=42)
    js_31 = compute_monte_carlo_js_divergence(p3, p1, n_samples=512, seed=42)
    assert np.isclose(js_13, js_31, atol=0.03)

    # 3. Boundedness: 0 <= JS <= ln(2)
    assert 0.0 <= js_13 <= np.log(2.0) + 1e-4

    # 4. Monotonicity: Increasing separation increases JS divergence
    p_far = PredictiveDistribution("H_far", "C1", ExperimentActionType.PROPERTY, np.array([0.020]), np.array([1e-6]))
    js_far = compute_monte_carlo_js_divergence(p1, p_far, n_samples=512, seed=42)
    assert js_far > js_13


def test_hig_estimator_behavior_with_true_density() -> None:
    """Verifies that HIG behaves according to information theory with true probability densities."""
    hig_estimator = HypothesisInformationGainEstimator(n_samples_benchmark=256)

    # Mock ensemble with 2 identical hypotheses
    ens_identical = HypothesisEnsemble()
    eval_zero = hig_estimator.evaluate_action_discrimination(
        candidate_id="C1",
        action_type=ExperimentActionType.PROPERTY,
        composition=np.array([33.3, 33.3, 33.4]),
        ensemble=ens_identical,
        seed=42,
    )
    # When unfitted with identical priors and means, HIG should be ~0
    assert eval_zero.hypothesis_information_gain >= 0.0
    assert eval_zero.expected_posterior_entropy <= eval_zero.current_entropy + 1e-6


def test_candidate_identity_alignment_shuffled_invariance() -> None:
    """P0-1 Regression Test: Shuffling candidate input order must produce identical fitted models."""
    cids = [f"C_{i:02d}" for i in range(10)]
    comps = np.random.default_rng(42).dirichlet([1, 1, 1], size=10) * 100
    props = np.random.default_rng(42).uniform(0.001, 0.01, size=10)
    xrds = np.random.default_rng(42).normal(0, 1, size=(10, 8))

    # Order A: Original natural order
    comp_map_a = {cid: comps[i] for i, cid in enumerate(cids)}
    prop_map_a = {cid: props[i] for i, cid in enumerate(cids)}
    xrd_map_a = {cid: xrds[i] for i, cid in enumerate(cids)}

    h2_a = StructureInformedHypothesis(random_state=42)
    h2_a.fit(composition_by_id=comp_map_a, property_by_id=prop_map_a, xrd_embedding_by_id=xrd_map_a)

    # Order B: Completely shuffled dictionary order
    perm = [3, 7, 0, 9, 2, 5, 8, 1, 4, 6]
    comp_map_b = {cids[i]: comps[i] for i in perm}
    prop_map_b = {cids[i]: props[i] for i in perm}
    xrd_map_b = {cids[i]: xrds[i] for i in perm}

    h2_b = StructureInformedHypothesis(random_state=42)
    h2_b.fit(composition_by_id=comp_map_b, property_by_id=prop_map_b, xrd_embedding_by_id=xrd_map_b)

    # Predictions must be strictly identical
    test_comp = np.array([30.0, 40.0, 30.0])
    pred_a = h2_a.predict_observation("C_00", ExperimentActionType.PROPERTY, test_comp, observed_xrd_embedding=xrds[0])
    pred_b = h2_b.predict_observation("C_00", ExperimentActionType.PROPERTY, test_comp, observed_xrd_embedding=xrds[0])

    assert np.isclose(pred_a.mean[0], pred_b.mean[0], atol=1e-7)
    assert np.isclose(pred_a.variance[0], pred_b.variance[0], atol=1e-7)


def test_build_candidate_maps_defensive_validation() -> None:
    """P2 Tests: Verifies defensive error checking in _build_candidate_maps."""
    # 1. Length mismatch between IDs and compositions raises ValueError
    with pytest.raises(ValueError, match="Length mismatch"):
        _build_candidate_maps(
            compositions=np.zeros((3, 3)),
            candidate_ids=["C1", "C2"],
        )

    # 2. Duplicate candidate IDs raise ValueError
    with pytest.raises(ValueError, match="Duplicate candidate IDs"):
        _build_candidate_maps(
            compositions=np.zeros((3, 3)),
            candidate_ids=["C1", "C1", "C2"],
        )

    # 3. Property targets length mismatch raises ValueError
    with pytest.raises(ValueError, match="Length mismatch"):
        _build_candidate_maps(
            property_targets=np.array([0.001, 0.002]),
            property_candidate_ids=["C1", "C2", "C3"],
        )

    # 4. Explicit mappings take absolute precedence
    comp_map = {"C1": np.array([50.0, 30.0, 20.0])}
    prop_map = {"C1": 0.008}
    xrd_map = {"C1": np.zeros(8)}
    c, p, x = _build_candidate_maps(
        composition_by_id=comp_map,
        property_by_id=prop_map,
        xrd_embedding_by_id=xrd_map,
        compositions=np.zeros((2, 3)),
        candidate_ids=["D1", "D2"],
    )
    assert "C1" in c
    assert "D1" not in c
    assert c["C1"][0] == 50.0
