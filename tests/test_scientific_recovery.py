from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.science.actions import ExperimentActionType
from src.science.falsification.identifiability import (
    compute_monte_carlo_js_divergence,
    moment_matched_gaussian_divergence_proxy,
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
)


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


def test_true_monte_carlo_js_divergence_properties() -> None:
    """P0-4 Tests: Verifies mathematical properties of Monte Carlo JS divergence."""
    # 1. Identical distributions -> 0.0
    p1 = PredictiveDistribution("H1", "C1", ExperimentActionType.PROPERTY, np.array([0.005]), np.array([1e-4]))
    p2 = PredictiveDistribution("H2", "C1", ExperimentActionType.PROPERTY, np.array([0.005]), np.array([1e-4]))
    js_zero = compute_monte_carlo_js_divergence(p1, p2, n_samples=256, seed=42)
    assert np.isclose(js_zero, 0.0, atol=1e-5)

    # 2. Symmetry: JS(p1, p3) == JS(p3, p1)
    p3 = PredictiveDistribution("H3", "C1", ExperimentActionType.PROPERTY, np.array([0.015]), np.array([1e-4]))
    js_13 = compute_monte_carlo_js_divergence(p1, p3, n_samples=512, seed=42)
    js_31 = compute_monte_carlo_js_divergence(p3, p1, n_samples=512, seed=42)
    assert np.isclose(js_13, js_31, atol=0.03)

    # 3. Boundedness: 0 <= JS <= ln(2)
    assert 0.0 <= js_13 <= np.log(2.0) + 1e-5

    # 4. Monotonicity: Increasing separation increases JS divergence
    p_far = PredictiveDistribution("H_far", "C1", ExperimentActionType.PROPERTY, np.array([0.050]), np.array([1e-4]))
    js_far = compute_monte_carlo_js_divergence(p1, p_far, n_samples=512, seed=42)
    assert js_far > js_13


def test_multivariate_embedding_normalized_log_density() -> None:
    """P0-6 Tests: Verifies that 8D Gaussian log-density is commensurable with 1D scalar log-density."""
    pred_8d = PredictiveDistribution(
        "H1", "C1", ExperimentActionType.XRD,
        mean=np.zeros(8),
        variance=np.ones(8) * 0.1,
    )
    obs_near = np.zeros(8)
    obs_mod = np.ones(8) * 0.2

    logp_near = pred_8d.log_pdf(obs_near)
    logp_mod = pred_8d.log_pdf(obs_mod)

    # Dimension-normalized log-density must not produce catastrophic -1000 nats scale
    assert logp_near > -5.0
    assert logp_mod > -10.0
    assert logp_near > logp_mod
