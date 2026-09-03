import numpy as np
import pytest

from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES
from src.domains.electrolyte.hypotheses import (
    DEFAULT_BROAD_CAPACITY_MEAN,
    DEFAULT_BROAD_CAPACITY_VAR,
    ElectrolyteHypothesisProvider,
    GlobalSmoothDescriptorHypothesis,
    LocalChemicalRegimeHypothesis,
    RegularizedAdditiveDescriptorHypothesis,
    SparseAdditiveDescriptorHypothesis,
    evaluate_hypothesis_calibration,
)
from src.science.actions import ScientificAction
from src.science.domain import HypothesisTrainingContext
from src.science.falsification.information_gain import HypothesisInformationGainEstimator
from src.science.hypothesis_models import HypothesisEnsemble, PredictiveDistribution


def test_three_hypotheses_support_capacity():
    """Verifies that all three predictive structural hypotheses support CAPACITY_TEST only."""
    provider = ElectrolyteHypothesisProvider()
    hyps = provider.build_hypotheses()

    assert len(hyps) == 3
    assert "global_smooth_descriptor" in hyps
    assert "regularized_additive_descriptor" in hyps
    assert "local_chemical_regime" in hyps

    for h in hyps.values():
        assert h.supports_action("CAPACITY_TEST") is True
        assert h.supports_action("XRD") is False
        assert h.supports_action("SEM") is False


def test_broad_priors_are_non_discriminating_before_minimum_data():
    """Verifies that when N < 3 observations exist, all hypotheses return identical broad priors."""
    provider = ElectrolyteHypothesisProvider()
    hyps = provider.build_hypotheses()

    dummy_comp = np.zeros(11, dtype=np.float64)
    # Fit on only 1 observation (< 3 minimum)
    ctx = HypothesisTrainingContext(
        candidate_features_by_id={"c1": dummy_comp},
        observations_by_modality={"CAPACITY_TEST": {"c1": 0.50}},
    )

    preds = []
    for h in hyps.values():
        h.fit_context(ctx)
        p = h.predict_observation(candidate_id="c_test", action_type="CAPACITY_TEST", composition=dummy_comp)
        preds.append(p)

    # All means and variances must be bit-for-bit identical before 3 samples
    for p in preds:
        assert float(p.mean[0]) == DEFAULT_BROAD_CAPACITY_MEAN
        assert float(p.variance[0]) == DEFAULT_BROAD_CAPACITY_VAR


def test_predictions_finite():
    """Verifies that predictions produce finite floating-point values."""
    provider = ElectrolyteHypothesisProvider()
    hyps = provider.build_hypotheses()

    X = np.random.default_rng(42).normal(size=(5, 11))
    comp_by_id = {f"c_{i}": X[i] for i in range(5)}
    obs = {f"c_{i}": 0.2 + 0.1 * i for i in range(5)}

    ctx = HypothesisTrainingContext(
        candidate_features_by_id=comp_by_id,
        observations_by_modality={"CAPACITY_TEST": obs},
    )

    for h in hyps.values():
        h.fit_context(ctx)
        pred = h.predict_observation(candidate_id="c_eval", action_type="CAPACITY_TEST", composition=X[0])
        assert np.isfinite(pred.mean).all()
        assert np.isfinite(pred.variance).all()


def test_variances_positive():
    """Verifies that all predictive variances are strictly positive and exceed the numerical floor."""
    provider = ElectrolyteHypothesisProvider()
    hyps = provider.build_hypotheses()

    X = np.random.default_rng(42).normal(size=(5, 11))
    ctx = HypothesisTrainingContext(
        candidate_features_by_id={f"c_{i}": X[i] for i in range(5)},
        observations_by_modality={"CAPACITY_TEST": {f"c_{i}": 0.5 for i in range(5)}},
    )

    for h in hyps.values():
        h.fit_context(ctx)
        pred = h.predict_observation(candidate_id="c_eval", action_type="CAPACITY_TEST", composition=X[0])
        assert (pred.variance > 0.0).all()
        assert (pred.variance >= 0.005).all()


def test_predictions_diverge_after_fitting():
    """Verifies that H1, H2, and H3 produce non-identical predictions after observing data."""
    provider = ElectrolyteHypothesisProvider()
    hyps = provider.build_hypotheses()

    rng = np.random.default_rng(123)
    X = rng.normal(size=(8, 11))
    y = np.clip(0.5 + 0.3 * np.sin(X[:, 0]) + 0.2 * (X[:, 1] > 0), 0.0, 1.0)

    ctx = HypothesisTrainingContext(
        candidate_features_by_id={f"c_{i}": X[i] for i in range(8)},
        observations_by_modality={"CAPACITY_TEST": {f"c_{i}": float(y[i]) for i in range(8)}},
    )

    means = []
    test_comp = rng.normal(size=11)
    for h in hyps.values():
        h.fit_context(ctx)
        pred = h.predict_observation(candidate_id="c_unseen", action_type="CAPACITY_TEST", composition=test_comp)
        means.append(float(pred.mean[0]))

    # Check that predictions are not all identical
    assert np.std(means) > 1e-4, f"Predictions across H1, H2, H3 failed to diverge: {means}"


def test_HIG_is_finite():
    """Verifies that hypothesis information gain is finite and well-conditioned."""
    ensemble = HypothesisEnsemble(hypotheses=ElectrolyteHypothesisProvider().build_hypotheses())
    comp = np.random.default_rng(42).normal(size=11)

    estimator = HypothesisInformationGainEstimator()
    disc = estimator.evaluate_action_discrimination(
        candidate_id="c_1",
        action_type="CAPACITY_TEST",
        composition=comp,
        ensemble=ensemble,
    )

    assert np.isfinite(disc.hypothesis_information_gain)
    assert disc.hypothesis_information_gain >= 0.0


def test_identical_predictions_produce_near_zero_HIG():
    """Verifies that identical hypothesis predictions produce zero information gain."""
    ensemble = HypothesisEnsemble(hypotheses=ElectrolyteHypothesisProvider().build_hypotheses())
    comp = np.zeros(11, dtype=np.float64)

    # Before fitting, all hypotheses return identical broad priors
    estimator = HypothesisInformationGainEstimator()
    disc = estimator.evaluate_action_discrimination(
        candidate_id="c_1",
        action_type="CAPACITY_TEST",
        composition=comp,
        ensemble=ensemble,
    )
    assert disc.hypothesis_information_gain < 1e-6, f"Expected near-zero HIG from identical uninformative priors, got {disc.hypothesis_information_gain}"


def test_posterior_updates_after_capacity_reveal():
    """Verifies that observing capacity updates posterior hypothesis beliefs via Bayesian evidence."""
    provider = ElectrolyteHypothesisProvider()
    ensemble = HypothesisEnsemble(hypotheses=provider.build_hypotheses())

    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, 11))
    comp_map = {f"c_{i}": X[i] for i in range(5)}
    obs_map = {f"c_{i}": 0.2 + 0.1 * i for i in range(4)}

    # Fit on 4 points
    ensemble.fit_all(
        composition_by_id=comp_map,
        observations_by_modality={"CAPACITY_TEST": obs_map},
    )

    prior_beliefs = ensemble.get_beliefs()

    # Reveal 5th point
    c5_comp = X[4]
    preds = ensemble.predict_all(
        candidate_id="c_4",
        action_type="CAPACITY_TEST",
        composition=c5_comp,
    )
    # Record observation (true capacity = 0.60)
    ensemble.record_observation_and_update(
        action_id="act_test_5",
        candidate_id="c_4",
        action_type="CAPACITY_TEST",
        observation=np.array([0.60]),
        pre_predictions=preds,
    )

    post_beliefs = ensemble.get_beliefs()
    diffs = [abs(post_beliefs[k] - prior_beliefs[k]) for k in prior_beliefs]
    assert max(diffs) > 1e-4, f"Posterior beliefs failed to update after observation: {post_beliefs}"


def test_hypothesis_calibration_and_sensitivity():
    """Phase 11: Verifies calibration evaluation and variance floor sensitivity on synthetic historical batches."""
    import pandas as pd
    from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES

    rng = np.random.default_rng(123)
    rows = []
    # Create 3 batches of 6 items each
    for b in range(3):
        for i in range(6):
            feats = rng.normal(size=len(ELECTROLYTE_SOLVENT_FEATURES))
            # True relationship is smooth with additive linear component + noise
            c_norm = float(np.clip(0.3 + 0.1 * feats[0] - 0.05 * feats[1] + rng.normal(scale=0.08), 0.0, 1.0))
            row = {
                "candidate_id": f"ELEC_SYNTH_{b}_{i}",
                "batch": b,
                "C_norm_20": c_norm,
            }
            for idx, col in enumerate(ELECTROLYTE_SOLVENT_FEATURES):
                row[col] = float(feats[idx])
            rows.append(row)

    df_synth = pd.DataFrame(rows)
    calib = evaluate_hypothesis_calibration(df_synth, feature_cols=ELECTROLYTE_SOLVENT_FEATURES)

    assert "hypotheses_calibration" in calib
    assert "variance_floor_sensitivity" in calib
    assert "global_smooth_descriptor" in calib["hypotheses_calibration"]
    assert "regularized_additive_descriptor" in calib["hypotheses_calibration"]
    assert "local_chemical_regime" in calib["hypotheses_calibration"]

    for hid, metrics in calib["hypotheses_calibration"].items():
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mean_log_predictive_density" in metrics
        assert "coverage_50pct_interval" in metrics
        assert "coverage_90pct_interval" in metrics
        assert 0.0 <= metrics["coverage_50pct_interval"] <= 1.0
        assert 0.0 <= metrics["coverage_90pct_interval"] <= 1.0

    sens = calib["variance_floor_sensitivity"]
    assert len(sens["sensitivity_runs"]) == 3
    assert "posterior_winner_stable" in sens

