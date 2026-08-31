from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.science.actions import ExperimentActionType
from src.science.falsification.identifiability import (
    _compute_raw_monte_carlo_js,
    compute_monte_carlo_js_divergence,
    moment_matched_gaussian_divergence_proxy,
)
from src.science.falsification.information_gain import (
    HypothesisInformationGainEstimator,
)
from src.science.hypothesis_models import (
    CompositionSufficientHypothesis,
    HypothesisEnsemble,
    LocalStructuralRegimeHypothesis,
    PredictiveDistribution,
    StructureInformedHypothesis,
    _build_candidate_maps,
)


class FixedGaussianHypothesis:
    """Lightweight test hypothesis generating strictly controlled Gaussian predictions."""

    def __init__(self, hypothesis_id: str, mean: np.ndarray, variance: np.ndarray) -> None:
        self._id = hypothesis_id
        self._mean = np.asarray(mean, dtype=np.float64)
        self._variance = np.asarray(variance, dtype=np.float64)

    @property
    def hypothesis_id(self) -> str:
        return self._id

    @property
    def title(self) -> str:
        return f"Fixed Hypothesis {self._id}"

    @property
    def statement(self) -> str:
        return f"Controlled test claim for {self._id}"

    @property
    def assumptions(self) -> list[str]:
        return ["Deterministic test distribution"]

    def supports_action(self, action_type: ExperimentActionType) -> bool:
        return True

    def fit(self, *args: Any, **kwargs: Any) -> None:
        pass

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ExperimentActionType,
        composition: np.ndarray,
        observed_xrd_embedding: np.ndarray | None = None,
    ) -> PredictiveDistribution:
        return PredictiveDistribution(
            hypothesis_id=self._id,
            candidate_id=candidate_id,
            action_type=action_type,
            mean=self._mean,
            variance=self._variance,
            metadata={"model_type": "controlled_fixed_gaussian"},
        )

    def log_predictive_density(
        self,
        observation: np.ndarray | float,
        prediction: PredictiveDistribution,
    ) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(self, *args: Any, **kwargs: Any) -> str:
        return "Fixed test criterion"


# ---------------------------------------------------------------------------
# 1. Probability Density & Distribution Tests
# ---------------------------------------------------------------------------
def test_scalar_gaussian_log_pdf_matches_analytical() -> None:
    """Verifies that 1D scalar Gaussian log_pdf matches exact analytical formula."""
    mu = 0.005
    var = 4.0e-6
    pred = PredictiveDistribution("H1", "C1", ExperimentActionType.PROPERTY, np.array([mu]), np.array([var]))

    obs_val = 0.007
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

    assert np.isclose(log_2d, 2.0 * log_1d, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Canonical Controlled Hypothesis Information Gain (HIG) Tests
# ---------------------------------------------------------------------------
def test_hig_estimator_canonical_identical_distributions() -> None:
    """Canonical Case A: Identical predictive distributions must yield HIG ~ 0 and H[P] unchanged."""
    estimator = HypothesisInformationGainEstimator(n_samples_benchmark=256)

    h1 = FixedGaussianHypothesis("H1", np.array([0.005]), np.array([1e-4]))
    h2 = FixedGaussianHypothesis("H2", np.array([0.005]), np.array([1e-4]))
    ensemble = HypothesisEnsemble({"H1": h1, "H2": h2})

    eval_res = estimator.evaluate_action_discrimination(
        candidate_id="C1",
        action_type=ExperimentActionType.PROPERTY,
        composition=np.array([33.3, 33.3, 33.4]),
        ensemble=ensemble,
        seed=42,
    )

    assert abs(eval_res.hypothesis_information_gain) < 0.05
    assert np.isclose(eval_res.expected_posterior_entropy, eval_res.current_entropy, atol=0.05)


def test_hig_estimator_canonical_separated_distributions() -> None:
    """Canonical Case B: Well-separated predictive distributions must yield high HIG and reduced entropy."""
    estimator = HypothesisInformationGainEstimator(n_samples_benchmark=512)

    # 10 sigma separation
    h1 = FixedGaussianHypothesis("H1", np.array([0.0]), np.array([0.01]))
    h2 = FixedGaussianHypothesis("H2", np.array([1.0]), np.array([0.01]))
    ensemble = HypothesisEnsemble({"H1": h1, "H2": h2})

    eval_res = estimator.evaluate_action_discrimination(
        candidate_id="C1",
        action_type=ExperimentActionType.PROPERTY,
        composition=np.array([33.3, 33.3, 33.4]),
        ensemble=ensemble,
        seed=42,
    )

    assert eval_res.hypothesis_information_gain > 0.5
    assert eval_res.expected_posterior_entropy < eval_res.current_entropy - 0.5


def test_hig_estimator_canonical_monotonic_separation() -> None:
    """Canonical Case C: Larger distribution separation must produce higher HIG."""
    estimator = HypothesisInformationGainEstimator(n_samples_benchmark=512)

    # Small separation (1 sigma)
    h1_small = FixedGaussianHypothesis("H1", np.array([0.0]), np.array([1.0]))
    h2_small = FixedGaussianHypothesis("H2", np.array([1.0]), np.array([1.0]))
    ens_small = HypothesisEnsemble({"H1": h1_small, "H2": h2_small})

    eval_small = estimator.evaluate_action_discrimination(
        candidate_id="C1",
        action_type=ExperimentActionType.PROPERTY,
        composition=np.array([33.3, 33.3, 33.4]),
        ensemble=ens_small,
        seed=42,
    )

    # Large separation (6 sigma)
    h1_large = FixedGaussianHypothesis("H1", np.array([0.0]), np.array([1.0]))
    h2_large = FixedGaussianHypothesis("H2", np.array([6.0]), np.array([1.0]))
    ens_large = HypothesisEnsemble({"H1": h1_large, "H2": h2_large})

    eval_large = estimator.evaluate_action_discrimination(
        candidate_id="C1",
        action_type=ExperimentActionType.PROPERTY,
        composition=np.array([33.3, 33.3, 33.4]),
        ensemble=ens_large,
        seed=42,
    )

    assert eval_large.hypothesis_information_gain > eval_small.hypothesis_information_gain


def test_hig_estimator_multidimensional_gaussian_xrd_case() -> None:
    """Canonical Multidimensional Case: True multivariate density integration in HIG estimator."""
    estimator = HypothesisInformationGainEstimator(n_samples_benchmark=512)

    # 8-dimensional separated Gaussians
    mu1 = np.zeros(8)
    mu2 = np.ones(8) * 2.0
    var = np.ones(8) * 0.5

    h1 = FixedGaussianHypothesis("H1", mu1, var)
    h2 = FixedGaussianHypothesis("H2", mu2, var)
    ensemble = HypothesisEnsemble({"H1": h1, "H2": h2})

    eval_res = estimator.evaluate_action_discrimination(
        candidate_id="C1",
        action_type=ExperimentActionType.XRD,
        composition=np.array([33.3, 33.3, 33.4]),
        ensemble=ensemble,
        seed=42,
    )

    assert eval_res.hypothesis_information_gain > 0.4
    assert eval_res.expected_posterior_entropy < eval_res.current_entropy


# ---------------------------------------------------------------------------
# 3. Monte Carlo Jensen-Shannon Validation Tests
# ---------------------------------------------------------------------------
def test_raw_and_bounded_monte_carlo_js_divergence() -> None:
    """Verifies unclipped raw and bounded Monte Carlo JS estimates for controlled Gaussian pairs."""
    p1 = PredictiveDistribution("H1", "C1", ExperimentActionType.PROPERTY, np.array([0.0]), np.array([1.0]))
    p2 = PredictiveDistribution("H2", "C1", ExperimentActionType.PROPERTY, np.array([3.0]), np.array([1.0]))

    # Raw MC estimate must naturally stay within theoretical bounds [0, ln(2)] +/- MC tolerance
    raw_js = _compute_raw_monte_carlo_js(p1, p2, n_samples=512, seed=42)
    assert -0.05 <= raw_js <= np.log(2.0) + 0.05

    bounded_js = compute_monte_carlo_js_divergence(p1, p2, n_samples=512, seed=42)
    assert 0.0 <= bounded_js <= np.log(2.0)

    # Identical distributions -> 0.0
    raw_zero = _compute_raw_monte_carlo_js(p1, p1, n_samples=256, seed=42)
    assert np.isclose(raw_zero, 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# 4. Defensive Candidate Map Precedence Tests
# ---------------------------------------------------------------------------
def test_build_candidate_maps_precedence_and_validation() -> None:
    """P2 Tests: Covers explicit maps, legacy arrays, conflicts, duplicate IDs, and length mismatches."""
    # 1. Explicit map only
    comp_map = {"C1": np.array([50.0, 30.0, 20.0])}
    prop_map = {"C1": 0.008}
    xrd_map = {"C1": np.zeros(8)}
    c, p, x = _build_candidate_maps(
        composition_by_id=comp_map,
        property_by_id=prop_map,
        xrd_embedding_by_id=xrd_map,
    )
    assert c["C1"][0] == 50.0
    assert p["C1"] == 0.008

    # 2. Legacy arrays only
    c_leg, p_leg, x_leg = _build_candidate_maps(
        compositions=np.array([[50.0, 30.0, 20.0]]),
        candidate_ids=["C1"],
        property_targets=np.array([0.008]),
        property_candidate_ids=["C1"],
    )
    assert "C1" in c_leg
    assert p_leg["C1"] == 0.008

    # 3. Explicit + identical legacy -> keeps explicit without error
    c_both, p_both, _ = _build_candidate_maps(
        composition_by_id=comp_map,
        property_by_id=prop_map,
        compositions=np.array([[50.0, 30.0, 20.0]]),
        candidate_ids=["C1"],
        property_targets=np.array([0.008]),
        property_candidate_ids=["C1"],
    )
    assert c_both["C1"][0] == 50.0

    # 4. Explicit + conflicting legacy -> raises ValueError
    with pytest.raises(ValueError, match="Conflicting explicit and legacy"):
        _build_candidate_maps(
            composition_by_id=comp_map,
            compositions=np.array([[99.0, 1.0, 0.0]]),
            candidate_ids=["C1"],
        )

    with pytest.raises(ValueError, match="Conflicting explicit and legacy"):
        _build_candidate_maps(
            property_by_id=prop_map,
            property_targets=np.array([0.999]),
            property_candidate_ids=["C1"],
        )

    # 5. Duplicate candidate IDs in legacy -> raises ValueError
    with pytest.raises(ValueError, match="Duplicate candidate IDs"):
        _build_candidate_maps(
            compositions=np.zeros((3, 3)),
            candidate_ids=["C1", "C1", "C2"],
        )

    # 6. Length mismatch in legacy -> raises ValueError
    with pytest.raises(ValueError, match="Length mismatch"):
        _build_candidate_maps(
            compositions=np.zeros((3, 3)),
            candidate_ids=["C1", "C2"],
        )


# ---------------------------------------------------------------------------
# 5. Partial-Fit State & Readiness Flag Tests (H1, H2, H3)
# ---------------------------------------------------------------------------
def test_h1_partial_fit_states() -> None:
    """Verifies that H1 handles zero observations, XRD only, and Property only without NotFittedError."""
    comp = np.array([40.0, 30.0, 30.0])

    # 1. Zero observations
    h1_zero = CompositionSufficientHypothesis()
    assert h1_zero._has_property_model is False
    assert h1_zero._has_structure_model is False
    pred_prop = h1_zero.predict_observation("C1", ExperimentActionType.PROPERTY, comp)
    pred_xrd = h1_zero.predict_observation("C1", ExperimentActionType.XRD, comp)
    assert np.isfinite(pred_prop.mean[0])
    assert np.isfinite(pred_xrd.mean[0])
    assert pred_prop.metadata["model_type"] == "unfitted_property_prior"

    # 2. XRD only
    h1_xrd = CompositionSufficientHypothesis()
    h1_xrd.fit(
        composition_by_id={"C1": comp, "C2": comp},
        xrd_embedding_by_id={"C1": np.zeros(8), "C2": np.ones(8)},
    )
    assert h1_xrd._has_property_model is False
    assert h1_xrd._has_structure_model is True
    pred_p = h1_xrd.predict_observation("C1", ExperimentActionType.PROPERTY, comp)
    pred_x = h1_xrd.predict_observation("C1", ExperimentActionType.XRD, comp)
    assert pred_p.metadata["model_type"] == "unfitted_property_prior"
    assert pred_x.metadata["model_type"] == "baseline_structure_gp"

    # 3. Property only
    h1_prop = CompositionSufficientHypothesis()
    h1_prop.fit(
        composition_by_id={"C1": comp, "C2": comp},
        property_by_id={"C1": 0.005, "C2": 0.008},
    )
    assert h1_prop._has_property_model is True
    assert h1_prop._has_structure_model is False
    pred_p2 = h1_prop.predict_observation("C1", ExperimentActionType.PROPERTY, comp)
    pred_x2 = h1_prop.predict_observation("C1", ExperimentActionType.XRD, comp)
    assert pred_p2.metadata["model_type"] == "composition_gp"
    assert pred_x2.metadata["model_type"] == "unfitted_structure_prior"


def test_h2_partial_fit_states() -> None:
    """Verifies that H2 handles all 5 critical partial-fit states without NotFittedError."""
    comp1 = np.array([40.0, 30.0, 30.0])
    comp2 = np.array([20.0, 50.0, 30.0])

    # 1. Zero observations
    h2_zero = StructureInformedHypothesis()
    p_zero = h2_zero.predict_observation("C1", ExperimentActionType.PROPERTY, comp1)
    x_zero = h2_zero.predict_observation("C1", ExperimentActionType.XRD, comp1)
    assert p_zero.metadata["model_type"] == "unfitted_prior"
    assert x_zero.metadata["model_type"] == "unfitted_structure_prior"

    # 2. XRD only
    h2_xrd = StructureInformedHypothesis()
    h2_xrd.fit(
        composition_by_id={"C1": comp1, "C2": comp2},
        xrd_embedding_by_id={"C1": np.zeros(8), "C2": np.ones(8)},
    )
    assert h2_xrd._has_structure_model is True
    assert h2_xrd._has_comp_property_model is False
    assert h2_xrd._has_joint_data is False
    p_xrd = h2_xrd.predict_observation("C1", ExperimentActionType.PROPERTY, comp1)
    assert p_xrd.metadata["model_type"] == "unfitted_property_prior"

    # 3. Property only
    h2_prop = StructureInformedHypothesis()
    h2_prop.fit(
        composition_by_id={"C1": comp1, "C2": comp2},
        property_by_id={"C1": 0.005, "C2": 0.008},
    )
    assert h2_prop._has_structure_model is False
    assert h2_prop._has_comp_property_model is True
    assert h2_prop._has_joint_data is False
    p_prop = h2_prop.predict_observation("C1", ExperimentActionType.PROPERTY, comp1)
    assert p_prop.metadata["model_type"] == "comp_gp_with_structure_uncertainty"

    # 4. Property + XRD with disjoint candidate IDs
    h2_disjoint = StructureInformedHypothesis()
    h2_disjoint.fit(
        composition_by_id={"C1": comp1, "C2": comp2},
        property_by_id={"C1": 0.005},
        xrd_embedding_by_id={"C2": np.ones(8)},
    )
    assert h2_disjoint._has_structure_model is True
    assert h2_disjoint._has_comp_property_model is True
    assert h2_disjoint._has_joint_data is False
    p_disjoint = h2_disjoint.predict_observation("C1", ExperimentActionType.PROPERTY, comp1)
    assert p_disjoint.metadata["model_type"] == "comp_gp_with_structure_uncertainty"

    # 5. Overlapping Property + XRD sufficient for joint model
    h2_joint = StructureInformedHypothesis()
    h2_joint.fit(
        composition_by_id={"C1": comp1, "C2": comp2},
        property_by_id={"C1": 0.005, "C2": 0.008},
        xrd_embedding_by_id={"C1": np.zeros(8), "C2": np.ones(8)},
    )
    assert h2_joint._has_joint_data is True
    p_joint_obs = h2_joint.predict_observation("C1", ExperimentActionType.PROPERTY, comp1, observed_xrd_embedding=np.zeros(8))
    p_joint_unobs = h2_joint.predict_observation("C1", ExperimentActionType.PROPERTY, comp1)
    assert p_joint_obs.metadata["model_type"] == "joint_observed_structure"
    assert p_joint_unobs.metadata["model_type"] == "joint_predicted_structure_inflated"


def test_h3_partial_fit_states() -> None:
    """Verifies that H3 handles zero observations, XRD only, and Property only without NotFittedError."""
    comp1 = np.array([40.0, 30.0, 30.0])
    comp2 = np.array([20.0, 50.0, 30.0])
    comp3 = np.array([10.0, 10.0, 80.0])

    # 1. Zero observations
    h3_zero = LocalStructuralRegimeHypothesis()
    p_zero = h3_zero.predict_observation("C1", ExperimentActionType.PROPERTY, comp1)
    x_zero = h3_zero.predict_observation("C1", ExperimentActionType.XRD, comp1)
    assert p_zero.metadata["model_type"] == "unfitted"
    assert x_zero.metadata["model_type"] == "unfitted_structure_prior"

    # 2. XRD only
    h3_xrd = LocalStructuralRegimeHypothesis()
    h3_xrd.fit(
        composition_by_id={"C1": comp1, "C2": comp2, "C3": comp3},
        xrd_embedding_by_id={"C1": np.zeros(8), "C2": np.ones(8), "C3": np.ones(8) * 2.0},
    )
    assert h3_xrd._has_structure_model is True
    assert h3_xrd._has_global_property_model is False
    p_xrd = h3_xrd.predict_observation("C1", ExperimentActionType.PROPERTY, comp1)
    x_xrd = h3_xrd.predict_observation("C1", ExperimentActionType.XRD, comp1)
    assert p_xrd.metadata["model_type"] == "unfitted_property_prior"
    assert x_xrd.metadata["model_type"] == "local_structure_matern_gp"

    # 3. Property only
    h3_prop = LocalStructuralRegimeHypothesis()
    h3_prop.fit(
        composition_by_id={"C1": comp1, "C2": comp2, "C3": comp3},
        property_by_id={"C1": 0.005, "C2": 0.008, "C3": 0.012},
    )
    assert h3_prop._has_global_property_model is True
    assert h3_prop._has_structure_model is False
    p_prop = h3_prop.predict_observation("C1", ExperimentActionType.PROPERTY, comp1)
    x_prop = h3_prop.predict_observation("C1", ExperimentActionType.XRD, comp1)
    assert "gp" in p_prop.metadata["model_type"]
    assert x_prop.metadata["model_type"] == "unfitted_structure_prior"
