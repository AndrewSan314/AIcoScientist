import numpy as np
import pytest

from src.domains.alab.config import ALAB_DOMAIN_CONFIG
from src.integrations.xrd.autoxrd_adapter import XRDObservableExtractor
from src.science.multimodal.decision import MultimodalDecisionEngine
from src.science.multimodal.hypotheses import build_alab_multimodal_hypotheses
from src.science.multimodal.measurement_models import PredictiveObservableDistribution
from src.science.multimodal.ontology import observable_names_for_modality
from src.science.multimodal.retrospective import build_group_holdout_protocols, canonical_elemental_system
from scripts.run_alab_multimodal_benchmark import _clean_world_distribution_consistency, _predictive_metrics, _raw_hig_bounds, hig_trajectory_sensitivity
from src.science.multimodal.schemas import ScientificObservable


def test_predictive_likelihood_is_schema_safe_and_uncertainty_aware():
    names = observable_names_for_modality("XRD")
    prediction = PredictiveObservableDistribution("H", "c", "XRD", np.ones(5), np.ones(5), names)
    with pytest.raises(ValueError, match="schema mismatch"):
        prediction.log_pdf(np.ones(5), observed_names=tuple(reversed(names)))
    assert prediction.log_pdf(np.ones(5), measurement_uncertainty=1.0) < prediction.log_pdf(np.ones(5))


def test_hig_is_invariant_to_candidate_iteration_order():
    features = {"a": np.linspace(0.0, 1.0, 49), "b": np.linspace(1.0, 0.0, 49)}
    hypotheses_a = build_alab_multimodal_hypotheses()
    hypotheses_b = build_alab_multimodal_hypotheses()
    first_a = MultimodalDecisionEngine(features, ALAB_DOMAIN_CONFIG.modalities, hypotheses_a, seed=11)
    first_b = MultimodalDecisionEngine(dict(reversed(list(features.items()))), ALAB_DOMAIN_CONFIG.modalities, hypotheses_b, seed=11)
    actions_a = {action.action_id: action for action in first_a.enumerate_actions()}
    actions_b = {action.action_id: action for action in first_b.enumerate_actions()}
    assert set(actions_a) == set(actions_b)
    for action_id in actions_a:
        assert first_a.expected_hypothesis_information_gain(actions_a[action_id], samples=12) == pytest.approx(
            first_b.expected_hypothesis_information_gain(actions_b[action_id], samples=12)
        )


def test_xrd_fallback_uses_honest_descriptor_names():
    observations = XRDObservableExtractor().extract(np.array([0.1, 0.4, 1.0, 0.2]))
    names = {observation.name for observation in observations}
    assert names == set(observable_names_for_modality("XRD"))
    assert not any("crystallinity" in name.lower() or "fwhm" in name.lower() for name in names)


def test_clean_world_keeps_unclipped_gaussian_draws_and_consistent_variance():
    result = _clean_world_distribution_consistency()
    assert result["status"] == "PASS"
    assert result["clean_observation_bounds_applied"] is False
    assert result["boundary_point_mass_detected"] is False
    assert result["uncertainty_convention"] == "PREDICTIVE_VARIANCE_IS_TOTAL_OBSERVATION_VARIANCE"


def test_chemistry_family_group_audit_is_explicit_and_order_invariant():
    metadata = {
        "a": {"target_compound": "Fe2O3", "precursor_formulas": ("FeO", "O2")},
        "b": {"target_compound": "FeO", "precursor_formulas": ("O2", "FeO")},
        "c": {"target_compound": "CoO", "precursor_formulas": ("Co", "O2")},
    }
    protocols = build_group_holdout_protocols(metadata)
    assert canonical_elemental_system("Fe2O3") == canonical_elemental_system("FeO")
    assert protocols["TARGET_ELEMENTAL_SYSTEM_GROUP_HOLDOUT"]["group_size_audit"]["number_of_groups"] == 2
    assert protocols["TARGET_ELEMENTAL_SYSTEM_GROUP_HOLDOUT"]["group_function_version"] == "2026-09-08"


def test_raw_hig_gate_checks_raw_value_not_clipped_flag():
    lower, upper = _raw_hig_bounds({
        "raw_hig_mc_nats": 999.0,
        "current_hypothesis_entropy_nats": 1.0,
        "hig_upper_bound_epsilon_nats": 0.1,
        "raw_hig_lower_bound_ok": True,
        "raw_hig_upper_bound_ok": True,
    })
    assert lower is True
    assert upper is False


def test_real_predictive_metrics_use_total_variance_once():
    names = observable_names_for_modality("XRD")
    prediction = PredictiveObservableDistribution("H", "c", "XRD", np.full(5, 0.5), np.full(5, 0.04), names)

    class Predictor:
        def predict_observable_distribution(self, candidate_id, modality, *, candidate_features=None):
            return prediction

    observed = ScientificObservable(
        "uncertainty-convention", "c", "XRD", names[0], np.full(5, 0.5), np.full(5, 10.0),
        provenance={"test": "total_predictive_variance"}, observable_type="vector", observable_names=names,
    )
    result = _predictive_metrics(Predictor(), "XRD", {"c": observed}, {"c": np.zeros(49)}, model_type="test", feature_family="test", training_n=1)
    assert result["NLL"] == pytest.approx(-prediction.log_pdf(observed.value, observed_names=names))
    assert result["coverage50"] == 1.0
    assert result["measurement_uncertainty_applied"] is False


def test_hig_sensitivity_is_paired_and_retains_raw_comparison(monkeypatch):
    import scripts.run_alab_multimodal_benchmark as benchmark

    monkeypatch.setattr(benchmark, "SEEDS", (7,))
    result = hig_trajectory_sensitivity(steps=1, candidate_count=2)
    assert result["status"] == "PASS"
    assert result["trajectory_count"] == 12
    assert result["design"]["low_samples"] == 12
    assert result["design"]["high_samples"] == 32
