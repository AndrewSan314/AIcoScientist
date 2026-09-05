import numpy as np
import pytest

from src.domains.alab.config import ALAB_DOMAIN_CONFIG
from src.integrations.xrd.autoxrd_adapter import XRDObservableExtractor
from src.science.multimodal.decision import MultimodalDecisionEngine
from src.science.multimodal.hypotheses import build_alab_multimodal_hypotheses
from src.science.multimodal.measurement_models import PredictiveObservableDistribution
from src.science.multimodal.ontology import observable_names_for_modality


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
