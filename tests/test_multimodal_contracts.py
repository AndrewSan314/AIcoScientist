import numpy as np
import pandas as pd

from src.domains.alab.config import ALAB_DOMAIN_CONFIG
from src.integrations.microscopy.atomai_adapter import AtomAIEDSExtractor, AtomAISEMExtractor
from src.integrations.xrd.autoxrd_adapter import XRDObservableExtractor
from src.science.actions import normalize_action_type
from src.science.domain import ModalityDefinition
from src.science.multimodal.decision import MultimodalDecisionEngine
from src.science.multimodal.hypotheses import build_alab_multimodal_hypotheses
from src.science.multimodal.schemas import ScientificObservable
from src.domains.electrolyte.surrogate_worlds import evaluate_cross_surrogate_worlds


def test_modality_cost_and_sample_contract_round_trip():
    legacy = ModalityDefinition("legacy", "characterization", 1.0, (), (), None, {"legacy": True})
    assert legacy.metadata["legacy"] is True
    modality = ModalityDefinition(
        "SEM", "image", cost=1.5, duration=2.0,
        required_existing_sample_state=("synthesized",), destructive=True,
        expected_observable_types=("scalar", "structured"), cost_units="USD_NORMALIZED",
    )
    restored = ModalityDefinition.from_dict(modality.to_dict())
    assert restored == modality


def test_hypotheses_predict_cross_modal_distributions_with_positive_uncertainty():
    features = {"c": np.linspace(0.0, 1.0, 49)}
    for hypothesis in build_alab_multimodal_hypotheses().values():
        hypothesis.fit(features, {})
        for modality in ("XRD", "REFINEMENT", "SEM", "EDS", "OUTCOME_TEST"):
            prediction = hypothesis.predict_observable_distribution("c", modality, {}, candidate_features=features["c"])
            assert prediction.mean.size
            assert np.all(prediction.variance > 0)


def test_multimodal_engine_preregisters_and_enforces_refinement_prerequisite():
    features = {"c": np.linspace(0.0, 1.0, 49)}
    engine = MultimodalDecisionEngine(features, ALAB_DOMAIN_CONFIG.modalities, build_alab_multimodal_hypotheses(), seed=3)
    action_types = {normalize_action_type(action.action_type) for action in engine.enumerate_actions()}
    assert "REFINEMENT" not in action_types
    assert "SEM" not in action_types and "EDS" not in action_types
    recommendation = engine.recommend(samples=8)
    assert 0.0 <= recommendation.why["expected_hig_nats"] <= recommendation.why["current_entropy_nats"]
    assert recommendation.preregistration["measurement_revealed"] is False
    hid = next(iter(engine.hypotheses))
    value = np.asarray(recommendation.preregistration["predictive_distributions"][hid]["mean"])
    observable = ScientificObservable(
        observable_id="test-reveal",
        candidate_id=recommendation.action.candidate_id,
        modality=normalize_action_type(recommendation.action.action_type),
        name="test",
        value=value,
        uncertainty=0.1,
        raw_artifact_ref="fixture://test",
    )
    engine.observe(observable)
    assert engine.step == 1


def test_extractors_return_traceable_observables():
    cases = [
        (XRDObservableExtractor(), np.array([0.1, 0.4, 1.0, 0.2])),
        (AtomAISEMExtractor(), np.arange(25, dtype=float).reshape(5, 5)),
        (AtomAIEDSExtractor(), np.array([[0.4, 0.6], [0.5, 0.5]])),
    ]
    for extractor, raw in cases:
        observations = extractor.extract(raw, "candidate", {"raw_artifact_ref": "fixture://raw"})
        assert observations
        assert all(obs.raw_artifact_ref == "fixture://raw" for obs in observations)
        assert all(obs.provenance["raw_artifact_sha256"] for obs in observations)


def test_independent_surrogate_worlds_use_shared_stage1_config():
    values = np.random.default_rng(4).random((12, 3))
    pool = pd.DataFrame(values, columns=["a", "b", "c"])
    pool["candidate_id"] = [str(i) for i in range(len(pool))]
    result = evaluate_cross_surrogate_worlds(pool, values[:6], np.arange(6.0), ["a", "b", "c"], working_set_size=4)
    assert set(result["worlds"]) == {"EXTRATREES", "GP", "NONLINEAR_SYNTHETIC"}
    assert result["stage1_config"]["evidence_mode"] == "HISTORICAL_EVIDENCE"
