import numpy as np
import pytest

from src.domains.alab.config import ALAB_DOMAIN_CONFIG
from src.science.multimodal.decision import MultimodalDecisionEngine
from src.science.multimodal.hypotheses import build_alab_multimodal_hypotheses
from src.science.multimodal.ontology import observable_names_for_modality
from src.science.multimodal.retrospective import (
    RetrospectiveCalibratedHypothesisModel,
    assert_no_evaluation_leakage,
    build_retrospective_hypotheses,
)
from src.science.multimodal.schemas import ScientificObservable


def _observation(candidate_id: str, modality: str, value: np.ndarray | float) -> ScientificObservable:
    names = observable_names_for_modality(modality)
    vector = np.atleast_1d(np.asarray(value, dtype=float))
    return ScientificObservable(
        observable_id=f"test:{modality}:{candidate_id}",
        candidate_id=candidate_id,
        modality=modality,
        name=names[0] if len(names) == 1 else f"{modality}.bundle",
        observable_names=names,
        value=float(vector[0]) if len(names) == 1 else vector,
        uncertainty=0.05 if len(names) == 1 else np.full(len(names), 0.05),
        provenance={"reaction_category": "transformed"} if modality == "OUTCOME_TEST" else {},
        observable_type="scalar" if len(names) == 1 else "vector",
    )


def test_retrospective_fit_is_declared_and_disjoint_from_evaluation():
    ids = ["a", "b", "c", "d"]
    features = {cid: np.asarray([index, 0.2 * index, 0.1 * index] + [0.0] * 46) for index, cid in enumerate(ids)}
    observations = {"XRD": {cid: _observation(cid, "XRD", np.full(5, 0.2 + 0.1 * index)) for index, cid in enumerate(ids)}}
    model = RetrospectiveCalibratedHypothesisModel(
        "H1_PHASE_PURITY_LIMITED",
        (0, 1, 2),
        {"XRD": "CALIBRATED_DIRECT_STRUCTURAL", "SEM": "NOT_EVALUATED_INSUFFICIENT_LINKAGE"},
    )
    model.fit(features, observations, training_ids=("a", "b"))
    fit_hash = model.diagnostics()["fit_ids_sha256"]
    model.fit(features, observations, training_ids=("c", "d"))
    assert model.diagnostics()["fit_ids_sha256"] == fit_hash
    assert_no_evaluation_leakage({"H1": model}, ("a", "b"), ("c", "d"))


def test_retrospective_predictions_are_named_and_condition_on_revealed_xrd():
    ids = ["a", "b", "c", "d"]
    features = {cid: np.asarray([index, 0.2 * index, 0.1 * index] + [0.0] * 46) for index, cid in enumerate(ids)}
    observations = {
        "XRD": {cid: _observation(cid, "XRD", np.asarray([0.1 + index * 0.1, 0.2, 0.3, 0.4, 1.0])) for index, cid in enumerate(ids)},
        "REFINEMENT": {cid: _observation(cid, "REFINEMENT", np.asarray([0.1 + index * 0.2, 0.8 - index * 0.1, 0.1, 0.2])) for index, cid in enumerate(ids)},
    }
    model = build_retrospective_hypotheses()["H1_PHASE_PURITY_LIMITED"]
    model.fit(features, observations, training_ids=("a", "b", "c"))
    before = model.predict_observable_distribution("d", "REFINEMENT", candidate_features=features["d"])
    after = model.predict_observable_distribution(
        "d",
        "REFINEMENT",
        {"XRD": {"d": _observation("d", "XRD", np.asarray([0.9, 0.2, 0.3, 0.4, 1.0]))}},
        candidate_features=features["d"],
    )
    assert before.observable_names == observable_names_for_modality("REFINEMENT")
    assert np.all(after.variance > 0)
    assert not np.allclose(before.mean, after.mean)


def test_real_h2_h3_mechanisms_fail_closed_when_linkage_is_missing():
    hypotheses = build_retrospective_hypotheses()
    assert hypotheses["H2_COMPOSITION_HOMOGENEITY_LIMITED"].identifiability_by_modality["EDS"].startswith("NOT_")
    assert hypotheses["H3_MORPHOLOGY_KINETICS_LIMITED"].identifiability_by_modality["SEM"].startswith("NOT_")


def test_leakage_assertion_rejects_overlapping_split_ids():
    model = build_retrospective_hypotheses()["H1_PHASE_PURITY_LIMITED"]
    with pytest.raises(AssertionError, match="overlap"):
        assert_no_evaluation_leakage({"H1": model}, ("a",), ("a",))


def test_evidence_lifecycle_rejects_orphan_and_duplicate_reveals():
    features = {"a": np.zeros(49)}
    engine = MultimodalDecisionEngine(features, ALAB_DOMAIN_CONFIG.modalities, build_alab_multimodal_hypotheses())
    action = next(item for item in engine.enumerate_actions() if item.action_type == "OUTCOME_TEST")
    observation = _observation("a", "OUTCOME_TEST", 0.75)
    with pytest.raises(ValueError, match="preregistered"):
        engine.observe(observation)
    recommendation = engine.register_selected_action(action)
    assert recommendation.preregistration["event"] == "PREREGISTERED_SELECTED_ACTION"
    engine.observe(observation)
    with pytest.raises(ValueError, match="already revealed"):
        engine.observe(observation)
