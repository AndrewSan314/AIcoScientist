import numpy as np
import pytest
import json
from pathlib import Path

from src.domains.alab.config import ALAB_DOMAIN_CONFIG
from src.science.multimodal.decision import MultimodalDecisionEngine
from src.science.multimodal.hypotheses import build_alab_multimodal_hypotheses
from src.science.multimodal.ontology import observable_names_for_modality
from src.science.multimodal.retrospective import (
    build_group_holdout_protocols,
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


def test_sample_holdout_has_no_id_overlap():
    protocols = build_group_holdout_protocols({
        "s1": {"target_compound": "NaCl", "precursor_formulas": ("Na", "Cl")},
        "s2": {"target_compound": "NaCl", "precursor_formulas": ("Na", "Cl")},
        "s3": {"target_compound": "KCl", "precursor_formulas": ("K", "Cl")},
        "s4": {"target_compound": "LiF", "precursor_formulas": ("Li", "F")},
    })
    sample = protocols["SAMPLE_ID_INTERPOLATION_HOLDOUT"]
    assert set(sample["calibration_ids"]).isdisjoint(sample["evaluation_ids"])
    assert sample["group_overlap"] == []


def test_reaction_signature_holdout_has_no_group_overlap():
    protocols = build_group_holdout_protocols({
        "s1": {"target_compound": "NaCl", "precursor_formulas": ("Na", "Cl")},
        "s2": {"target_compound": "NaCl", "precursor_formulas": ("Na", "Cl")},
        "s3": {"target_compound": "KCl", "precursor_formulas": ("K", "Cl")},
        "s4": {"target_compound": "LiF", "precursor_formulas": ("Li", "F")},
    })
    reaction = protocols["REACTION_SIGNATURE_GROUP_HOLDOUT"]
    assert reaction["group_overlap"] == []
    assert reaction["precursor_signature_overlap"] == []


def test_target_holdout_has_no_target_overlap():
    protocols = build_group_holdout_protocols({
        "s1": {"target_compound": "NaCl", "precursor_formulas": ("Na", "Cl")},
        "s2": {"target_compound": "NaCl", "precursor_formulas": ("Na", "Cl")},
        "s3": {"target_compound": "KCl", "precursor_formulas": ("K", "Cl")},
        "s4": {"target_compound": "LiF", "precursor_formulas": ("Li", "F")},
    })
    target = protocols["TARGET_COMPOUND_GROUP_HOLDOUT"]
    assert target["group_overlap"] == []
    assert target["target_overlap"] == []


def test_group_split_preprocessing_uses_train_only():
    ids = ["a", "b", "c", "d"]
    features = {cid: np.asarray([index, 0.2 * index, 0.1 * index] + [0.0] * 46) for index, cid in enumerate(ids)}
    model = build_retrospective_hypotheses()["H1_PHASE_PURITY_LIMITED"]
    model.fit(features, {"XRD": {cid: _observation(cid, "XRD", np.full(5, 0.2)) for cid in ids}}, training_ids=("a", "b"))
    assert model.diagnostics()["preprocessing_fit_ids_sha256"] == model.diagnostics()["fit_ids_sha256"]
    assert set(model.fitted_ids) == {"a", "b"}


def test_nondiagnostic_shared_nuisance_does_not_change_relative_beliefs():
    from scripts.run_alab_multimodal_benchmark import _nondiagnostic_evidence_diagnostics

    result = _nondiagnostic_evidence_diagnostics()
    assert result["status"] == "PASS"
    assert result["relative_beliefs_unchanged"] is True


def test_shared_nuisance_preserves_h2_h3_odds_for_xrd():
    models = build_retrospective_hypotheses()
    features = {"a": np.asarray([0.1, 0.2, 0.3] + [0.0] * 46)}
    observations = {"XRD": {"a": _observation("a", "XRD", np.full(5, 0.2))}}
    for model in models.values():
        model.fit(features, observations, training_ids=("a",))
    h2 = models["H2_COMPOSITION_HOMOGENEITY_LIMITED"].predict_observable_distribution("a", "XRD", candidate_features=features["a"])
    h3 = models["H3_MORPHOLOGY_KINETICS_LIMITED"].predict_observable_distribution("a", "XRD", candidate_features=features["a"])
    assert h2.metadata["modality_role"] == "SHARED_NUISANCE"
    assert np.allclose(h2.mean, h3.mean)
    assert np.allclose(h2.variance, h3.variance)


def test_hig_lower_and_upper_bounds_are_recorded():
    engine = MultimodalDecisionEngine({"a": np.zeros(49)}, ALAB_DOMAIN_CONFIG.modalities, build_alab_multimodal_hypotheses(), seed=3)
    recommendation = engine.recommend(samples=8)
    record = recommendation.preregistration
    assert record["hig_lower_bound_ok"] is True
    assert record["hig_upper_bound_ok"] is True


def test_release_readiness_pending_if_ci_uninspected(monkeypatch):
    from scripts.run_alab_multimodal_benchmark import build_validation

    monkeypatch.delenv("AICOSCIENTIST_EXTERNAL_CI_GATE", raising=False)
    result = build_validation({}, {}, {}, {"policy_formulas": {}}, {}, {})
    assert result["external_ci_status"] == "NOT_INSPECTED"
    assert result["release_readiness"] == "PENDING_EXTERNAL_CI"


def test_clean_world_draws_from_true_hypothesis_distribution():
    from scripts.run_alab_multimodal_benchmark import _controlled_modalities, _make_reveal

    features = {"c": np.zeros(49)}
    hypotheses = build_alab_multimodal_hypotheses()
    engine = MultimodalDecisionEngine(features, _controlled_modalities("CLEAN_WORLD_H1_PHASE_PURITY"), hypotheses, seed=7, policy_name="PURE_HIG")
    action = next(item for item in engine.enumerate_actions() if item.action_type == "XRD")
    prediction = hypotheses["H1_PHASE_PURITY_LIMITED"].predict_observable_distribution("c", "XRD", candidate_features=features["c"])
    observation = _make_reveal(engine, action, hypotheses["H1_PHASE_PURITY_LIMITED"], "CLEAN_WORLD_H1_PHASE_PURITY", 7, features)
    assert observation.provenance["world_type"] == "CLEAN_CORRECTLY_SPECIFIED"
    assert np.allclose(observation.uncertainty, np.sqrt(prediction.variance))


def test_stress_world_is_labeled_misspecified():
    from scripts.run_alab_multimodal_benchmark import _controlled_modalities, _make_reveal

    features = {"c": np.zeros(49)}
    hypotheses = build_alab_multimodal_hypotheses()
    engine = MultimodalDecisionEngine(features, _controlled_modalities("STRESS_WORLD_H1_PHASE_PURITY"), hypotheses, seed=7, policy_name="PURE_HIG")
    action = next(item for item in engine.enumerate_actions() if item.action_type == "XRD")
    observation = _make_reveal(engine, action, hypotheses["H1_PHASE_PURITY_LIMITED"], "STRESS_WORLD_H1_PHASE_PURITY", 7, features)
    assert observation.provenance["world_type"] == "STRESS_INTENTIONALLY_MISSPECIFIED"
    assert observation.provenance["nominal_true_hypothesis"] == "H1_PHASE_PURITY_LIMITED"


def test_xrd_metrics_are_per_observable():
    metrics = json.loads((Path("outputs/alab/multimodal") / "per_observable_calibration.json").read_text(encoding="utf-8"))
    assert set(metrics["XRD"]) == {
        "XRD.normalized_intensity_std_proxy", "XRD.dominant_peak_index_fraction",
        "XRD.global_halfmax_span_proxy", "XRD.spectral_entropy", "XRD.peak_count_proxy",
    }
    assert all("MAE" in item and "RMSE" in item and "NLL" in item for item in metrics["XRD"].values())


def test_refinement_metrics_are_per_observable():
    metrics = json.loads((Path("outputs/alab/multimodal") / "per_observable_calibration.json").read_text(encoding="utf-8"))
    assert set(metrics["REFINEMENT"]) == {
        "REFINEMENT.target_phase_fraction", "REFINEMENT.precursor_phase_fraction",
        "REFINEMENT.other_identified_phase_fraction", "REFINEMENT.rwp_scaled",
    }


def test_full_policy_matrix_has_3_worlds_5_seeds_6_policies():
    artifact = json.loads((Path("outputs/alab/multimodal") / "full_policy_matrix.json").read_text(encoding="utf-8"))
    assert artifact["trajectory_count"] == 180
    assert len(artifact["policies"]) == 6 and len(artifact["seeds"]) == 5
    assert len(artifact["summary_by_world_policy"]) == 6


def test_hybrid_score_independent_recomputation():
    artifact = json.loads((Path("outputs/alab/multimodal") / "full_policy_matrix.json").read_text(encoding="utf-8"))
    validation = artifact["policy_validation"]
    assert validation["hybrid_score_recomputation_gate"] == "PASS"
    assert all(item["matches"] for item in validation["hybrid_score_recomputation"])


def test_hybrid_cost_counterfactual():
    artifact = json.loads((Path("outputs/alab/multimodal") / "hybrid_counterfactual_cost.json").read_text(encoding="utf-8"))
    assert artifact["status"] == "PASS"
    assert artifact["effects"]


def test_discovery_hig_conflict_is_computed():
    artifact = json.loads((Path("outputs/alab/multimodal") / "policy_conflict_diagnostics.json").read_text(encoding="utf-8"))
    assert artifact["policy_validation"]["discovery_hig_conflict_gate"] == "PASS"
    assert artifact["discovery_vs_hig"]


def test_real_replay_does_not_use_hidden_evaluation_outcomes():
    artifact = json.loads((Path("outputs/alab/multimodal") / "retrospective_policy_comparison.json").read_text(encoding="utf-8"))
    matrix = artifact["real_retrospective_policy_matrix"]
    assert matrix["status"] == "METHODOLOGY_VALID"
    assert matrix["hidden_evaluation_outcomes_used_by_policy"] is False


def test_real_hybrid_has_nonzero_discovery():
    artifact = json.loads((Path("outputs/alab/multimodal") / "retrospective_policy_comparison.json").read_text(encoding="utf-8"))
    values = [row["discovery_utility"] for row in artifact["real_retrospective_policy_matrix"]["summary"]["HYBRID"]]
    assert any(value > 0 for value in values)
