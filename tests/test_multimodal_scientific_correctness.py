import json
from pathlib import Path

import numpy as np

from scripts.run_alab_multimodal_benchmark import _clean_world_distribution_consistency
from src.science.multimodal.retrospective import reaction_signature


OUT = Path("outputs/alab/multimodal")


def _artifact(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_clean_world_predictive_variance_matches_empirical():
    result = _clean_world_distribution_consistency()
    assert result["generator_variance_match"] is True
    assert np.allclose(result["empirical_generator_variance"], result["generator_variance"], rtol=0.25, atol=0.01)


def test_clean_world_likelihood_uses_same_variance():
    result = _clean_world_distribution_consistency()
    assert np.isclose(result["log_pdf_actual"], result["log_pdf_expected"])


def test_clean_world_hig_uses_same_variance():
    result = _clean_world_distribution_consistency()
    assert result["hig_variance"] == result["generator_variance"]


def test_clean_world_no_double_count():
    result = _clean_world_distribution_consistency()
    assert result["inference_variance"] == result["generator_variance"]
    assert result["measurement_uncertainty"] == [0.0] * len(result["measurement_uncertainty"])


def test_clean_world_distribution_consistency_gate():
    assert _artifact("multimodal_validation.json")["gates"]["clean_world_distribution_consistency_gate"] == "PASS"


def test_threshold_metrics_use_censoring_not_999():
    artifact = _artifact("full_policy_matrix.json")
    for policies in artifact["summary_by_world_policy"].values():
        for summary in policies.values():
            for metric in summary["threshold_metrics"].values():
                assert metric["N_crossed"] <= metric["N_total"]
                assert metric["crossing_rate"] == metric["N_crossed"] / metric["N_total"]
                assert all(value != 999 for value in metric.values() if isinstance(value, (int, float)))


def test_threshold_metrics_within_horizon():
    artifact = _artifact("full_policy_matrix.json")
    for policies in artifact["summary_by_world_policy"].values():
        for summary in policies.values():
            horizon = summary["trajectory_horizon"]
            for metric in summary["threshold_metrics"].values():
                for key in ("mean_steps_conditional_on_crossing", "median_steps_conditional_on_crossing", "min_steps", "max_steps"):
                    assert metric[key] is None or 0 <= metric[key] <= horizon


def test_split_metadata_protocols_and_hashes():
    protocols = _artifact("split_protocols.json")
    for name, split in protocols.items():
        assert split["split_protocol"] == name
        assert split["group_key"]
        assert split["calibration_n"] > 0 and split["evaluation_n"] > 0
        assert split["calibration_ids_sha256"] and split["evaluation_ids_sha256"]
        assert split["preprocessing_fit_scope"] == "calibration_ids_only"


def test_group_split_preprocessing_is_calibration_only():
    for name in ("sample_holdout_metrics.json", "reaction_signature_holdout_metrics.json", "target_holdout_metrics.json"):
        split = _artifact(name)["split"]
        assert split["disjoint"] is True
        assert split["sample_overlap"] == []
        assert split["group_overlap"] == []
        assert split["preprocessing_fit_scope"] == "calibration_ids_only"


def test_shared_nuisance_model_is_calibration_only():
    artifact = _artifact("shared_nuisance_model_metrics.json")
    assert artifact["status"] == "PASS"
    assert artifact["model"]["fit_scope"] == "calibration_ids_only"
    assert artifact["model"]["feature_family"] == "all_allowed_non_mechanistic_context_features"
    assert artifact["model"]["feature_indices"] == list(range(49))
    assert artifact["model"]["training_N"] > 0 and artifact["model"]["evaluation_N"] > 0


def test_shared_nuisance_is_identical_for_h2_h3_xrd_and_refinement():
    artifact = _artifact("shared_nuisance_model_metrics.json")
    assert artifact["h2_h3_symmetry"]["status"] == "PASS"
    assert all(row["mean_max_abs_diff"] == 0.0 and row["variance_max_abs_diff"] == 0.0 for row in artifact["h2_h3_symmetry"]["rows"])


def test_shared_nuisance_h2_h3_symmetry_artifact():
    artifact = _artifact("shared_nuisance_model_metrics.json")
    assert artifact["h2_h3_odds_contribution"] == "BF_H2/H3 = 1 for identical shared XRD/refinement nuisance predictions"
    for modality in ("XRD", "REFINEMENT"):
        models = artifact["per_modality"][modality]
        assert set(models) == {"pooled_mean_baseline", "shared_predictive_nuisance", "H1_scientific_structural_model"}
        assert all(models[name]["evaluation_N"] > 0 for name in models)


def test_raw_hig_is_persisted_and_not_clipped_for_audit():
    record = next(json.loads(line) for line in (OUT / "evidence_ledger.jsonl").open(encoding="utf-8") if '"event": "ACTION_SCORE_RECORD"' in line)
    diagnostics = record["hig_diagnostics"]
    assert diagnostics["raw_hig_mc_nats"] == record["raw_hig_mc_nats"]
    assert diagnostics["clipped_hig_nats"] <= diagnostics["current_entropy_nats"]
    assert diagnostics["mc_samples"] > 0


def test_hig_bound_epsilon_uses_mc_se():
    record = next(json.loads(line) for line in (OUT / "evidence_ledger.jsonl").open(encoding="utf-8") if '"event": "ACTION_SCORE_RECORD"' in line)
    diagnostics = record["hig_diagnostics"]
    expected = 3.0 * diagnostics["hig_mc_standard_error"] + diagnostics["hig_numeric_epsilon_nats"]
    assert np.isclose(diagnostics["hig_bound_epsilon_nats"], expected)


def test_hig_rank_stability_artifact():
    artifact = _artifact("hig_monte_carlo_diagnostics.json")
    assert artifact["status"] == "PASS"
    assert artifact["samples"] == [16, 32, 64, 128]
    assert artifact["top1_agreement_64_vs_128"] is True
    assert artifact["rank_correlation_64_vs_128"] >= 0.9


def test_reaction_signature_is_order_invariant():
    assert reaction_signature("NaCl", ("Na", "Cl")) == reaction_signature("NaCl", ("Cl", "Na"))


def test_reaction_signature_respects_stoichiometric_identity():
    assert reaction_signature("NaCl", ("Na", "Cl")) != reaction_signature("Na2Cl", ("Na", "Cl"))


def test_reaction_signature_equivalent_formula_spelling():
    assert reaction_signature("NaCl", ("Na1", "Cl1")) == reaction_signature("ClNa", ("Cl", "Na"))


def test_posterior_concentration_warning_is_explicit():
    artifact = _artifact("posterior_concentration_diagnostics.json")
    assert artifact["status"] == "PASS"
    assert artifact["n_observations"] > 0
    assert artifact["n_unique_target_groups"] > 0
    assert artifact["n_unique_reaction_signatures"] > 0
    assert "correlated" in artifact["independence_warning"]


def test_refinement_target_phase_fraction_is_flagged_overdispersed():
    artifact = _artifact("per_observable_calibration.json")
    annotation = artifact["interpretation_annotations"]["REFINEMENT.target_phase_fraction"]
    assert "over-dispersed" in annotation["interpretation"]
    assert annotation["coverage50"] > 0.65


def test_no_threshold_metric_uses_999():
    text = (OUT / "multimodal_report.md").read_text(encoding="utf-8")
    assert "999" not in text


def test_report_uses_safe_posterior_language():
    text = (OUT / "multimodal_report.md").read_text(encoding="utf-8")
    assert "not probabilities that exactly one mechanism is true" in text
