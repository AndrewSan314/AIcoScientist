from __future__ import annotations

import copy
from typing import Any
import pytest

from scripts.run_alab_validation import (
    audit_alab_dataset,
    generate_demonstration_report,
    validate_optimizer_gate,
    validate_report_consistency,
    validate_representation_contract,
)


def _get_valid_mock_audit() -> dict[str, Any]:
    """Returns a valid mock audit dictionary satisfying all 6 gates."""
    return {
        "candidate_identity": {
            "total_candidates": 1035,
            "unique_precursors_in_dataset": 46,
            "canonical_precursors_defined": 46,
            "unique_target_compounds": 90,
        },
        "outcome_semantics": {
            "total_samples": 1035,
            "classified_samples": 1009,
            "unclassified_samples": 26,
            "samples_with_physical_failure_field": 26,
            "samples_with_physical_failure_record": 26,
            "samples_with_phases_unavailable_due_to_physical_failure": 26,
            "unclassified_and_physical_failure": 26,
            "unclassified_without_physical_failure": 0,
            "utility_mapping": {
                "completely_reacted": 1.0,
                "transformed": 0.75,
                "partially_reacted": 0.5,
                "unreacted": 0.0,
                "unclassified": None,
            },
        },
        "characterization_coverage": {
            "total_scans": 1351,
            "samples_with_scans": 1035,
            "samples_with_no_scans": 0,
            "total_refinements": 1950,
            "samples_with_refinements": 1030,
        },
        "canonical_xrd_usability": {
            "canonical_xrd_usable_for_replay": 1035,
        },
        "canonical_refinement_usability": {
            "canonical_refinement_usable_for_replay": 1030,
        },
    }


def _get_valid_mock_benchmark() -> dict[str, Any]:
    return {
        "RANDOM": {
            "summary": {
                "mean_bootstrap_best_utility": 0.85,
                "mean_autonomous_improvement_amount": 0.05,
                "mean_autonomous_cost": 13.0,
                "mean_final_utility": 0.9,
                "std_final_utility": 0.1,
                "mean_final_entropy_nats": 1.0,
                "std_final_entropy_nats": 0.1,
                "total_objective_actions": 6,
                "total_characterization_actions": 0,
                "optimizer_expected": False,
                "optimizer_calls": 0,
                "optimizer_success_count": 0,
                "optimizer_failure_count": 0,
                "optimizer_degraded_count": 0,
                "optimizer_success_rate": None,
                "optimizer_validation_status": "NOT_APPLICABLE",
            },
        },
        "DISCOVERY_ONLY": {
            "summary": {
                "mean_bootstrap_best_utility": 0.85,
                "mean_autonomous_improvement_amount": 0.1,
                "mean_autonomous_cost": 13.0,
                "mean_final_utility": 0.95,
                "std_final_utility": 0.05,
                "mean_final_entropy_nats": 1.0,
                "std_final_entropy_nats": 0.1,
                "total_objective_actions": 6,
                "total_characterization_actions": 0,
                "optimizer_expected": True,
                "optimizer_calls": 6,
                "optimizer_success_count": 6,
                "optimizer_failure_count": 0,
                "optimizer_degraded_count": 0,
                "optimizer_success_rate": 1.0,
                "optimizer_validation_status": "VALIDATED",
            },
        },
        "PURE_FALSIFICATION": {
            "summary": {
                "mean_bootstrap_best_utility": 0.85,
                "mean_autonomous_improvement_amount": 0.0,
                "mean_autonomous_cost": 13.0,
                "mean_final_utility": 0.85,
                "std_final_utility": 0.1,
                "mean_final_entropy_nats": 0.7,
                "std_final_entropy_nats": 0.1,
                "total_objective_actions": 0,
                "total_characterization_actions": 13,
                "optimizer_expected": False,
                "optimizer_calls": 0,
                "optimizer_success_count": 0,
                "optimizer_failure_count": 0,
                "optimizer_degraded_count": 0,
                "optimizer_success_rate": None,
                "optimizer_validation_status": "NOT_APPLICABLE",
            },
        },
        "HYBRID": {
            "summary": {
                "mean_bootstrap_best_utility": 0.85,
                "mean_autonomous_improvement_amount": 0.1,
                "mean_autonomous_cost": 13.0,
                "mean_final_utility": 0.95,
                "std_final_utility": 0.05,
                "mean_final_entropy_nats": 0.8,
                "std_final_entropy_nats": 0.1,
                "total_objective_actions": 6,
                "total_characterization_actions": 0,
                "optimizer_expected": True,
                "optimizer_calls": 6,
                "optimizer_success_count": 6,
                "optimizer_failure_count": 0,
                "optimizer_degraded_count": 0,
                "optimizer_success_rate": 1.0,
                "optimizer_validation_status": "VALIDATED",
            },
        },
    }


@pytest.mark.external_data
def test_audit_dataset_sanity_gates_and_canonical_usability(tmp_path):
    """Verifies that audit_alab_dataset evaluates canonical usability and sanity gates on real data."""
    audit = audit_alab_dataset("data/external/precursor_genome_2026", tmp_path)

    # Sanity gates
    assert audit["candidate_identity"]["total_candidates"] == 1035
    assert audit["candidate_identity"]["unique_precursors_in_dataset"] == 46
    assert audit["outcome_semantics"]["classified_samples"] == 1009
    assert audit["outcome_semantics"]["unclassified_samples"] == 26
    assert audit["outcome_semantics"]["samples_with_physical_failure_field"] == 26
    assert audit["outcome_semantics"]["samples_with_physical_failure_record"] == 26
    assert audit["outcome_semantics"]["samples_with_phases_unavailable_due_to_physical_failure"] == 26
    assert audit["outcome_semantics"]["unclassified_and_physical_failure"] == 26
    assert audit["outcome_semantics"]["unclassified_without_physical_failure"] == 0

    # Canonical usability
    xrd_use = audit["canonical_xrd_usability"]
    assert xrd_use["canonical_xrd_usable_for_replay"] == 1035
    assert xrd_use["canonical_xrd_artifact_resolvable"] == 1035
    assert xrd_use["canonical_xrd_xml_parsable"] == 1035
    assert xrd_use["canonical_xrd_xml_malformed"] == 0
    assert xrd_use["canonical_xrd_axis_missing"] == 0
    assert xrd_use["canonical_xrd_intensity_missing"] == 0
    assert xrd_use["canonical_xrd_preprocessing_failed"] == 0

    ref_use = audit["canonical_refinement_usability"]
    assert ref_use["canonical_refinement_usable_for_replay"] == 1030
    assert ref_use["canonical_refinement_from_structured_ledger"] == 1030


def test_validation_verdict_earned_from_explicit_gates(tmp_path):
    """Verifies that generate_demonstration_report computes gate-driven verdict."""
    audit = _get_valid_mock_audit()
    mock_benchmark = _get_valid_mock_benchmark()
    mock_wow = {"seed": 42, "verification_status": "HONEST_UNFABRICATED_REPLAY", "steps": []}

    report = generate_demonstration_report(audit, mock_benchmark, mock_wow, tmp_path)

    assert "SCIENTIFIC VALIDATION READY" in report
    assert "PRODUCTION READY" not in report
    assert "`dataset_schema_sane`: **PASS**" in report
    assert "`canonical_artifact_identity_valid`: **PASS**" in report
    assert "`missing_outcomes_fail_closed`: **PASS**" in report
    assert "`optimizer_semantics_valid`: **PASS**" in report
    assert "`report_consistency_valid`: **PASS**" in report
    assert "The discovery threshold was already reached during initialization" in report


def test_validation_verdict_integration_ready_when_scientific_gate_fails(tmp_path):
    """Verifies that if basic integration passes but scientific validation gate fails, verdict is INTEGRATION READY."""
    audit = _get_valid_mock_audit()
    mock_benchmark = _get_valid_mock_benchmark()
    # Corrupt optimizer semantics: DISCOVERY_ONLY made 0 calls
    mock_benchmark["DISCOVERY_ONLY"]["summary"]["optimizer_calls"] = 0
    mock_benchmark["DISCOVERY_ONLY"]["summary"]["optimizer_validation_status"] = "FAILED_NO_CALLS"
    mock_benchmark["DISCOVERY_ONLY"]["summary"]["optimizer_success_rate"] = 0.0
    mock_wow = {"seed": 42, "verification_status": "HONEST_UNFABRICATED_REPLAY", "steps": []}

    report = generate_demonstration_report(audit, mock_benchmark, mock_wow, tmp_path)

    assert "INTEGRATION READY" in report
    assert "SCIENTIFIC VALIDATION READY" not in report
    assert "NOT READY" not in report
    assert "`optimizer_semantics_valid`: **FAIL**" in report


def test_validation_verdict_fails_closed_when_gate_fails(tmp_path):
    """Verifies that if basic schema gate fails, the verdict is NOT READY."""
    audit = _get_valid_mock_audit()
    # Corrupt schema count
    audit["outcome_semantics"]["classified_samples"] = 999

    mock_benchmark = _get_valid_mock_benchmark()
    mock_wow = {"seed": 42, "verification_status": "HONEST_UNFABRICATED_REPLAY", "steps": []}

    report = generate_demonstration_report(audit, mock_benchmark, mock_wow, tmp_path)

    assert "NOT READY" in report
    assert "SCIENTIFIC VALIDATION READY" not in report
    assert "INTEGRATION READY" not in report
    assert "`dataset_schema_sane`: **FAIL**" in report


def test_optimizer_gate_fails_when_expected_optimizer_never_runs():
    """Verifies that validate_optimizer_gate fails when DISCOVERY_ONLY made zero calls."""
    benchmark = _get_valid_mock_benchmark()
    benchmark["DISCOVERY_ONLY"]["summary"]["optimizer_calls"] = 0
    benchmark["DISCOVERY_ONLY"]["summary"]["optimizer_validation_status"] = "FAILED_NO_CALLS"
    benchmark["DISCOVERY_ONLY"]["summary"]["optimizer_success_rate"] = 0.0

    passed, msg = validate_optimizer_gate(benchmark)
    assert passed is False
    assert "expected optimizer calls" in msg or "FAILED_NO_CALLS" in msg


def test_optimizer_gate_fails_when_discovery_optimizer_has_failures():
    """Verifies that validate_optimizer_gate fails when discovery optimizer encountered failures."""
    benchmark = _get_valid_mock_benchmark()
    benchmark["DISCOVERY_ONLY"]["summary"]["optimizer_failure_count"] = 2
    benchmark["DISCOVERY_ONLY"]["summary"]["optimizer_success_rate"] = 0.6667
    benchmark["DISCOVERY_ONLY"]["summary"]["optimizer_validation_status"] = "UNEXPECTED_FAILURES"

    passed, msg = validate_optimizer_gate(benchmark)
    assert passed is False
    assert "optimizer failures" in msg or "UNEXPECTED_FAILURES" in msg


def test_optimizer_gate_ignores_random_and_pure_falsification_as_not_applicable():
    """Verifies that RANDOM and PURE_FALSIFICATION have optimizer_expected=False and pass gate as NOT_APPLICABLE."""
    benchmark = _get_valid_mock_benchmark()
    assert benchmark["RANDOM"]["summary"]["optimizer_expected"] is False
    assert benchmark["RANDOM"]["summary"]["optimizer_validation_status"] == "NOT_APPLICABLE"
    assert benchmark["PURE_FALSIFICATION"]["summary"]["optimizer_expected"] is False
    assert benchmark["PURE_FALSIFICATION"]["summary"]["optimizer_validation_status"] == "NOT_APPLICABLE"

    passed, msg = validate_optimizer_gate(benchmark)
    assert passed is True


def test_report_consistency_detects_action_count_mismatch():
    """Verifies that validate_report_consistency detects when summary action counts do not match seed sums."""
    audit = _get_valid_mock_audit()
    benchmark = _get_valid_mock_benchmark()
    benchmark["HYBRID"]["seeds"] = {
        "42": {
            "autonomous_action_counts": {"OUTCOME_TEST": 2},
            "autonomous_steps": 2,
            "final_max_utility": 0.95,
            "final_entropy_nats": 0.8,
        },
        "101": {
            "autonomous_action_counts": {"OUTCOME_TEST": 2},
            "autonomous_steps": 2,
            "final_max_utility": 0.95,
            "final_entropy_nats": 0.8,
        },
    }
    # Summary claims 6, but seeds sum to 4
    benchmark["HYBRID"]["summary"]["total_objective_actions"] = 6
    wow = {"seed": 42, "verification_status": "HONEST_UNFABRICATED_REPLAY", "steps": []}

    passed, issues = validate_report_consistency(audit, benchmark, wow)
    assert passed is False
    assert any("total_objective_actions" in issue for issue in issues)


def test_report_consistency_detects_entropy_aggregate_mismatch():
    """Verifies that validate_report_consistency detects entropy mean discrepancies."""
    audit = _get_valid_mock_audit()
    benchmark = _get_valid_mock_benchmark()
    benchmark["HYBRID"]["seeds"] = {
        "42": {
            "autonomous_action_counts": {"OUTCOME_TEST": 3},
            "autonomous_steps": 3,
            "final_max_utility": 0.95,
            "final_entropy_nats": 0.6,
        },
        "101": {
            "autonomous_action_counts": {"OUTCOME_TEST": 3},
            "autonomous_steps": 3,
            "final_max_utility": 0.95,
            "final_entropy_nats": 0.8,
        },
    }
    benchmark["HYBRID"]["summary"]["total_objective_actions"] = 6
    benchmark["HYBRID"]["summary"]["mean_final_entropy_nats"] = 0.9999  # Should be 0.7000
    wow = {"seed": 42, "verification_status": "HONEST_UNFABRICATED_REPLAY", "steps": []}

    passed, issues = validate_report_consistency(audit, benchmark, wow)
    assert passed is False
    assert any("mean_final_entropy_nats" in issue for issue in issues)


def test_report_policy_description_matches_action_mix(tmp_path):
    """Verifies that generate_demonstration_report generates dynamic prose faithful to actual action distributions."""
    audit = _get_valid_mock_audit()
    benchmark = _get_valid_mock_benchmark()
    wow = {"seed": 42, "verification_status": "HONEST_UNFABRICATED_REPLAY", "steps": []}

    report = generate_demonstration_report(audit, benchmark, wow, tmp_path)
    assert "allocated all autonomous actions to characterization (13 actions, 0 objective tests)" in report
    assert "allocated all autonomous actions to objective measurements (6 objective, 0 characterization)" in report
    assert "HYBRID` selected only outcome measurements during the autonomous phase" in report


@pytest.mark.external_data
def test_audit_counts_sample_physical_failure_separately_from_phase_unavailable_reason(tmp_path):
    """Verifies that sample.physical_failure is counted independently from phases_unavailable_reason."""
    audit = audit_alab_dataset("data/external/precursor_genome_2026", tmp_path)
    outc = audit["outcome_semantics"]
    assert "samples_with_physical_failure_record" in outc
    assert "samples_with_phases_unavailable_due_to_physical_failure" in outc
    assert "unclassified_and_physical_failure" in outc
    assert "unclassified_without_physical_failure" in outc
    assert outc["samples_with_physical_failure_record"] == 26
    assert outc["samples_with_phases_unavailable_due_to_physical_failure"] == 26
    assert outc["unclassified_and_physical_failure"] == 26
    assert outc["unclassified_without_physical_failure"] == 0

