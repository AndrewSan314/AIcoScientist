from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.run_alab_validation import (
    audit_alab_dataset,
    generate_demonstration_report,
)


def test_audit_dataset_sanity_gates_and_canonical_usability(tmp_path):
    """Verifies that audit_alab_dataset evaluates canonical usability and sanity gates."""
    audit = audit_alab_dataset("data/external/precursor_genome_2026", tmp_path)

    # Sanity gates
    assert audit["candidate_identity"]["total_candidates"] == 1035
    assert audit["candidate_identity"]["unique_precursors_in_dataset"] == 46
    assert audit["outcome_semantics"]["classified_samples"] == 1009
    assert audit["outcome_semantics"]["unclassified_samples"] == 26
    assert audit["outcome_semantics"]["samples_with_physical_failure_field"] == 26

    # Canonical usability
    xrd_use = audit["canonical_xrd_usability"]
    assert xrd_use["canonical_xrd_usable_for_replay"] == 1035
    assert xrd_use["canonical_xrd_artifact_resolvable"] == 1035

    ref_use = audit["canonical_refinement_usability"]
    assert ref_use["canonical_refinement_usable_for_replay"] == 1030
    assert ref_use["canonical_refinement_from_structured_ledger"] == 1030


def test_validation_verdict_earned_from_explicit_gates(tmp_path):
    """Verifies that generate_demonstration_report computes gate-driven verdict."""
    audit = audit_alab_dataset("data/external/precursor_genome_2026", tmp_path)

    mock_benchmark = {
        "RANDOM": {"summary": {"mean_bootstrap_best_utility": 0.85, "mean_autonomous_improvement_amount": 0.05, "mean_autonomous_cost": 13.0, "mean_final_utility": 0.9, "std_final_utility": 0.1, "mean_final_entropy_nats": 1.0, "std_final_entropy_nats": 0.1, "total_objective_actions": 6, "total_characterization_actions": 0, "optimizer_success_rate": 1.0}},
        "DISCOVERY_ONLY": {"summary": {"mean_bootstrap_best_utility": 0.85, "mean_autonomous_improvement_amount": 0.1, "mean_autonomous_cost": 13.0, "mean_final_utility": 0.95, "std_final_utility": 0.05, "mean_final_entropy_nats": 1.0, "std_final_entropy_nats": 0.1, "total_objective_actions": 6, "total_characterization_actions": 0, "optimizer_success_rate": 1.0}},
        "PURE_FALSIFICATION": {"summary": {"mean_bootstrap_best_utility": 0.85, "mean_autonomous_improvement_amount": 0.0, "mean_autonomous_cost": 13.0, "mean_final_utility": 0.85, "std_final_utility": 0.1, "mean_final_entropy_nats": 0.7, "std_final_entropy_nats": 0.1, "total_objective_actions": 2, "total_characterization_actions": 11, "optimizer_success_rate": 1.0}},
        "HYBRID": {"summary": {"mean_bootstrap_best_utility": 0.85, "mean_autonomous_improvement_amount": 0.1, "mean_autonomous_cost": 13.0, "mean_final_utility": 0.95, "std_final_utility": 0.05, "mean_final_entropy_nats": 0.8, "std_final_entropy_nats": 0.1, "total_objective_actions": 6, "total_characterization_actions": 0, "optimizer_success_rate": 1.0}},
    }
    mock_wow = {"seed": 42, "verification_status": "HONEST_UNFABRICATED_REPLAY", "steps": []}

    report = generate_demonstration_report(audit, mock_benchmark, mock_wow, tmp_path)

    assert "SCIENTIFIC VALIDATION READY" in report
    assert "PRODUCTION READY" not in report
    assert "`dataset_schema_sane`: **PASS**" in report
    assert "`canonical_artifact_identity_valid`: **PASS**" in report
    assert "`missing_outcomes_fail_closed`: **PASS**" in report
    assert "The discovery threshold was already reached during initialization" in report


def test_validation_verdict_fails_closed_when_gate_fails(tmp_path):
    """Verifies that if any gate fails, the verdict is NOT READY."""
    audit = audit_alab_dataset("data/external/precursor_genome_2026", tmp_path)
    # Simulate a corrupted schema audit
    audit["outcome_semantics"]["classified_samples"] = 999  # Invalid!

    mock_benchmark = {
        "RANDOM": {"summary": {"mean_bootstrap_best_utility": 0.85, "mean_autonomous_improvement_amount": 0.0, "mean_autonomous_cost": 13.0, "mean_final_utility": 0.85, "std_final_utility": 0.0, "mean_final_entropy_nats": 1.0, "std_final_entropy_nats": 0.0, "total_objective_actions": 0, "total_characterization_actions": 0, "optimizer_success_rate": 1.0}}
    }
    mock_wow = {"seed": 42, "verification_status": "HONEST_UNFABRICATED_REPLAY", "steps": []}

    report = generate_demonstration_report(audit, mock_benchmark, mock_wow, tmp_path)

    assert "NOT READY" in report
    assert "SCIENTIFIC VALIDATION READY" not in report
    assert "`dataset_schema_sane`: **FAIL**" in report
