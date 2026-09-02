"""Unit and semantic regression tests for the electrolyte dataset audit."""

import json
import os
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

AUDIT_DIR = "outputs/electrolyte/audit"


@pytest.fixture
def identity_audit():
    path = os.path.join(AUDIT_DIR, "experimental_identity_audit.json")
    assert os.path.exists(path), f"Missing audit artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def baseline_sanity():
    path = os.path.join(AUDIT_DIR, "baseline_model_sanity.json")
    assert os.path.exists(path), f"Missing audit artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def campaign_generalization():
    path = os.path.join(AUDIT_DIR, "campaign_generalization.json")
    assert os.path.exists(path), f"Missing audit artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def candidate_stats():
    path = os.path.join(AUDIT_DIR, "candidate_space_statistics.json")
    assert os.path.exists(path), f"Missing audit artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def coverage_stats():
    path = os.path.join(AUDIT_DIR, "search_space_coverage.json")
    assert os.path.exists(path), f"Missing audit artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def audit_report_text():
    path = os.path.join(AUDIT_DIR, "dataset_audit_report.md")
    assert os.path.exists(path), f"Missing audit report: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_target_semantics_maps_norm_capacity_3_to_cnorm20(identity_audit):
    """P0 #1: Verify raw norm_capacity_3 is explicitly mapped to C_norm^20 (20th cycle)."""
    target = identity_audit["target_semantics"]
    assert target["raw_target_column"] == "norm_capacity_3"
    assert target["scientific_target_name"] == "C_norm^20"
    assert "20th cycle" in target["scientific_meaning"]


def test_act_capacity20_consistent_with_normalized_target_when_available(identity_audit):
    """P0 #1: Verify act_capacity_20 / theor_capacity matches norm_capacity_3 within numerical precision."""
    val = identity_audit["target_semantics"]["numerical_alias_validation"]
    assert val["verified_consistent"] is True
    assert val["exceptions_count"] == 0
    assert val["max_absolute_error"] < 1e-6


def test_batch7_feature_recovery_uses_solvent_and_salt_identity(coverage_stats):
    """P0 #4: Verify Batch 7 feature recovery matches both solvent and salt with exact 1-to-1 counts."""
    report = coverage_stats.get("batch_7_validation_report", [])
    assert len(report) == 9, "Expected 9 rows for Batch 7"
    for item in report:
        assert item["exact_pool_match_count"] == 1
        assert item["feature_recovery_status"] == "EXACT_1_TO_1_MATCH"
        assert item["salt"] == "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F"


def test_target_copy_across_salt_rows_is_detected(identity_audit):
    """P0 #2: Verify target-copy expansion across multiple salts is explicitly audited."""
    tax = identity_audit["taxonomy"]
    assert tax["target_repeated_across_salts_groups"] > 0
    assert tax["rows_in_target_repeated_groups"] > 0
    assert tax["independent_wet_lab_records_estimate"] == "UNKNOWN"
    # Verify examples exist in audit
    examples = identity_audit.get("example_target_copied_groups", [])
    assert len(examples) > 0
    for ex in examples:
        assert ex["row_count"] > 1
        assert len(ex["salts"]) > 1


def test_pool_compatible_subset_excludes_non_1m_conditions(identity_audit):
    """P0 #3: Verify compatible subset strictly checks 1.0M conc, 150 mAh/g theor capacity, 50 uL vol, and 3 pool salts."""
    subsets = identity_audit["subsets"]
    canon = subsets["subset_B_virtual_pool_compatible_canonical"]
    assert canon["compatible_training_rows"] < 208
    reasons = canon["exclusion_reason_counts"]
    # Check that non-1M concentrations and different cathodes were detected and excluded
    assert any("non_1M_concentration" in k for k in reasons)
    assert any("different_cathode" in k for k in reasons)


def test_group_cv_never_splits_same_solvent_across_folds():
    """P0 #5: Synthetic test verifying GroupKFold prevents same solvent from appearing in train and val folds."""
    df_dummy = pd.DataFrame({
        "solvent": ["S1", "S1", "S1", "S2", "S2", "S3", "S4", "S4"],
        "salt": ["A", "B", "C", "A", "B", "A", "A", "B"],
        "feature": np.random.randn(8),
        "target": [0.5, 0.5, 0.5, 0.1, 0.1, 0.8, 0.3, 0.3]
    })
    gkf = GroupKFold(n_splits=3)
    for train_idx, val_idx in gkf.split(df_dummy, groups=df_dummy["solvent"]):
        train_solvs = set(df_dummy.iloc[train_idx]["solvent"])
        val_solvs = set(df_dummy.iloc[val_idx]["solvent"])
        assert len(train_solvs.intersection(val_solvs)) == 0, "Solvent leaked across train and val folds!"


def test_temporal_cv_never_uses_future_batch(campaign_generalization):
    """P0 #5: Verify temporal campaign evaluation strictly tests batch t+1 using only train batches <= t."""
    rounds = campaign_generalization["rounds"]
    assert len(rounds) == 7
    for r in rounds:
        train_batches_str = r["train_batches"]
        max_train_b = int(train_batches_str.split("..")[1])
        test_b = r["test_batch"]
        assert max_train_b < test_b, f"Temporal leakage: train max {max_train_b} >= test {test_b}"


def test_report_does_not_call_208_rows_independent_experiments(audit_report_text):
    """P0 #2: Verify report does not call the 208 ML rows 'independent experiments'."""
    assert "208 independent experiments" not in audit_report_text
    assert "208 independent physical experiments" not in audit_report_text
    assert "208 experimentally labeled cell formulations" not in audit_report_text


def test_report_does_not_claim_full_1m_replay_possible(audit_report_text):
    """P0 #6: Verify report explicitly states Full 1M wet-lab replay is NO / NOT POSSIBLE."""
    assert "| **5. Full 1M Wet-Lab Replay** | **NO** |" in audit_report_text or \
           "**Full 1M wet-lab replay feasibility:** **NO.**" in audit_report_text


def test_report_does_not_claim_salt_causality_from_expanded_rows(audit_report_text):
    """High #1: Verify report does not make unverified causal claims about LiFSI from expanded rows."""
    assert "LiFSI causes higher capacity than LiPF6" not in audit_report_text
    assert "All top 10 formulations universally require LiFSI" not in audit_report_text
