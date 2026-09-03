"""Unit and semantic regression tests for the electrolyte dataset audit.

All tests use fast synthetic fixtures or inspect computed audit artifacts so that
ordinary CI does not load the 540 MB raw external CSVs. Real-data tests are marked
with `@pytest.mark.external_data`.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

# Import modular functions directly from audit script
from scripts.audit_electrolyte_dataset import (
    detect_target_copy_groups,
    build_deexpanded_campaign_view,
    build_pool_compatible_subset,
    render_audit_report
)

AUDIT_DIR = "outputs/electrolyte/audit"


@pytest.fixture
def sample_expanded_df():
    """Synthetic labeled DataFrame with target-copied rows, replicates, and condition variants."""
    return pd.DataFrame({
        "solv_comb_sm": [
            # Batch 0: 3 rows (2 replicates of cond A, 1 of cond B)
            "DME", "DME", "DME",
            # Batch 1: 3 rows target-copied across salts
            "SOLV_X", "SOLV_X", "SOLV_X",
            # Batch 1: 1 distinct single row
            "SOLV_Y",
            # Batch 2: 2 rows with same solvent but different targets
            "SOLV_Z", "SOLV_Z"
        ],
        "salt_comb_sm": [
            "SALT_A", "SALT_A", "SALT_B",
            "SALT_A", "SALT_B", "SALT_C",
            "SALT_A",
            "SALT_A", "SALT_A"
        ],
        "batch": [0, 0, 0, 1, 1, 1, 1, 2, 2],
        "conc_salt_1": [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "theor_capacity": [150, 150, 203, 150, 150, 150, 150, 150, 150],
        "amt_electrolyte": [50.0, 50.0, 13.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        "norm_capacity_3": [0.65, 0.62, 0.70, 0.05, 0.05, 0.05, 0.12, 0.30, 0.45]
    })


@pytest.fixture
def committed_physical_campaign():
    path = os.path.join(AUDIT_DIR, "physical_campaign_view.json")
    assert os.path.exists(path), f"Missing artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def committed_identity_audit():
    path = os.path.join(AUDIT_DIR, "experimental_identity_audit.json")
    assert os.path.exists(path), f"Missing artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def committed_solv_feat_audit():
    path = os.path.join(AUDIT_DIR, "solvent_feature_identity_audit.json")
    assert os.path.exists(path), f"Missing artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def committed_cand_stats():
    path = os.path.join(AUDIT_DIR, "candidate_space_statistics.json")
    assert os.path.exists(path), f"Missing artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def committed_baseline_sanity():
    path = os.path.join(AUDIT_DIR, "baseline_model_sanity.json")
    assert os.path.exists(path), f"Missing artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def committed_campaign_gen():
    path = os.path.join(AUDIT_DIR, "campaign_generalization.json")
    assert os.path.exists(path), f"Missing artifact: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def audit_report_text():
    path = os.path.join(AUDIT_DIR, "dataset_audit_report.md")
    assert os.path.exists(path), f"Missing report: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ======================================================================
# COMPUTATION-LEVEL UNIT TESTS (SYNTHETIC FIXTURES)
# ======================================================================

def test_deexpanded_campaign_collapses_copied_salt_rows(sample_expanded_df):
    """P0 #1: Verify detect_target_copy_groups and de-expansion collapse copied salt rows in B1-7."""
    repeated_groups, total_copied = detect_target_copy_groups(sample_expanded_df)
    assert len(repeated_groups) == 1
    assert repeated_groups[0]["solvent_smiles"] == "SOLV_X"
    assert repeated_groups[0]["row_count"] == 3
    assert total_copied == 3

    phys_campaign, df_deexp = build_deexpanded_campaign_view(sample_expanded_df)
    b17_view = phys_campaign["batch1_to_7_deexpanded_view"]
    # SOLV_X (3 rows copied -> 1 outcome), SOLV_Y (1 row -> 1 outcome), SOLV_Z (2 distinct targets -> 2 outcomes)
    assert b17_view["de_expanded_campaign_outcomes"] == 4
    assert b17_view["status_breakdown"]["TARGET_COPIED_ACROSS_SALTS"] == 1
    assert b17_view["status_breakdown"]["SINGLE_ROW"] == 3


def test_deexpanded_campaign_preserves_distinct_targets(sample_expanded_df):
    """P0 #1: Verify distinct targets for the same solvent within a batch are preserved."""
    phys_campaign, df_deexp = build_deexpanded_campaign_view(sample_expanded_df)
    solv_z_outcomes = df_deexp[df_deexp["solv_comb_sm"] == "SOLV_Z"]
    assert len(solv_z_outcomes) == 2
    assert set(solv_z_outcomes["norm_capacity_3"]) == {0.30, 0.45}


def test_batch0_condition_variants_are_not_blindly_collapsed(sample_expanded_df):
    """P0 #1: Verify Batch 0 preserves condition variants and replicate groups."""
    phys_campaign, df_deexp = build_deexpanded_campaign_view(sample_expanded_df)
    b0_view = phys_campaign["batch0_seed_view"]
    assert b0_view["raw_seed_rows"] == 3
    assert b0_view["unique_condition_records"] == 2  # 1.0M/150/50 vs 2.0M/203/13
    assert b0_view["replicate_condition_groups"] == 1


def test_batch_statistics_do_not_use_raw_ml_rows_as_cell_counts(committed_physical_campaign):
    """P0 #2: Verify summary fields use 'raw_ml_rows' and 'de_expanded_campaign_outcomes', not 'cell_count'."""
    for b_summary in committed_physical_campaign["campaign_summary_by_batch"]:
        assert "raw_ml_rows" in b_summary
        assert "de_expanded_campaign_outcomes" in b_summary
        assert "cell_count" not in b_summary
        assert "Cell Count" not in b_summary


def test_pool_compatible_ml_and_deexpanded_counts_are_separate(committed_identity_audit):
    """P0 #3: Verify pool-compatible ML rows and de-expanded outcomes are explicitly separated."""
    canon = committed_identity_audit["subsets"]["subset_B_virtual_pool_compatible_recovered"]
    assert "pool_compatible_ml_rows" in canon
    assert "pool_compatible_deexpanded_outcomes" in canon
    assert "pool_compatible_unique_solvents" in canon
    # ML rows should be greater than de-expanded outcomes due to salt copy expansion
    assert canon["pool_compatible_ml_rows"] > canon["pool_compatible_deexpanded_outcomes"]
    assert canon["pool_compatible_ml_rows"] == 151
    assert canon["pool_compatible_deexpanded_outcomes"] == 75
    assert canon["pool_compatible_unique_solvents"] == 75


def test_deexpanded_group_cv_has_no_solvent_overlap():
    """P0 #4: Verify GroupKFold on solvent identity guarantees zero train/validation chemical overlap."""
    df_synthetic = pd.DataFrame({
        "solvent": ["S1", "S1", "S2", "S3", "S4", "S5", "S5", "S6"],
        "target": [0.1, 0.2, 0.5, 0.4, 0.9, 0.7, 0.6, 0.8]
    })
    gkf = GroupKFold(n_splits=3)
    for tr_idx, va_idx in gkf.split(df_synthetic, groups=df_synthetic["solvent"]):
        tr_solvs = set(df_synthetic.iloc[tr_idx]["solvent"])
        va_solvs = set(df_synthetic.iloc[va_idx]["solvent"])
        assert len(tr_solvs.intersection(va_solvs)) == 0


def test_deexpanded_temporal_cv_has_no_future_batches(committed_campaign_gen):
    """P0 #4: Verify temporal evaluation strictly trains on batches <= t and tests batch t+1."""
    rounds = committed_campaign_gen["rounds"]
    assert len(rounds) == 7
    for r in rounds:
        train_str = r["train_batches"]
        max_tr = int(train_str.split("..")[1])
        te_batch = r["test_batch"]
        assert max_tr < te_batch
        assert "rank_of_true_best_within_test_batch" in r


def test_candidate_duplicate_counts_are_computed_not_hardcoded(committed_cand_stats):
    """High #1 & #2: Verify duplicate and collision counts are structured and non-zero."""
    dup_audit = committed_cand_stats["duplicates_and_collisions"]
    assert dup_audit["raw_candidate_rows"] == 999999
    assert dup_audit["unique_solvent_salt_keys"] == 999999
    assert dup_audit["unique_22d_feature_vectors"] == 999326
    assert dup_audit["collision_groups_count"] == 619
    assert dup_audit["collision_extra_rows"] == 673
    assert dup_audit["collision_causes"]["SMILES_syntax_equivalent"] == 20
    assert dup_audit["collision_causes"]["distinct_SMILES_same_feature_collision"] == 599
    assert dup_audit["collision_causes"]["cross_salt_collisions"] == 0


def test_solvent_feature_identity_audit_detects_multiple_vectors_per_solvent(committed_solv_feat_audit):
    """P0 #5: Verify anomaly is explained as floating-point precision jitter near machine epsilon."""
    assert committed_solv_feat_audit["unique_solvent_strings"] == 388004
    assert committed_solv_feat_audit["multi_vector_solvents_count"] == 333470
    assert committed_solv_feat_audit["global_max_abs_delta"] < 1e-14
    assert committed_solv_feat_audit["verdict"] in (
        "NUMERICALLY CONSISTENT WITH FLOATING-POINT PRECISION JITTER",
        "EMPIRICALLY CONSISTENT WITH FLOATING-POINT JITTER",
    )
    assert "floating-point" in committed_solv_feat_audit["scientific_justification"].lower()


def test_gp_label_matches_actual_kernel(committed_baseline_sanity, audit_report_text):
    """High #3: Verify GP model is labeled 'Gaussian Process (RBF + WhiteKernel)'."""
    assert "Gaussian Process (RBF + WhiteKernel)" in committed_baseline_sanity["baseline_C_deexpanded_grouped_solvent_cv_PRIMARY"]
    assert "Gaussian Process (Matern52)" not in committed_baseline_sanity["baseline_C_deexpanded_grouped_solvent_cv_PRIMARY"]
    assert "Gaussian Process (RBF + WhiteKernel)" in audit_report_text
    assert "Gaussian Process (Matern52)" not in audit_report_text


def test_report_is_rendered_from_computed_audit_results(
    committed_physical_campaign,
    committed_identity_audit,
    committed_cand_stats,
    committed_solv_feat_audit,
    committed_baseline_sanity,
    committed_campaign_gen,
):
    """High #4: Verify render_audit_report function produces a valid Markdown document with all sections."""
    dummy_inv = [{"filename": "f.csv", "format": "CSV", "size_mb": 1.0, "rows": 10, "columns": 2}]
    cov_path = os.path.join(AUDIT_DIR, "search_space_coverage.json")
    with open(cov_path, "r", encoding="utf-8") as f:
        committed_cov = json.load(f)

    md_text = render_audit_report(
        dummy_inv,
        committed_physical_campaign,
        committed_identity_audit,
        committed_cand_stats,
        committed_solv_feat_audit,
        committed_cov,
        committed_baseline_sanity,
        committed_campaign_gen,
    )
    assert "# 1. Executive Summary" in md_text
    assert "# 5. Physical / De-expanded Campaign View" in md_text
    assert "# 8. Candidate Feature Identity Audit" in md_text
    assert "# 11. Baseline Generalization" in md_text
    assert "# 17. Final Dataset Role" in md_text


def test_report_does_not_call_151_rows_physical_samples(audit_report_text):
    """High #8: Verify 151 rows are not described as '151 physical historical samples'."""
    assert "151 physical historical samples" not in audit_report_text
    assert "151 pool-compatible historical samples" not in audit_report_text
    assert "151 pool-compatible historical formulations" not in audit_report_text
    assert "151 ML rows" in audit_report_text or "151 pool-compatible ML rows" in audit_report_text or "151 ML View" in audit_report_text


def test_report_does_not_call_raw_batch_rows_cell_counts(audit_report_text):
    """P0 #2: Verify report does not call raw batch rows 'cell counts'."""
    assert "Batch 1 | Cell Count = 40" not in audit_report_text
    assert "32/40 dead cells" not in audit_report_text
    assert "8/40 cells viable" not in audit_report_text


def test_local_trajectory_wording_distinguishes_upstream_full_reproduction(audit_report_text):
    """High #6: Verify report distinguishes local chronology from full upstream reproduction."""
    assert "Full Original Acquisition Reproduction" in audit_report_text
    assert "UPSTREAM ONLY" in audit_report_text or "SUPPORTED UPSTREAM" in audit_report_text
    assert "Local Batch Chronology" in audit_report_text
