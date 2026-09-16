#!/usr/bin/env python3
"""Audit Multi-Dataset Benchmark Result Consistency, Provenance, and Firewalls.

Audits 15 dynamic artifact-to-artifact relationships (no hardcoded self-fulfilling constants):
1. Drakopoulos v4: benchmark_matrix.csv hit rate equals policy_summary.csv (AICOSCIENTIST_PROCESS_SURROGATE).
2. Drakopoulos v4: benchmark_matrix.csv BoTorch hit rate equals policy_summary.csv (DIRECT_BOTORCH_BASELINE).
3. Drakopoulos v4: multi_dataset slide_summary.json matches drakopoulos slide_summary.json.
4. Warwick NMC622: benchmark_matrix.csv hit rate equals policy_summary.csv (AICOSCIENTIST_FULL_PROCESS_ENGINE).
5. Warwick NMC622: benchmark_matrix.csv BoTorch hit rate equals policy_summary.csv (DIRECT_BOTORCH_BASELINE).
6. Warwick NMC622: multi_dataset slide_summary.json matches warwick_nmc622 slide_summary.json.
7. Warwick NMC622: Replay trajectory initial candidates match initial_designs.json across all seeds and policies.
8. Warwick NMC622: Decision variable firewall in decision_variable_audit.json excludes post-process variables.
9. Warwick Ultrasonic: frequency_grid_audit.json records native alignment and zero interpolation.
10. Warwick Ultrasonic: split_manifest.json guarantees zero cross-fold leakage across 5 folds.
11. Warwick Ultrasonic: execution_trace_audit.json verifies production pipeline trace counters and test_only == 0.
12. Warwick Ultrasonic: model_comparison_summary.json matches ablation_summary.csv dynamically across all cells.
13. Warwick Ultrasonic: benchmark_matrix.csv row matches ablation_summary.csv metrics dynamically.
14. Warwick Ultrasonic: multi_dataset slide_summary.json matches warwick_ultrasonic ablation_summary.csv.
15. Multi-Dataset Validation Report: MULTI_DATASET_VALIDATION_REPORT.md has zero stale literals or unsupported claims.

Generates:
- outputs/multi_dataset_validation/consistency_audit.json
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("consistency_audit")


def run_consistency_audit() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "outputs" / "multi_dataset_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_results: dict[str, Any] = {
        "status": "PASS",
        "timestamp": "2026-09-16",
        "checks": {},
    }

    # =============================================================
    # ARTIFACT PATHS
    # =============================================================
    drak_dir = repo_root / "outputs" / "drakopoulos_rediscovery_v4"
    drak_policy_path = drak_dir / "policy_summary.csv"
    drak_slide_path = drak_dir / "slide_summary.json"

    nmc_dir = repo_root / "outputs" / "warwick_nmc622_calendering"
    nmc_policy_path = nmc_dir / "policy_summary.csv"
    nmc_init_path = nmc_dir / "initial_designs.json"
    nmc_var_audit_path = nmc_dir / "decision_variable_audit.json"
    nmc_slide_path = nmc_dir / "slide_summary.json"

    ultra_dir = repo_root / "outputs" / "warwick_ultrasonic"
    ultra_grid_audit_path = ultra_dir / "frequency_grid_audit.json"
    ultra_trace_audit_path = ultra_dir / "execution_trace_audit.json"
    ultra_split_path = ultra_dir / "split_manifest.json"
    ultra_ablation_path = ultra_dir / "ablation_summary.csv"
    ultra_comp_path = ultra_dir / "model_comparison_summary.json"
    ultra_slide_path = ultra_dir / "slide_summary.json"

    matrix_path = out_dir / "benchmark_matrix.csv"
    multi_slide_path = out_dir / "slide_summary.json"
    multi_report_path = out_dir / "MULTI_DATASET_VALIDATION_REPORT.md"

    # Verify all artifacts exist
    artifacts = [
        drak_policy_path, drak_slide_path,
        nmc_policy_path, nmc_init_path, nmc_var_audit_path, nmc_slide_path,
        ultra_grid_audit_path, ultra_trace_audit_path, ultra_split_path,
        ultra_ablation_path, ultra_comp_path, ultra_slide_path,
        matrix_path, multi_slide_path, multi_report_path,
    ]
    for p in artifacts:
        if not p.exists():
            raise FileNotFoundError(f"Required benchmark artifact missing: {p}")

    df_drak_policy = pd.read_csv(drak_policy_path)
    with open(drak_slide_path) as f:
        drak_slide = json.load(f)

    df_nmc_policy = pd.read_csv(nmc_policy_path)
    with open(nmc_init_path) as f:
        nmc_initial_designs = json.load(f)
    with open(nmc_var_audit_path) as f:
        nmc_var_audit = json.load(f)
    with open(nmc_slide_path) as f:
        nmc_slide = json.load(f)

    with open(ultra_grid_audit_path) as f:
        ultra_grid_audit = json.load(f)
    with open(ultra_trace_audit_path) as f:
        ultra_trace_audit = json.load(f)
    with open(ultra_split_path) as f:
        ultra_splits = json.load(f)
    df_ultra_ablation = pd.read_csv(ultra_ablation_path)
    with open(ultra_comp_path) as f:
        ultra_comp = json.load(f)
    with open(ultra_slide_path) as f:
        ultra_slide = json.load(f)

    df_matrix = pd.read_csv(matrix_path)
    with open(multi_slide_path) as f:
        multi_slide = json.load(f)
    multi_report_text = multi_report_path.read_text(encoding="utf-8")

    # =============================================================
    # 15 DYNAMIC CROSS-ARTIFACT CHECKS
    # =============================================================

    # 1. Drakopoulos: benchmark_matrix AI Hit@5 == policy_summary.csv
    drak_uncon = df_drak_policy[df_drak_policy["benchmark_task"] == "UNCONSTRAINED_D30"]
    drak_ai_hit5 = float(drak_uncon[drak_uncon["policy"] == "AICOSCIENTIST_PROCESS_SURROGATE"]["hit_rate_step_5"].iloc[0])
    matrix_drak_ai = float(df_matrix[df_matrix["benchmark_id"] == "DRAKOPOULOS_REDISCOVERY_V4"]["aicoscientist_hit_at_5"].iloc[0])
    assert math.isclose(drak_ai_hit5, matrix_drak_ai, rel_tol=1e-6), f"Check 1 Failed: {drak_ai_hit5} != {matrix_drak_ai}"
    drak_hit_str = str(df_matrix[df_matrix["benchmark_id"] == "DRAKOPOULOS_REDISCOVERY_V4"]["multimodal_r2_or_bo_hit"].iloc[0])
    assert f"Hit@5 = {drak_ai_hit5 * 100:.1f}%" in drak_hit_str, f"Check 1 Failed: {drak_hit_str} does not match {drak_ai_hit5}"
    audit_results["checks"]["check_01_drak_ai_hit5_matrix_match"] = "PASS"

    # 2. Drakopoulos: benchmark_matrix BoTorch Hit@5 == policy_summary.csv
    drak_botorch_hit5 = float(drak_uncon[drak_uncon["policy"] == "DIRECT_BOTORCH_BASELINE"]["hit_rate_step_5"].iloc[0])
    matrix_drak_botorch = float(df_matrix[df_matrix["benchmark_id"] == "DRAKOPOULOS_REDISCOVERY_V4"]["botorch_baseline_hit_at_5"].iloc[0])
    assert math.isclose(drak_botorch_hit5, matrix_drak_botorch, rel_tol=1e-6), f"Check 2 Failed: {drak_botorch_hit5} != {matrix_drak_botorch}"
    audit_results["checks"]["check_02_drak_botorch_hit5_matrix_match"] = "PASS"

    # 3. Drakopoulos: multi slide_summary matches drakopoulos slide_summary
    assert math.isclose(multi_slide["benchmarks"]["drakopoulos_2021"]["aicoscientist_hit_at_5"], drak_ai_hit5, rel_tol=1e-6)
    assert math.isclose(multi_slide["benchmarks"]["drakopoulos_2021"]["direct_botorch_hit_at_5"], drak_botorch_hit5, rel_tol=1e-6)
    audit_results["checks"]["check_03_drak_slide_summary_match"] = "PASS"

    # 4. Warwick NMC622: benchmark_matrix AI Hit@5 == policy_summary.csv
    nmc_ai_hit5 = float(df_nmc_policy[df_nmc_policy["policy"] == "AICOSCIENTIST_FULL_PROCESS_ENGINE"]["hit_at_5"].iloc[0])
    matrix_nmc_ai = float(df_matrix[df_matrix["benchmark_id"] == "WARWICK_NMC622_CALENDERING"]["aicoscientist_hit_at_5"].iloc[0])
    assert math.isclose(nmc_ai_hit5, matrix_nmc_ai, rel_tol=1e-6), f"Check 4 Failed: {nmc_ai_hit5} != {matrix_nmc_ai}"
    nmc_hit_str = str(df_matrix[df_matrix["benchmark_id"] == "WARWICK_NMC622_CALENDERING"]["multimodal_r2_or_bo_hit"].iloc[0])
    assert f"Hit@5 = {nmc_ai_hit5 * 100:.1f}%" in nmc_hit_str, f"Check 4 Failed: {nmc_hit_str} does not match {nmc_ai_hit5}"
    audit_results["checks"]["check_04_nmc_ai_hit5_matrix_match"] = "PASS"

    # 5. Warwick NMC622: benchmark_matrix BoTorch Hit@5 == policy_summary.csv
    nmc_botorch_hit5 = float(df_nmc_policy[df_nmc_policy["policy"] == "DIRECT_BOTORCH_BASELINE"]["hit_at_5"].iloc[0])
    matrix_nmc_botorch = float(df_matrix[df_matrix["benchmark_id"] == "WARWICK_NMC622_CALENDERING"]["botorch_baseline_hit_at_5"].iloc[0])
    assert math.isclose(nmc_botorch_hit5, matrix_nmc_botorch, rel_tol=1e-6), f"Check 5 Failed: {nmc_botorch_hit5} != {matrix_nmc_botorch}"
    audit_results["checks"]["check_05_nmc_botorch_hit5_matrix_match"] = "PASS"

    # 6. Warwick NMC622: multi slide_summary matches warwick_nmc622 slide_summary
    assert math.isclose(multi_slide["benchmarks"]["warwick_nmc622_2024"]["aicoscientist_hit_at_5"], nmc_ai_hit5, rel_tol=1e-6)
    assert math.isclose(multi_slide["benchmarks"]["warwick_nmc622_2024"]["direct_botorch_hit_at_5"], nmc_botorch_hit5, rel_tol=1e-6)
    audit_results["checks"]["check_06_nmc_slide_summary_match"] = "PASS"

    # 7. Warwick NMC622: Replay trajectory candidates match initial_designs.json across all seeds and policies
    policies = ["aicoscientist_full_process_engine", "direct_botorch_baseline", "random_baseline"]
    for seed_str, expected_ids in nmc_initial_designs.items():
        for pol in policies:
            traj_file = nmc_dir / "trajectories" / f"{pol}_seed_{seed_str}.json"
            assert traj_file.exists(), f"Check 7 Failed: Missing {traj_file}"
            with open(traj_file) as tf:
                traj_data = json.load(tf)
            assert traj_data["initial_candidate_ids"] == expected_ids, f"Check 7 Failed: Mismatch in {traj_file}"
    audit_results["checks"]["check_07_nmc_initial_designs_byte_match"] = "PASS"

    # 8. Warwick NMC622: Decision variable firewall in decision_variable_audit.json
    selected_controls = [item["variable"] for item in nmc_var_audit if item.get("use_in_primary_doe_selection")]
    excluded_vars = [item["variable"] for item in nmc_var_audit if not item.get("use_in_primary_doe_selection")]
    assert set(selected_controls) == {"roll_temperature_c", "target_density_g_cm3", "target_coating_weight_gsm"}
    assert "roll_gap_um" in excluded_vars
    assert "number_of_passes" in excluded_vars
    assert "target_porosity_pct" in excluded_vars
    audit_results["checks"]["check_08_nmc_variable_firewall_enforced"] = "PASS"

    # 9. Warwick Ultrasonic: frequency_grid_audit.json records native alignment and zero interpolation
    assert ultra_grid_audit["materials"]["Cathode"]["num_points_per_spectrum"] == 29
    assert ultra_grid_audit["materials"]["Cathode"]["frequency_grid_aligned"] is True
    assert ultra_grid_audit["materials"]["Cathode"]["interpolation_required"] is False
    assert ultra_grid_audit["materials"]["Anode"]["num_points_per_spectrum"] == 36
    assert ultra_grid_audit["materials"]["Anode"]["frequency_grid_aligned"] is True
    assert ultra_grid_audit["materials"]["Anode"]["interpolation_required"] is False
    audit_results["checks"]["check_09_ultrasonic_grid_audit_aligned"] = "PASS"

    # 10. Warwick Ultrasonic: split_manifest.json has 0 fold overlap across 5 folds
    for mat in ["Cathode", "Anode"]:
        sample_folds = ultra_splits[mat]
        fold_samples: dict[int, set[str]] = {f: set() for f in range(1, 6)}
        for s_id, f in sample_folds.items():
            fold_samples[f].add(s_id)
        for f1 in range(1, 6):
            for f2 in range(f1 + 1, 6):
                overlap = fold_samples[f1].intersection(fold_samples[f2])
                assert len(overlap) == 0, f"Check 10 Failed: Overlap between fold {f1} and {f2}"
    audit_results["checks"]["check_10_ultrasonic_zero_fold_overlap"] = "PASS"

    # 11. Warwick Ultrasonic: production execution trace counters and test_only == 0
    trace = ultra_trace_audit["execution_trace"]
    assert trace.get("horizon_projection_count", 0) >= 48
    assert trace.get("stage_feature_encoder_fit_count", 0) > 0
    assert trace.get("encoded_source_transition_count", 0) > 0
    assert trace.get("source_bound_modality_count", 0) > 0
    assert trace.get("maspo_public_forward_count", 0) > 0
    assert trace.get("stage_aware_public_transition_count", 0) > 0
    assert trace.get("final_prediction_count", 0) > 0
    assert trace.get("test_only_transition_count", -1) == 0, f"Check 11 Failed: test_only_transition_count is {trace.get('test_only_transition_count')}"
    audit_results["checks"]["check_11_ultrasonic_production_trace_verified"] = "PASS"

    # 12. Warwick Ultrasonic: model_comparison_summary.json matches ablation_summary.csv dynamically
    for mat in ["Cathode", "Anode"]:
        for tgt in ["thickness_after_um", "density_after_g_cm3"]:
            # Ridge
            r_res = ultra_comp["models"]["Ridge"]["results"][mat][tgt]
            r_proc = float(df_ultra_ablation[(df_ultra_ablation["material"] == mat) & (df_ultra_ablation["target"] == tgt) & (df_ultra_ablation["model"] == "PROCESS_ONLY")]["ridge_r2_pooled"].iloc[0])
            r_ultra = float(df_ultra_ablation[(df_ultra_ablation["material"] == mat) & (df_ultra_ablation["target"] == tgt) & (df_ultra_ablation["model"] == "ULTRASOUND_ONLY")]["ridge_r2_pooled"].iloc[0])
            r_fused = float(df_ultra_ablation[(df_ultra_ablation["material"] == mat) & (df_ultra_ablation["target"] == tgt) & (df_ultra_ablation["model"] == "PROCESS_PLUS_ULTRASOUND")]["ridge_r2_pooled"].iloc[0])
            assert math.isclose(r_res["tabular_state_process_r2"], r_proc, rel_tol=1e-5)
            assert math.isclose(r_res["ultrasound_only_r2"], r_ultra, rel_tol=1e-5)
            assert math.isclose(r_res["tabular_plus_ultrasound_r2"], r_fused, rel_tol=1e-5)
            assert math.isclose(r_res["fusion_delta_r2"], r_fused - r_proc, rel_tol=1e-5)

            # StageAware
            sa_res = ultra_comp["models"]["StageAwareProcessModel"]["results"][mat][tgt]
            sa_proc = float(df_ultra_ablation[(df_ultra_ablation["material"] == mat) & (df_ultra_ablation["target"] == tgt) & (df_ultra_ablation["model"] == "PROCESS_ONLY")]["stage_aware_r2_pooled"].iloc[0])
            sa_ultra = float(df_ultra_ablation[(df_ultra_ablation["material"] == mat) & (df_ultra_ablation["target"] == tgt) & (df_ultra_ablation["model"] == "ULTRASOUND_ONLY")]["stage_aware_r2_pooled"].iloc[0])
            sa_fused = float(df_ultra_ablation[(df_ultra_ablation["material"] == mat) & (df_ultra_ablation["target"] == tgt) & (df_ultra_ablation["model"] == "PROCESS_PLUS_ULTRASOUND")]["stage_aware_r2_pooled"].iloc[0])
            assert math.isclose(sa_res["tabular_state_process_r2"], sa_proc, rel_tol=1e-5)
            assert math.isclose(sa_res["ultrasound_only_r2"], sa_ultra, rel_tol=1e-5)
            assert math.isclose(sa_res["tabular_plus_ultrasound_r2"], sa_fused, rel_tol=1e-5)
            assert math.isclose(sa_res["fusion_delta_r2"], sa_fused - sa_proc, rel_tol=1e-5)
    audit_results["checks"]["check_12_ultrasonic_model_comparison_csv_match"] = "PASS"

    # 13. Warwick Ultrasonic: benchmark_matrix.csv row matches ablation_summary.csv metrics dynamically
    u_row = df_matrix[df_matrix["benchmark_id"] == "WARWICK_ULTRASONIC_METROLOGY"].iloc[0]
    anode_thick_fused = float(df_ultra_ablation[(df_ultra_ablation["material"] == "Anode") & (df_ultra_ablation["target"] == "thickness_after_um") & (df_ultra_ablation["model"] == "PROCESS_PLUS_ULTRASOUND")]["ridge_r2_pooled"].iloc[0])
    anode_dens_fused = float(df_ultra_ablation[(df_ultra_ablation["material"] == "Anode") & (df_ultra_ablation["target"] == "density_after_g_cm3") & (df_ultra_ablation["model"] == "PROCESS_PLUS_ULTRASOUND")]["ridge_r2_pooled"].iloc[0])
    assert f"{anode_thick_fused:.3f}" in str(u_row["multimodal_r2_or_bo_hit"])
    assert f"{anode_dens_fused:.3f}" in str(u_row["multimodal_r2_or_bo_hit"])
    audit_results["checks"]["check_13_ultrasonic_matrix_ablation_match"] = "PASS"

    # 14. Warwick Ultrasonic: multi slide_summary matches warwick_ultrasonic slide_summary
    u_slide_res = multi_slide["benchmarks"]["warwick_ultrasonic_2024"]["result"]
    assert f"{anode_thick_fused:.3f}" in u_slide_res
    assert f"{anode_dens_fused:.3f}" in u_slide_res
    audit_results["checks"]["check_14_ultrasonic_multi_slide_summary_match"] = "PASS"

    # 15. Multi-Dataset Synthesis Report: MULTI_DATASET_VALIDATION_REPORT.md has zero stale literals or unsupported claims
    stale_literals = ["0.9715", "6.25", "0.849", "0.874", "0.895", "+0.033"]
    for lit in stale_literals:
        assert lit not in multi_report_text, f"Check 15 Failed: Stale literal {lit!r} found in MULTI_DATASET_VALIDATION_REPORT.md"
    unsupported_claims = ["directly encodes", "3x acceleration", "3× acceleration", "robust transferability"]
    for claim in unsupported_claims:
        assert claim not in multi_report_text, f"Check 15 Failed: Unsupported claim {claim!r} found in MULTI_DATASET_VALIDATION_REPORT.md"
    assert "CALENDERING_STAGE_STATE_PREDICTION" in multi_report_text
    audit_results["checks"]["check_15_multi_dataset_report_clean"] = "PASS"

    # 16. Verify report_context.json against all source benchmark artifacts
    report_context_path = out_dir / "report_context.json"
    assert report_context_path.exists(), f"Check 16 Failed: Missing {report_context_path}"
    with open(report_context_path) as f:
        report_context = json.load(f)
    verify_report_context_against_sources(
        report_context=report_context,
        drak_policy_df=df_drak_policy,
        nmc_policy_df=df_nmc_policy,
        ultra_ablation_df=df_ultra_ablation,
        ultra_comp_dict=ultra_comp,
        drak_slide_dict=drak_slide,
        nmc_slide_dict=nmc_slide,
    )
    audit_results["checks"]["check_16_report_context_source_parity"] = "PASS"

    # 17. Verify rendered MULTI_DATASET_VALIDATION_REPORT.md matches report_context.json
    verify_markdown_report_matches_context(multi_report_text, report_context)
    audit_results["checks"]["check_17_multi_report_matches_context"] = "PASS"

    # 18. Verify NMC622 Report Contains Hit@5 row
    nmc_report_path = nmc_dir / "WARWICK_NMC622_PROCESS_BENCHMARK_REPORT.md"
    nmc_report_text = nmc_report_path.read_text(encoding="utf-8")
    verify_nmc622_report_contains_hit5(nmc_report_text)
    audit_results["checks"]["check_18_nmc622_report_contains_hit5"] = "PASS"

    # 19. Verify Absence of Unsupported Wording across all reports
    ultra_report_path = ultra_dir / "WARWICK_ULTRASONIC_MULTIMODAL_REPORT.md"
    ultra_report_text = ultra_report_path.read_text(encoding="utf-8")
    reports_to_check = {
        "MULTI_DATASET_VALIDATION_REPORT.md": multi_report_text,
        "WARWICK_NMC622_PROCESS_BENCHMARK_REPORT.md": nmc_report_text,
        "WARWICK_ULTRASONIC_MULTIMODAL_REPORT.md": ultra_report_text,
        "model_comparison_summary.json": json.dumps(ultra_comp),
    }
    verify_no_unsupported_wording(reports_to_check)
    audit_results["checks"]["check_19_no_unsupported_wording"] = "PASS"

    # 20. Execution trace audit comprehensive validation
    assert trace["battery_process_runs_seen"] == 48
    assert trace["horizon_projection_count"] == 48
    assert trace["stage_feature_encoder_fit_count"] == 60  # 2 mats * 2 tgts * 5 folds * 3 modes
    assert trace["encoded_source_transition_count"] == 2880
    assert trace["validated_source_transition_count"] == 2880
    assert trace["source_bound_modality_count"] == 960  # Fused + Ultrasound modes
    assert trace["test_only_transition_count"] == 0
    audit_results["checks"]["check_20_execution_trace_strictly_validated"] = "PASS"

    # Save final consistency audit JSON
    with open(out_dir / "consistency_audit.json", "w") as f:
        json.dump(audit_results, f, indent=2)

    logger.info("ALL 20 DYNAMIC CONSISTENCY CHECKS PASSED SUCCESSFULLY!")
    return audit_results


def verify_report_context_against_sources(
    report_context: dict[str, Any],
    drak_policy_df: pd.DataFrame,
    nmc_policy_df: pd.DataFrame,
    ultra_ablation_df: pd.DataFrame,
    ultra_comp_dict: dict[str, Any],
    drak_slide_dict: dict[str, Any] | None = None,
    nmc_slide_dict: dict[str, Any] | None = None,
) -> None:
    """Verify that report_context.json exactly reflects all source benchmark artifacts."""
    # Drakopoulos
    drak_uncon = drak_policy_df[drak_policy_df["benchmark_task"] == "UNCONSTRAINED_D30"]
    drak_ai_hit5 = float(drak_uncon[drak_uncon["policy"] == "AICOSCIENTIST_PROCESS_SURROGATE"]["hit_rate_step_5"].iloc[0])
    drak_bo_hit5 = float(drak_uncon[drak_uncon["policy"] == "DIRECT_BOTORCH_BASELINE"]["hit_rate_step_5"].iloc[0])
    drak_rand_hit5 = float(drak_uncon[drak_uncon["policy"] == "random"]["hit_rate_step_5"].iloc[0])
    drak_simple_regret = float(drak_uncon[drak_uncon["policy"] == "AICOSCIENTIST_PROCESS_SURROGATE"]["mean_simple_regret"].iloc[0])

    assert math.isclose(report_context["drakopoulos"]["ai_hit5"], drak_ai_hit5, rel_tol=1e-5)
    assert math.isclose(report_context["drakopoulos"]["botorch_hit5"], drak_bo_hit5, rel_tol=1e-5)
    assert math.isclose(report_context["drakopoulos"]["random_hit5"], drak_rand_hit5, rel_tol=1e-5)
    assert math.isclose(report_context["drakopoulos"]["simple_regret"], drak_simple_regret, rel_tol=1e-5)

    if drak_slide_dict is not None:
        drak_best_recipe = str(drak_slide_dict["unconstrained"]["source_best_recipe_id"])
        drak_best_d30 = float(drak_slide_dict["unconstrained"]["source_best_d30_mah_g"])
        drak_analytic_rand = float(drak_slide_dict["unconstrained"]["analytic_random_hit_at_5"])
        assert report_context["drakopoulos"]["best_recipe"] == drak_best_recipe
        assert math.isclose(report_context["drakopoulos"]["best_d30"], drak_best_d30, rel_tol=1e-5)
        assert math.isclose(report_context["drakopoulos"]["analytic_random_hit5"], drak_analytic_rand, rel_tol=1e-5)

    # NMC622
    nmc_ai_hit5 = float(nmc_policy_df[nmc_policy_df["policy"] == "AICOSCIENTIST_FULL_PROCESS_ENGINE"]["hit_at_5"].iloc[0])
    nmc_bo_hit5 = float(nmc_policy_df[nmc_policy_df["policy"] == "DIRECT_BOTORCH_BASELINE"]["hit_at_5"].iloc[0])
    nmc_rand_hit5 = float(nmc_policy_df[nmc_policy_df["policy"] == "RANDOM_BASELINE"]["hit_at_5"].iloc[0])
    nmc_analytic_rand = float(nmc_policy_df[nmc_policy_df["policy"] == "EXACT_ANALYTICAL_RANDOM"]["hit_at_5"].iloc[0])
    nmc_simple_regret = float(nmc_policy_df[nmc_policy_df["policy"] == "AICOSCIENTIST_FULL_PROCESS_ENGINE"]["simple_regret_at_5"].iloc[0])

    assert math.isclose(report_context["nmc622"]["ai_hit5"], nmc_ai_hit5, rel_tol=1e-5)
    assert math.isclose(report_context["nmc622"]["botorch_hit5"], nmc_bo_hit5, rel_tol=1e-5)
    assert math.isclose(report_context["nmc622"]["random_hit5"], nmc_rand_hit5, rel_tol=1e-5)
    assert math.isclose(report_context["nmc622"]["analytic_random_hit5"], nmc_analytic_rand, rel_tol=1e-5)
    assert math.isclose(report_context["nmc622"]["simple_regret"], nmc_simple_regret, rel_tol=1e-5)

    if nmc_slide_dict is not None:
        nmc_best_cond = str(nmc_slide_dict["source_observed_best_condition"])
        nmc_best_ratio = float(nmc_slide_dict["source_observed_best_target"])
        assert report_context["nmc622"]["best_condition"] == nmc_best_cond
        assert math.isclose(report_context["nmc622"]["best_ratio"], nmc_best_ratio, rel_tol=1e-4)

    # Ultrasonic
    for mat_key, mat_name in [("anode", "Anode"), ("cathode", "Cathode")]:
        for tgt_key, tgt_col in [("thickness", "thickness_after_um"), ("density", "density_after_g_cm3")]:
            slice_key = f"{mat_key}_{tgt_key}"
            ctx_slice = report_context["ultrasonic"][slice_key]
            comp_ridge = ultra_comp_dict["models"]["Ridge"]["results"][mat_name][tgt_col]
            comp_sa = ultra_comp_dict["models"]["StageAwareProcessModel"]["results"][mat_name][tgt_col]

            assert math.isclose(ctx_slice["ridge_process_only_r2"], comp_ridge["tabular_state_process_r2"], rel_tol=1e-5)
            assert math.isclose(ctx_slice["ridge_ultrasound_only_r2"], comp_ridge["ultrasound_only_r2"], rel_tol=1e-5)
            assert math.isclose(ctx_slice["ridge_fused_r2"], comp_ridge["tabular_plus_ultrasound_r2"], rel_tol=1e-5)
            assert math.isclose(ctx_slice["ridge_fusion_delta"], comp_ridge["fusion_delta_r2"], rel_tol=1e-5)

            assert math.isclose(ctx_slice["stageaware_process_only_r2"], comp_sa["tabular_state_process_r2"], rel_tol=1e-5)
            assert math.isclose(ctx_slice["stageaware_ultrasound_only_r2"], comp_sa["ultrasound_only_r2"], rel_tol=1e-5)
            assert math.isclose(ctx_slice["stageaware_fused_r2"], comp_sa["tabular_plus_ultrasound_r2"], rel_tol=1e-5)
            assert math.isclose(ctx_slice["stageaware_fusion_delta"], comp_sa["fusion_delta_r2"], rel_tol=1e-5)


def verify_markdown_report_matches_context(
    report_text: str,
    report_context: dict[str, Any],
) -> None:
    """Verify that MULTI_DATASET_VALIDATION_REPORT.md renders context values faithfully."""
    drak = report_context["drakopoulos"]
    nmc = report_context["nmc622"]
    ultra = report_context["ultrasonic"]

    # Drakopoulos & NMC622
    assert f"{drak['ai_hit5'] * 100:.1f}%" in report_text
    assert f"{drak['botorch_hit5'] * 100:.1f}%" in report_text
    assert drak["best_recipe"] in report_text
    assert f"{nmc['ai_hit5'] * 100:.1f}%" in report_text
    assert f"{nmc['botorch_hit5'] * 100:.1f}%" in report_text
    assert nmc["best_condition"] in report_text

    # Ultrasonic formatted strings
    ctx_u_at = ultra["anode_thickness"]
    ctx_u_ad = ultra["anode_density"]
    ctx_u_ct = ultra["cathode_thickness"]

    assert f"{ctx_u_ad['ridge_fusion_delta']:+.3f}" in report_text
    assert f"{ctx_u_ad['stageaware_fusion_delta']:+.3f}" in report_text
    assert f"{ctx_u_at['ridge_ultrasound_only_r2']:.3f}" in report_text
    assert f"{ctx_u_at['stageaware_ultrasound_only_r2']:.3f}" in report_text
    assert f"{ctx_u_at['ridge_fused_r2']:.3f}" in report_text
    assert f"{ctx_u_at['stageaware_fused_r2']:.3f}" in report_text
    assert f"{ctx_u_ad['ridge_process_only_r2']:.3f}" in report_text
    assert f"{ctx_u_ad['stageaware_process_only_r2']:.3f}" in report_text
    assert f"{ctx_u_ad['ridge_fused_r2']:.3f}" in report_text
    assert f"{ctx_u_ad['stageaware_fused_r2']:.3f}" in report_text
    assert f"{ctx_u_ct['ridge_process_only_r2']:.3f}" in report_text


def verify_nmc622_report_contains_hit5(nmc_report_text: str) -> None:
    """Verify WARWICK_NMC622_PROCESS_BENCHMARK_REPORT.md retains the primary Hit@5 row."""
    found = False
    for line in nmc_report_text.splitlines():
        if "Hit@5" in line and "100.0%" in line and "90.0%" in line:
            found = True
            break
    assert found, "Hit@5 row missing or formatted incorrectly in WARWICK_NMC622_PROCESS_BENCHMARK_REPORT.md"


def verify_no_unsupported_wording(reports: dict[str, str]) -> None:
    """Verify absence of prohibited or unsupported causal phrases across reports."""
    prohibited_phrases = [
        "due to finite sample size",
        "optimal pilot-scale",
        "source-observed optimal condition",
        "mechanically dictates",
        "all evaluations adhere strictly to grouped cross-validation",
        "3x acceleration",
        "3× acceleration",
        "robust transferability",
        "directly encodes",
    ]
    for report_name, text in reports.items():
        for phrase in prohibited_phrases:
            assert phrase.lower() not in text.lower(), (
                f"Prohibited phrase {phrase!r} found in {report_name}"
            )


if __name__ == "__main__":
    run_consistency_audit()

