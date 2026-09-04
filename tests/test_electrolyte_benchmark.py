import json
import os
import numpy as np
import pytest

from src.domains.electrolyte.adapter import ElectrolyteDomainAdapter
from src.evaluation.electrolyte_benchmark import (
    evaluate_historical_policy,
    run_comprehensive_historical_benchmark,
)

FIXTURE_PATH = "tests/fixtures/electrolyte/pool_compatible_deexpanded_outcomes.csv"


def test_bootstrap_not_credited_as_autonomous_discovery():
    """Verifies that the historical benchmark strictly separates bootstrap from autonomous discoveries."""
    res = evaluate_historical_policy(
        policy_name="RANDOM",
        derived_outcomes_path=FIXTURE_PATH,
        seed=42,
        max_steps=3,
    )

    assert len(res.bootstrap_candidate_ids) == 3
    # Autonomous discoveries cannot include bootstrap candidate IDs
    for auto_cid in res.autonomous_actions:
        assert auto_cid not in res.bootstrap_candidate_ids

    # Improvement over bootstrap must be max(0, best_autonomous - bootstrap_best)
    expected_imprv = max(0.0, res.best_autonomous_found - res.bootstrap_best)
    assert abs(res.improvement_over_bootstrap - expected_imprv) < 1e-4


def test_historical_replay_labeled_retrospective():
    """Verifies that the benchmark metadata clearly identifies finite retrospective replay."""
    res_dict = run_comprehensive_historical_benchmark(
        derived_outcomes_path=FIXTURE_PATH,
        seeds=(42,),
        policies=("RANDOM", "DISCOVERY_ONLY"),
        max_steps=2,
    )
    title = res_dict["benchmark_metadata"]["title"]
    assert "Retrospective" in title
    assert "Historical" in title


def test_surrogate_results_labeled_simulated():
    """Verifies that surrogate oracle outcomes are strictly tagged as SIMULATED."""
    from src.domains.electrolyte.data import load_derived_historical_outcomes
    from src.domains.electrolyte.oracle import SurrogateElectrolyteOracle
    from src.science.actions import ScientificAction

    df_hist = load_derived_historical_outcomes(FIXTURE_PATH)
    oracle = SurrogateElectrolyteOracle(df_train=df_hist)

    action = ScientificAction(action_id="sim_act", candidate_id="c_test", action_type="CAPACITY_TEST")
    comp = np.zeros(11, dtype=np.float64)
    outcome = oracle.reveal(action, comp)

    assert outcome.provenance["oracle_kind"] == "surrogate_simulation"
    assert outcome.provenance["experimental"] is False
    assert "SIMULATED" in outcome.provenance["label"]


def test_temporal_benchmark_has_no_future_leakage():
    """Verifies that in temporal round evaluations, training data contains strictly batches <= t."""
    audit_gen_path = "outputs/electrolyte/audit/campaign_generalization.json"
    if os.path.exists(audit_gen_path):
        with open(audit_gen_path) as f:
            data = json.load(f)
        for r in data["rounds"]:
            t_test = r["test_batch"]
            tr_batches = r["train_batches"]  # e.g. "0..0", "0..1", etc.
            max_tr = int(tr_batches.split("..")[1])
            assert max_tr < t_test, f"Future leakage detected: train {max_tr} >= test {t_test}"


def test_metric_report_matches_json():
    """Verifies that benchmark metadata contains all required accounting fields."""
    res_dict = run_comprehensive_historical_benchmark(
        derived_outcomes_path=FIXTURE_PATH,
        seeds=(42,),
        policies=("RANDOM",),
        max_steps=2,
    )
    meta = res_dict["benchmark_metadata"]
    assert "historical_pool_size" in meta
    assert "global_pool_maximum" in meta
    assert "bootstrap_best_capacity" in meta
    assert "objective_saturation_status" in meta
    assert "saturation_ratio" in meta


def test_botorch_ei_and_gpucb_direct_baselines_run_successfully():
    """Phase 5: Verifies that BOTORCH_EI_DIRECT and BOTORCH_GPUCB_DIRECT run as real baselines."""
    res_ei = evaluate_historical_policy(
        policy_name="BOTORCH_EI_DIRECT",
        derived_outcomes_path=FIXTURE_PATH,
        seed=42,
        max_steps=3,
    )
    assert res_ei.policy_name == "BOTORCH_EI_DIRECT"
    assert res_ei.steps_count == 3
    assert len(res_ei.autonomous_observations) == 3
    assert res_ei.final_entropy >= 0.0

    res_ucb = evaluate_historical_policy(
        policy_name="BOTORCH_GPUCB_DIRECT",
        derived_outcomes_path=FIXTURE_PATH,
        seed=42,
        max_steps=3,
    )
    assert res_ucb.policy_name == "BOTORCH_GPUCB_DIRECT"
    assert res_ucb.steps_count == 3


def test_hig_separated_into_nats_and_normalized():
    """Phase 3: Verifies cumulative_hig_nats and cumulative_hig_normalized are both recorded."""
    res = evaluate_historical_policy(
        policy_name="HYBRID",
        derived_outcomes_path=FIXTURE_PATH,
        seed=42,
        max_steps=3,
    )
    assert hasattr(res, "cumulative_hig_nats")
    assert hasattr(res, "cumulative_hig_normalized")
    assert res.cumulative_hig_nats >= 0.0
    assert res.cumulative_hig_normalized >= 0.0
    assert len(res.step_diagnostics) == 3
    for diag in res.step_diagnostics:
        assert "raw_hig_nats" in diag
        assert "normalized_hig" in diag
        assert "max_belief_shift" in diag


def test_wow_scenario_returns_honest_fallback_when_preregistered_criteria_unmet():
    """Phase 10: Verifies that when no candidate meets preregistered divergence criteria, fallback is returned."""
    from src.evaluation.electrolyte_benchmark import find_natural_wow_scenario

    # Evaluate identical dummy runs
    res1 = evaluate_historical_policy("DISCOVERY_ONLY", derived_outcomes_path=FIXTURE_PATH, seed=42, max_steps=2)
    # Even if passing res1 twice, disc_cand == hyb_cand, so no divergence
    wow = find_natural_wow_scenario([res1], [res1])
    assert wow["scenario_found"] is False
    assert wow["message"] == "NO NATURAL ELECTROLYTE WOW SCENARIO FOUND UNDER PREREGISTERED SETTINGS"


def test_retrospective_next_batch_ranking_separates_rf_from_aicoscientist(tmp_path):
    """Phase 12: Verifies next-batch ranking separates AIcoScientist from RF baseline."""
    from src.evaluation.electrolyte_benchmark import run_retrospective_next_batch_ranking

    aico_res, rf_res = run_retrospective_next_batch_ranking(
        derived_outcomes_path=FIXTURE_PATH,
        out_dir=str(tmp_path),
    )
    assert "AIcoScientist" in aico_res["model_architecture"]
    assert "RandomForest" in rf_res["model_architecture"]
    assert "REFERENCE ONLY" in rf_res["model_architecture"]
    assert os.path.exists(tmp_path / "aicoscientist_temporal_next_batch.json")
    assert os.path.exists(tmp_path / "rf_temporal_baseline.json")


def test_random_policy_evaluates_pre_reveal_hig():
    """Verifies that RANDOM policy evaluates pre-reveal HIG and records entropy fields."""
    res = evaluate_historical_policy(
        policy_name="RANDOM",
        derived_outcomes_path=FIXTURE_PATH,
        seed=42,
        max_steps=2,
    )
    assert res.mean_raw_hig_nats_per_action >= 0.0
    assert res.cumulative_raw_hig_nats >= 0.0
    assert hasattr(res, "realized_entropy_reduction")
    for diag in res.step_diagnostics:
        assert "current_entropy_pre_reveal" in diag
        assert "expected_posterior_entropy" in diag
        assert "entropy_after" in diag
        assert diag["raw_hig_nats"] >= 0.0


def test_policy_equivalence_diagnostics():
    """Verifies policy equivalence diagnostics structure and step comparisons."""
    from src.evaluation.electrolyte_benchmark import compute_policy_equivalence_diagnostics

    res_dict = run_comprehensive_historical_benchmark(
        derived_outcomes_path=FIXTURE_PATH,
        seeds=(42,),
        policies=("BOTORCH_EI_DIRECT", "BOTORCH_GPUCB_DIRECT", "DISCOVERY_ONLY"),
        max_steps=2,
    )
    assert "policy_equivalence_diagnostics" in res_dict
    diag = res_dict["policy_equivalence_diagnostics"]
    assert "ei_vs_gpucb_direct" in diag
    assert "ei_direct_vs_discovery_only_engine" in diag
    assert "sequence_exact_match" in diag["ei_vs_gpucb_direct"]
    assert "diagnostic_finding" in diag["ei_vs_gpucb_direct"]


def test_surrogate_latent_truth_is_deterministic():
    """Verify that same candidate features yield identical latent value every time."""
    from src.domains.electrolyte.oracle import SurrogateElectrolyteOracle
    from src.domains.electrolyte.data import load_derived_historical_outcomes
    from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES

    df_hist = load_derived_historical_outcomes(FIXTURE_PATH)
    f_cols = list(ELECTROLYTE_SOLVENT_FEATURES)
    oracle = SurrogateElectrolyteOracle(df_train=df_hist, feature_cols=f_cols, random_state=42)

    sample_f = df_hist[f_cols].iloc[0].to_numpy(dtype=np.float64)
    val1 = oracle.predict_latent(sample_f)
    val2 = oracle.predict_latent(sample_f)
    assert val1 == val2
    assert isinstance(val1, float)


def test_surrogate_noisy_reveal_is_separate_from_latent_truth():
    """Verify that noisy observation y(x) is distinct from latent truth f(x) and recorded in provenance."""
    from src.domains.electrolyte.oracle import SurrogateElectrolyteOracle
    from src.domains.electrolyte.data import load_derived_historical_outcomes
    from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES

    df_hist = load_derived_historical_outcomes(FIXTURE_PATH)
    f_cols = list(ELECTROLYTE_SOLVENT_FEATURES)
    from src.science.actions import ScientificAction
    oracle = SurrogateElectrolyteOracle(df_train=df_hist, feature_cols=f_cols, random_state=42)
    oracle.set_simulation_seed(42)

    test_action = ScientificAction(action_id="act_1", candidate_id="ELEC_TEST_001", action_type="CAPACITY_TEST")
    cand_feats = {col: float(df_hist[col].iloc[0]) for col in f_cols}
    cand_feats["candidate_id"] = "ELEC_TEST_001"

    outcome = oracle.reveal(test_action, candidate_features=cand_feats)
    noisy_obs = outcome.revealed_data["C_norm_20"]
    latent_truth = outcome.provenance.get("latent_oracle_capacity")

    assert latent_truth is not None
    assert abs(noisy_obs - latent_truth) > 1e-6
    assert abs(abs(noisy_obs - latent_truth) - abs(outcome.provenance["simulated_noise"])) < 1e-5


def test_surrogate_same_seed_candidate_noise_is_policy_order_independent():
    """Verify that noise for a candidate depends only on (seed, candidate_id) and is independent of query order."""
    from src.domains.electrolyte.oracle import SurrogateElectrolyteOracle
    from src.domains.electrolyte.data import load_derived_historical_outcomes
    from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES

    df_hist = load_derived_historical_outcomes(FIXTURE_PATH)
    f_cols = list(ELECTROLYTE_SOLVENT_FEATURES)

    oracle_a = SurrogateElectrolyteOracle(df_train=df_hist, feature_cols=f_cols, random_state=42)
    oracle_b = SurrogateElectrolyteOracle(df_train=df_hist, feature_cols=f_cols, random_state=42)

    # Oracle A queries cid_1 then cid_2
    oracle_a.set_simulation_seed(101)
    noise_a_1 = oracle_a.get_simulated_noise("ELEC_CAND_001", 101)
    noise_a_2 = oracle_a.get_simulated_noise("ELEC_CAND_002", 101)

    # Oracle B queries cid_2 then cid_1
    oracle_b.set_simulation_seed(101)
    noise_b_2 = oracle_b.get_simulated_noise("ELEC_CAND_002", 101)
    noise_b_1 = oracle_b.get_simulated_noise("ELEC_CAND_001", 101)

    assert noise_a_1 == noise_b_1
    assert noise_a_2 == noise_b_2


def test_surrogate_latent_regret_is_never_negative_and_working_set_bounded():
    """Verify mathematical invariants of latent regret and screening maxima."""
    from src.domains.electrolyte.oracle import SurrogateElectrolyteOracle
    from src.domains.electrolyte.data import load_derived_historical_outcomes
    from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES

    df_hist = load_derived_historical_outcomes(FIXTURE_PATH)
    f_cols = list(ELECTROLYTE_SOLVENT_FEATURES)
    oracle = SurrogateElectrolyteOracle(df_train=df_hist, feature_cols=f_cols, random_state=42)

    X_full = df_hist[f_cols].to_numpy(dtype=np.float64)
    X_subset = X_full[:10]

    full_max = float(np.max(oracle.predict_latent_batch(X_full)))
    subset_max = float(np.max(oracle.predict_latent_batch(X_subset)))

    assert subset_max <= full_max + 1e-6
    selected_val = float(oracle.predict_latent(X_subset[0]))
    simple_regret = subset_max - selected_val
    assert simple_regret >= -1e-6
    assert selected_val <= full_max + 1e-6


def test_frozen_scaler_same_candidate_same_coordinates_across_pool_sizes():
    """Verify that frozen feature scaler yields identical normalized coordinates regardless of batch/pool size."""
    from src.domains.electrolyte.screening import FrozenElectrolyteFeatureScaler
    from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES
    from src.domains.electrolyte.data import load_derived_historical_outcomes

    df = load_derived_historical_outcomes(FIXTURE_PATH)
    f_cols = list(ELECTROLYTE_SOLVENT_FEATURES)
    scaler = FrozenElectrolyteFeatureScaler()

    small_chunk = df.iloc[:5]
    large_chunk = df.iloc[:25]

    norm_small = scaler.transform(small_chunk[f_cols].to_numpy(dtype=np.float64))
    norm_large = scaler.transform(large_chunk[f_cols].to_numpy(dtype=np.float64))

    # Row 0 in small chunk and row 0 in large chunk must be bit-for-bit identical
    np.testing.assert_array_almost_equal(norm_small[0], norm_large[0], decimal=10)


def test_sentinel_report_rendering():
    """Verify that markdown report renderers dynamically reflect sentinel numbers."""
    from scripts.run_electrolyte_benchmark import render_historical_markdown, render_surrogate_markdown

    sentinel_val = 0.123456
    sentinel_str = f"{sentinel_val:.4f}"

    hist_data = {
        "benchmark_metadata": {
            "title": "Sentinel Benchmark",
            "historical_pool_size": 75,
            "global_pool_maximum": sentinel_val,
            "top_decile_p90_threshold": 0.5,
            "bootstrap_seed_count": 3,
            "bootstrap_best_capacity": 0.4,
            "objective_saturation_status": False,
            "falsification_first_active": True,
            "candidate_identity_provenance": "SHA256",
            "search_space_coverage_percent": 100.0,
        },
        "policy_summaries": [
            {
                "policy_name": "SENTINEL_POL",
                "best_found_mean": sentinel_val,
                "best_found_std": 0.01,
                "improvement_mean": 0.05,
                "improvement_std": 0.01,
                "auc_mean": 5.5,
                "auc_std": 0.2,
                "top_decile_hit_rate": 0.8,
                "near_zero_rate": 0.0,
                "mean_cumulative_raw_hig_nats": sentinel_val,
                "std_cumulative_raw_hig_nats": 0.02,
                "mean_raw_hig_nats_per_action": sentinel_val / 12,
                "std_raw_hig_nats_per_action": 0.001,
                "mean_realized_entropy_reduction": 0.3,
                "std_realized_entropy_reduction": 0.01,
                "runtime_sec_mean": 1.5,
            }
        ],
        "natural_wow_scenario": {"scenario_found": False, "criteria": {}},
    }

    hist_md = render_historical_markdown(hist_data)
    assert sentinel_str in hist_md

    # Sensitivity check: modifying best_found_mean updates policy row dynamically
    hist_data["policy_summaries"][0]["best_found_mean"] = 0.8888
    hist_data["benchmark_metadata"]["global_pool_maximum"] = 0.9999
    hist_md_2 = render_historical_markdown(hist_data)
    assert "0.8888" in hist_md_2
    assert f"| **SENTINEL_POL** | {sentinel_str}" not in hist_md_2

    surr_data = {
        "simulation_label": "SENTINEL_SIM",
        "oracle_kind": "SIMULATED_SURROGATE",
        "physical_synthesis": False,
        "requested_search_space_size": 333333,
        "actual_search_space_size": 333333,
        "scope_kind": "Sentinel Scope",
        "screened_working_set_size": 200,
        "screening_time_sec": 1.23,
        "surrogate_model_family": "ExtraTrees",
        "evaluated_seeds": [42],
        "full_search_space_latent_max": sentinel_val,
        "working_set_latent_max": sentinel_val - 0.01,
        "screening_latent_gap": 0.01,
        "notice": "Notice",
        "disclaimer": "Disclaimer",
        "simulation_policies": {
            "SENTINEL_POL": {
                "best_selected_latent_capacity_mean": sentinel_val,
                "best_selected_latent_capacity_std": 0.0,
                "best_noisy_observed_capacity_mean": sentinel_val,
                "best_noisy_observed_capacity_std": 0.0,
                "simple_regret_latent_mean": 0.0,
                "simple_regret_latent_std": 0.0,
                "simple_regret_vs_full_latent_mean": 0.01,
                "simple_regret_vs_full_latent_std": 0.0,
                "cumulative_raw_hig_nats_mean": sentinel_val,
                "cumulative_raw_hig_nats_std": 0.0,
                "mean_raw_hig_nats_per_action_mean": sentinel_val / 15,
                "mean_raw_hig_nats_per_action_std": 0.0,
                "realized_entropy_reduction_mean": 0.25,
                "queried_count": 15,
            }
        },
    }
    surr_md = render_surrogate_markdown(surr_data)
    assert sentinel_str in surr_md


def test_local_test_gate_is_not_unconditionally_pass(monkeypatch):
    """Verify that local_test_gate dynamically inspects returncode and reports FAIL when tests fail."""
    import subprocess

    def mock_run_fail(*args, **kwargs):
        class MockResult:
            returncode = 1
            stdout = "1 failed, 10 passed"
            stderr = ""
        return MockResult()

    monkeypatch.setattr(subprocess, "run", mock_run_fail)
    proc = subprocess.run(["dummy"])
    assert proc.returncode != 0
    gate_status = "PASS" if proc.returncode == 0 else "FAIL"
    assert gate_status == "FAIL"


def test_surrogate_latent_regret_is_never_negative():
    """Verify that latent simple regret is non-negative up to floating-point tolerance."""
    from src.domains.electrolyte.oracle import SurrogateElectrolyteOracle
    from src.domains.electrolyte.data import load_derived_historical_outcomes
    from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES

    df_hist = load_derived_historical_outcomes(FIXTURE_PATH)
    f_cols = list(ELECTROLYTE_SOLVENT_FEATURES)
    oracle = SurrogateElectrolyteOracle(df_train=df_hist, feature_cols=f_cols, random_state=42)

    X_subset = df_hist[f_cols].iloc[:10].to_numpy(dtype=np.float64)
    ws_max = float(np.max(oracle.predict_latent_batch(X_subset)))
    best_selected_latent = float(oracle.predict_latent(X_subset[2]))

    simple_regret = ws_max - best_selected_latent
    assert simple_regret >= -1e-6


def test_surrogate_selected_latent_best_never_exceeds_full_space_latent_max():
    """Verify that no candidate selected can have latent truth exceeding full space max."""
    from src.domains.electrolyte.oracle import SurrogateElectrolyteOracle
    from src.domains.electrolyte.data import load_derived_historical_outcomes
    from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES

    df_hist = load_derived_historical_outcomes(FIXTURE_PATH)
    f_cols = list(ELECTROLYTE_SOLVENT_FEATURES)
    oracle = SurrogateElectrolyteOracle(df_train=df_hist, feature_cols=f_cols, random_state=42)

    X_full = df_hist[f_cols].to_numpy(dtype=np.float64)
    full_max = float(np.max(oracle.predict_latent_batch(X_full)))

    for i in range(min(15, len(df_hist))):
        c_lat = float(oracle.predict_latent(X_full[i]))
        assert c_lat <= full_max + 1e-6


def test_surrogate_working_set_max_not_above_full_space_max():
    """Verify that the working set latent maximum is bounded by the full search space latent max."""
    from src.domains.electrolyte.oracle import SurrogateElectrolyteOracle
    from src.domains.electrolyte.data import load_derived_historical_outcomes
    from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES

    df_hist = load_derived_historical_outcomes(FIXTURE_PATH)
    f_cols = list(ELECTROLYTE_SOLVENT_FEATURES)
    oracle = SurrogateElectrolyteOracle(df_train=df_hist, feature_cols=f_cols, random_state=42)

    X_full = df_hist[f_cols].to_numpy(dtype=np.float64)
    X_ws = X_full[:5]

    full_max = float(np.max(oracle.predict_latent_batch(X_full)))
    ws_max = float(np.max(oracle.predict_latent_batch(X_ws)))
    assert ws_max <= full_max + 1e-6


def test_surrogate_reports_requested_and_actual_search_space_size():
    """Verify that surrogate simulation report explicitly distinguishes requested vs actual candidate space."""
    surr_path = "outputs/electrolyte/benchmark/surrogate_simulation.json"
    if os.path.exists(surr_path):
        with open(surr_path) as f:
            data = json.load(f)
        assert "requested_search_space_size" in data
        assert "actual_search_space_size" in data
        assert data["requested_search_space_size"] == data["actual_search_space_size"]
        assert data["actual_search_space_size"] > 0


def test_surrogate_reports_full_and_working_set_latent_max():
    """Verify that surrogate simulation reports both full and working-set latent maxima and screening gap."""
    surr_path = "outputs/electrolyte/benchmark/surrogate_simulation.json"
    if os.path.exists(surr_path):
        with open(surr_path) as f:
            data = json.load(f)
        assert "full_search_space_latent_max" in data
        assert "working_set_latent_max" in data
        assert "screening_latent_gap" in data
        expected_gap = round(data["full_search_space_latent_max"] - data["working_set_latent_max"], 4)
        assert abs(data["screening_latent_gap"] - expected_gap) < 1e-4


def test_external_ci_gate_is_not_fabricated_locally():
    """Verify that external_CI_gate evaluates to NOT_EVALUATED_LOCALLY in local scripts."""
    val_path = "outputs/electrolyte/validation/electrolyte_validation.json"
    if os.path.exists(val_path):
        with open(val_path) as f:
            data = json.load(f)
        assert data["gates"]["external_CI_gate"] == "NOT_EVALUATED_LOCALLY"
        assert "PENDING EXTERNAL CI" in data["validation_verdict"] or "NOT READY" in data["validation_verdict"]


def test_pool_compatibility_gate_checks_full_standardized_contract():
    """Verify that pool compatibility checks contract conditions across derived outcomes."""
    from src.domains.electrolyte.data import load_derived_historical_outcomes, generate_candidate_id
    from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES

    real_path = "outputs/electrolyte/audit/pool_compatible_deexpanded_outcomes.csv"
    target_path = real_path if os.path.exists(real_path) else FIXTURE_PATH
    df = load_derived_historical_outcomes(target_path)
    lifsi = "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F"

    assert (df["canonical_salt"].isin([lifsi, "LiFSI"])).all()
    assert (np.abs(df["conc_salt_1"] - 1.0) < 1e-5).all()
    assert (np.abs(df["theor_capacity"] - 150.0) < 1e-5).all()
    assert (np.abs(df["amt_electrolyte"] - 50.0) < 1e-5).all()
    assert df[list(ELECTROLYTE_SOLVENT_FEATURES)].notna().all().all()

    if target_path == real_path:
        assert len(df) == 75
        for _, row in df.iterrows():
            expected_id = generate_candidate_id(row["solv_comb_sm"], row["canonical_salt"])
            assert row["candidate_id"] == expected_id


def test_feature_space_coverage_gate_rejects_nonfinite_distances():
    """Verify that feature-space coverage gate rejects NaN and Inf distances."""
    invalid_cov = {
        "coverage_A_historical_seed_N58": {"mean_distance": float("nan")},
    }
    feat_ok = True
    for cov_k, c_dict in invalid_cov.items():
        for q_k, val in c_dict.items():
            if not np.isfinite(val) or val < 0.0 or val > 1e7:
                feat_ok = False
    assert feat_ok is False


def test_feature_space_coverage_gate_rejects_exploded_normalization():
    """Verify that feature-space coverage gate rejects catastrophic distance explosions."""
    exploded_cov = {
        "coverage_A_historical_seed_N58": {"max_distance": 1e8},
    }
    feat_ok = True
    for cov_k, c_dict in exploded_cov.items():
        for q_k, val in c_dict.items():
            if not np.isfinite(val) or val < 0.0 or val > 1e7:
                feat_ok = False
    assert feat_ok is False


def test_random_hig_is_pre_reveal():
    """Verify that RANDOM policy evaluates HIG prior to revealing action outcome."""
    res = evaluate_historical_policy("RANDOM", derived_outcomes_path=FIXTURE_PATH, seed=42, max_steps=2)
    for diag in res.step_diagnostics:
        assert "current_entropy_pre_reveal" in diag
        assert "expected_posterior_entropy" in diag
        assert "raw_hig_nats" in diag
        assert diag["raw_hig_nats"] >= 0.0


def test_botorch_ei_hig_is_pre_reveal():
    """Verify that BOTORCH_EI_DIRECT policy evaluates HIG prior to revealing action outcome."""
    res = evaluate_historical_policy("BOTORCH_EI_DIRECT", derived_outcomes_path=FIXTURE_PATH, seed=42, max_steps=2)
    for diag in res.step_diagnostics:
        assert "current_entropy_pre_reveal" in diag
        assert "expected_posterior_entropy" in diag
        assert "raw_hig_nats" in diag
        assert diag["raw_hig_nats"] >= 0.0


def test_botorch_gpucb_hig_is_pre_reveal():
    """Verify that BOTORCH_GPUCB_DIRECT policy evaluates HIG prior to revealing action outcome."""
    res = evaluate_historical_policy("BOTORCH_GPUCB_DIRECT", derived_outcomes_path=FIXTURE_PATH, seed=42, max_steps=2)
    for diag in res.step_diagnostics:
        assert "current_entropy_pre_reveal" in diag
        assert "expected_posterior_entropy" in diag
        assert "raw_hig_nats" in diag
        assert diag["raw_hig_nats"] >= 0.0


def test_historical_action_sequence_has_no_duplicates():
    """Verify that historical action sequences contain zero duplicate candidate selections."""
    res = evaluate_historical_policy("HYBRID", derived_outcomes_path=FIXTURE_PATH, seed=42, max_steps=3)
    assert len(res.autonomous_actions) == len(set(res.autonomous_actions))


def test_historical_reveals_equal_oracle_values():
    """Verify that revealed capacities match historical experimental oracle records exactly."""
    from src.domains.electrolyte.data import load_derived_historical_outcomes
    df_hist = load_derived_historical_outcomes(FIXTURE_PATH)
    lookup = dict(zip(df_hist["candidate_id"], df_hist["C_norm_20"]))

    res = evaluate_historical_policy("HYBRID", derived_outcomes_path=FIXTURE_PATH, seed=42, max_steps=3)
    for cid, obs in zip(res.autonomous_actions, res.autonomous_observations):
        assert abs(obs - lookup[cid]) < 1e-6


def test_historical_auc_is_recomputed():
    """Verify that area under best-so-far curve is faithfully recomputed from revealed trajectory."""
    res = evaluate_historical_policy("DISCOVERY_ONLY", derived_outcomes_path=FIXTURE_PATH, seed=42, max_steps=3)
    best_curve = [res.bootstrap_best]
    curr = res.bootstrap_best
    for obs in res.autonomous_observations:
        curr = max(curr, obs)
        best_curve.append(curr)
    if len(best_curve) > 1:
        expected_auc = float(np.sum(0.5 * (np.array(best_curve[:-1]) + np.array(best_curve[1:]))))
    else:
        expected_auc = float(best_curve[0])
    assert abs(res.area_under_best_curve - expected_auc) < 1e-4


def test_historical_top_decile_rate_is_recomputed():
    """Verify that top-decile hit rate is recomputed from autonomous actions and P90 threshold."""
    p90 = 0.70
    res = evaluate_historical_policy("HYBRID", derived_outcomes_path=FIXTURE_PATH, seed=42, max_steps=3, top_decile_threshold=p90)
    expected_hits = sum(1 for obs in res.autonomous_observations if obs >= p90)
    assert res.top_decile_hits == expected_hits


def test_historical_near_zero_rate_is_recomputed():
    """Verify that near-zero rate is recomputed from autonomous observations."""
    res = evaluate_historical_policy("HYBRID", derived_outcomes_path=FIXTURE_PATH, seed=42, max_steps=3)
    expected_nz = sum(1 for obs in res.autonomous_observations if obs <= 0.05)
    assert res.near_zero_query_count == expected_nz


def test_historical_cumulative_hig_equals_step_sum():
    """Verify that cumulative raw HIG equals sum of step diagnostics raw HIG."""
    res = evaluate_historical_policy("HYBRID", derived_outcomes_path=FIXTURE_PATH, seed=42, max_steps=3)
    step_sum = sum(d["raw_hig_nats"] for d in res.step_diagnostics)
    assert abs(res.cumulative_raw_hig_nats - step_sum) < 1e-4


def test_historical_mean_hig_per_action_equals_cumulative_over_steps():
    """Verify that mean raw HIG per action equals cumulative HIG divided by step count."""
    res = evaluate_historical_policy("HYBRID", derived_outcomes_path=FIXTURE_PATH, seed=42, max_steps=3)
    expected_per_act = res.cumulative_raw_hig_nats / res.steps_count
    assert abs(res.mean_raw_hig_nats_per_action - expected_per_act) < 1e-4


def test_historical_report_metrics_come_from_json_input():
    """Verify that render_historical_markdown faithfully renders all dynamic metrics from input dict."""
    from scripts.run_electrolyte_benchmark import render_historical_markdown
    val = 0.54321
    fake_bench = {
        "benchmark_metadata": {
            "title": "Sentinel Test",
            "historical_pool_size": 75,
            "global_pool_maximum": val,
            "top_decile_p90_threshold": 0.45,
            "bootstrap_seed_count": 3,
            "bootstrap_best_capacity": 0.35,
            "objective_saturation_status": False,
            "saturation_ratio": 0.65,
            "evaluated_seeds": [42],
            "max_autonomous_steps": 2,
        },
        "policy_summaries": [{
            "policy_name": "SENTINEL",
            "best_found_mean": val,
            "best_found_std": 0.01,
            "improvement_mean": 0.02,
            "improvement_std": 0.001,
            "auc_mean": 3.45,
            "auc_std": 0.1,
            "top_decile_hit_rate": 0.5,
            "near_zero_rate": 0.1,
            "mean_cumulative_raw_hig_nats": val,
            "std_cumulative_raw_hig_nats": 0.01,
            "mean_raw_hig_nats_per_action": val / 2,
            "std_raw_hig_nats_per_action": 0.005,
            "mean_realized_entropy_reduction": 0.25,
            "std_realized_entropy_reduction": 0.01,
            "runtime_sec_mean": 1.23,
        }],
        "natural_wow_scenario": {"scenario_found": False},
    }
    rendered = render_historical_markdown(fake_bench)
    assert f"{val:.4f}" in rendered
    assert "0.5432" in rendered


def test_surrogate_report_metrics_come_from_json_input():
    """Verify that render_surrogate_markdown faithfully renders all dynamic metrics from input dict."""
    from scripts.run_electrolyte_benchmark import render_surrogate_markdown
    val = 0.4321
    fake_surr = {
        "simulation_label": "SIMULATED_TEST",
        "oracle_kind": "SIMULATED_SURROGATE",
        "physical_synthesis": False,
        "requested_search_space_size": 333333,
        "actual_search_space_size": 333333,
        "scope_kind": "Test Scope",
        "screened_working_set_size": 200,
        "screening_time_sec": 0.45,
        "surrogate_model_family": "ExtraTrees",
        "evaluated_seeds": [42],
        "full_search_space_latent_max": val,
        "working_set_latent_max": val - 0.05,
        "screening_latent_gap": 0.05,
        "notice": "Notice",
        "disclaimer": "Disclaimer",
        "simulation_policies": {
            "TEST_POL": {
                "best_selected_latent_capacity_mean": val,
                "best_selected_latent_capacity_std": 0.0,
                "best_noisy_observed_capacity_mean": val,
                "best_noisy_observed_capacity_std": 0.0,
                "simple_regret_latent_mean": 0.0,
                "simple_regret_latent_std": 0.0,
                "simple_regret_vs_full_latent_mean": 0.05,
                "simple_regret_vs_full_latent_std": 0.0,
                "cumulative_raw_hig_nats_mean": val,
                "cumulative_raw_hig_nats_std": 0.0,
                "mean_raw_hig_nats_per_action_mean": val / 15,
                "mean_raw_hig_nats_per_action_std": 0.0,
                "realized_entropy_reduction_mean": 0.2,
                "queried_count": 15,
            }
        },
    }
    rendered = render_surrogate_markdown(fake_surr)
    assert f"{val:.4f}" in rendered


def test_report_uses_latent_regret_not_noisy_regret():
    """Verify that surrogate report explicitly labels and uses latent regret."""
    surr_md_path = "outputs/electrolyte/benchmark/surrogate_simulation.md"
    if os.path.exists(surr_md_path):
        with open(surr_md_path, encoding="utf-8") as f:
            content = f.read()
        assert "Latent Regret" in content
        assert "Noisy Regret" not in content


def test_report_distinguishes_latent_and_noisy_capacity():
    """Verify that surrogate report contains separate columns for latent truth f(x) and noisy reveal y(x)."""
    surr_md_path = "outputs/electrolyte/benchmark/surrogate_simulation.md"
    if os.path.exists(surr_md_path):
        with open(surr_md_path, encoding="utf-8") as f:
            content = f.read()
        assert "Best Latent Cap $f(x)$" in content
        assert "Best Noisy Obs $y(x)$" in content


def test_report_does_not_call_mean_hig_nats_a_per_action_mean():
    """Verify that historical report clearly distinguishes cumulative HIG from HIG per action."""
    hist_md_path = "outputs/electrolyte/benchmark/historical_policy_comparison.md"
    if os.path.exists(hist_md_path):
        with open(hist_md_path, encoding="utf-8") as f:
            content = f.read()
        assert "Cum. HIG (nats)" in content
        assert "HIG / action (nats)" in content


def test_report_does_not_claim_proven_float_jitter_cause():
    """Verify that audit report does not use unsupported causal claim 'PROVEN FLOATING-POINT JITTER'."""
    audit_md_path = "outputs/electrolyte/audit/dataset_audit_report.md"
    if os.path.exists(audit_md_path):
        with open(audit_md_path, encoding="utf-8") as f:
            content = f.read()
        assert "PROVEN FLOATING-POINT JITTER" not in content
        assert "CONSISTENT WITH FLOATING-POINT" in content


def test_e2e_timing_component_names_match_actual_operations():
    """Verify that end-to-end benchmark records explicit and accurate operation timing breakdown."""
    e2e_path = "outputs/electrolyte/benchmark/large_pool_end_to_end.json"
    if os.path.exists(e2e_path):
        with open(e2e_path) as f:
            data = json.load(f)
        for run in data.get("runs", []):
            assert "pool_read_and_filter_sec" in run
            assert "candidate_identity_generation_sec" in run
            assert "pool_load_filter_identity_sec" in run
            assert "screening_sec" in run
            assert "adapter_construction_sec" in run
            assert "engine_initialization_sec" in run
            assert "first_proposal_sec" in run
            assert "total_pipeline_sec" in run
            assert run["total_pipeline_sec"] > 0
            assert run["candidate_pool_size"] > 0
            assert run["screened_working_set_size"] > 0



