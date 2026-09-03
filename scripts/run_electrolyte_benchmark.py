"""Executes all electrolyte benchmarks:
1. Retrospective finite historical label-pool replay (multi-seed, multi-policy).
2. Large-pool two-stage screening scalability (10k, 100k, 333k LiFSI, 999k).
3. In-silico surrogate simulation benchmark (333k LiFSI space, labeled SIMULATED).
4. Natural Wow scenario discovery.
"""

import json
import os
import sys
import time
from typing import Any, Sequence

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd

from src.domains.electrolyte.config import (
    ELECTROLYTE_DOMAIN_CONFIG,
    ELECTROLYTE_SOLVENT_FEATURES,
)
from src.domains.electrolyte.data import (
    DEFAULT_COMPATIBLE_DERIVED_PATH,
    DEFAULT_VIRTUAL_1M_PATH,
    load_derived_historical_outcomes,
    load_lifsi_virtual_candidate_chunk,
)
from src.domains.electrolyte.oracle import SurrogateElectrolyteOracle
from src.domains.electrolyte.screening import (
    benchmark_large_pool_screening,
    screen_large_pool_candidates,
)
from src.domains.electrolyte.adapter import ElectrolyteDomainAdapter
from src.domains.electrolyte.hypotheses import evaluate_hypothesis_calibration
from src.evaluation.electrolyte_benchmark import (
    run_comprehensive_historical_benchmark,
    run_retrospective_next_batch_ranking,
)
from src.optimization.botorch_backend import BoTorchBackend
from src.science.decision_engine import ScientificDecisionEngine
from src.science.falsification.policy import FalsificationFirstPolicy, FalsificationPolicyMode

OUT_BENCHMARK_DIR = "outputs/electrolyte/benchmark"


def render_historical_markdown(bench_data: dict) -> str:
    meta = bench_data["benchmark_metadata"]
    summaries = bench_data["policy_summaries"]
    wow = bench_data["natural_wow_scenario"]

    rows = []
    for s in summaries:
        cum_hig = s.get("mean_cumulative_raw_hig_nats", s.get("mean_hig_nats", 0.0))
        std_cum_hig = s.get("std_cumulative_raw_hig_nats", 0.0)
        per_act_hig = s.get("mean_raw_hig_nats_per_action", 0.0)
        std_per_act_hig = s.get("std_raw_hig_nats_per_action", 0.0)
        ent_red = s.get("mean_realized_entropy_reduction", s.get("mean_entropy_reduction", 0.0))
        std_ent_red = s.get("std_realized_entropy_reduction", 0.0)

        rows.append(
            f"| **{s['policy_name']}** | {s['best_found_mean']:.4f} ± {s['best_found_std']:.4f} | "
            f"{s['improvement_mean']:.4f} ± {s['improvement_std']:.4f} | "
            f"{s['auc_mean']:.2f} ± {s['auc_std']:.2f} | "
            f"{s['top_decile_hit_rate']*100:.1f}% | {s['near_zero_rate']*100:.1f}% | "
            f"{cum_hig:.4f} ± {std_cum_hig:.4f} | "
            f"{per_act_hig:.4f} ± {std_per_act_hig:.4f} | "
            f"{ent_red:.4f} ± {std_ent_red:.4f} | "
            f"{s['runtime_sec_mean']:.2f}s |"
        )
    table_text = "\n".join(rows)

    wow_text = ""
    if wow.get("scenario_found"):
        wow_text = f"""
### Natural Wow Scenario Discovered:
* **Seed:** {wow['seed']} at Step {wow['step_index'] + 1}
* **Discovery-Only Choice:** `{wow['discovery_only_choice']['candidate_id']}` (Revealed $C_{{\\text{{norm}}}}^{{20}} = {wow['discovery_only_choice']['revealed_C_norm_20']:.4f}$)
* **Hybrid Choice:** `{wow['hybrid_choice']['candidate_id']}` (Revealed $C_{{\\text{{norm}}}}^{{20}} = {wow['hybrid_choice']['revealed_C_norm_20']:.4f}$)
* **Analysis:** {wow['scientific_divergence_analysis']}
"""
    else:
        wow_text = f"""
* **Status:** `NO NATURAL ELECTROLYTE WOW SCENARIO FOUND UNDER PREREGISTERED SETTINGS`
* **Preregistered Criteria Enforced:**
  1. $c_{{\\text{{disc}}}} \\neq c_{{\\text{{hyb}}}}$
  2. $EI(c_{{\\text{{disc}}}}) > EI(c_{{\\text{{hyb}}}})$
  3. $HIG_{{\\text{{nats}}}}(c_{{\\text{{hyb}}}}) > HIG_{{\\text{{nats}}}}(c_{{\\text{{disc}}}})$
  4. $V_{{\\text{{hyb}}}}(c_{{\\text{{hyb}}}}) > V_{{\\text{{hyb}}}}(c_{{\\text{{disc}}}})$
  5. Material epistemic shift: $\\max |P_{{\\text{{post}}}} - P_{{\\text{{prior}}}}| \\ge 0.01$
* **Outcome:** No candidate pair across {meta['evaluated_seeds']} seeds met all 5 criteria simultaneously.
"""

    md = f"""# Retrospective Finite Historical Label-Pool Replay Benchmark Report

**Dataset:** `AmanchukwuLab/AL-anode-free` (Pool-Compatible Historical Outcomes, N={meta['historical_pool_size']})  
**Evaluation Scope:** Retrospective replay across 5 deterministic seeds ({', '.join(map(str, meta['evaluated_seeds']))}).  
**Initial Evidence:** Batch 0 compatible seed cells (N={meta['bootstrap_seed_count']}, Best $C_{{\\text{{norm}}}}^{{20}} = {meta['bootstrap_best_capacity']:.4f}$).  
**Global Historical Pool Maximum:** $C_{{\\text{{norm}}}}^{{20}} = {meta['global_pool_maximum']:.4f}$  
**Objective Saturation Status:** `{meta['objective_saturation_status']}` (Saturation ratio: {meta['saturation_ratio']:.4f})  
**Top-Decile Threshold ($P_{{90}}$):** $C_{{\\text{{norm}}}}^{{20}} \\ge {meta['top_decile_p90_threshold']:.4f}$  

---

## 1. Multi-Policy Benchmark Comparison

| Policy | Best Found ($C_{{\\text{{norm}}}}^{{20}}$) | Autonomous Imprv | AUC Best Curve | Top-Decile Hit % | Near-Zero % | Cum. HIG (nats) | HIG / action (nats) | Entropy reduction | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{table_text}

---

## 2. Saturation & Bootstrap Accounting

* **Bootstrap Independence:** The 3 Batch-0 seed observations are established as prior baseline evidence. All reported improvements and top-decile hits measure strictly autonomous policy actions beyond the bootstrap.
* **Objective Saturation Analysis:** The historical dataset has high saturation ({meta['saturation_ratio']*100:.1f}% of global optimum already present in Batch 0 seed). Therefore, autonomous improvement and area under best-so-far curve are the appropriate discriminating metrics.

---

## 3. Natural Policy Divergence ("Wow" Scenario)
{wow_text}

---
*Generated by AIcoScientist Electrolyte Benchmark Suite.*
"""
    return md


def run_large_pool_end_to_end_decision_benchmark(
    virtual_csv_path: str = DEFAULT_VIRTUAL_1M_PATH,
    pool_sizes: Sequence[int] = (10000, 100000, 333333),
    working_set_size: int = 200,
    feature_cols: Sequence[str] = ELECTROLYTE_SOLVENT_FEATURES,
    random_state: int = 42,
    out_path: str = "outputs/electrolyte/benchmark/large_pool_end_to_end.json",
) -> dict[str, Any]:
    """Measures end-to-end timing across the complete decision pipeline:
    pool loading -> candidate identity -> two-stage screening -> adapter construction ->
    engine initialization -> first experiment proposal.
    """
    df_hist = load_derived_historical_outcomes(DEFAULT_COMPATIBLE_DERIVED_PATH)
    surrogate_oracle = SurrogateElectrolyteOracle(df_train=df_hist, feature_cols=feature_cols)
    f_cols = list(feature_cols)

    runs = []
    for N in pool_sizes:
        t_start = time.perf_counter()

        # 1. Pool load timing
        t0 = time.perf_counter()
        cands_chunk = load_lifsi_virtual_candidate_chunk(
            virtual_csv_path=virtual_csv_path,
            nrows=N,
            feature_cols=f_cols,
        )
        pool_load_sec = time.perf_counter() - t0

        # 2. Candidate identity timing
        t0 = time.perf_counter()
        cand_ids = cands_chunk["candidate_id"].tolist()
        candidate_identity_sec = time.perf_counter() - t0

        # 3. Two-stage screening timing (using FrozenElectrolyteFeatureScaler)
        t0 = time.perf_counter()
        working_set = screen_large_pool_candidates(
            candidates_df=cands_chunk,
            working_set_size=working_set_size,
            feature_cols=f_cols,
            random_state=random_state,
        )
        screening_sec = time.perf_counter() - t0

        # 4. Adapter construction timing
        t0 = time.perf_counter()
        adapter = ElectrolyteDomainAdapter(
            candidate_pool_df=working_set,
            oracle=surrogate_oracle,
        )
        adapter_construction_sec = time.perf_counter() - t0

        # 5. Engine initialization timing
        t0 = time.perf_counter()
        policy = FalsificationFirstPolicy(
            mode=FalsificationPolicyMode.HYBRID,
            w_hig=1.0,
            w_disc=0.8,
            w_cost=0.0,
        )
        engine = ScientificDecisionEngine(
            domain=adapter,
            policy=policy,
            optimizer_backend=BoTorchBackend(),
            seed=random_state,
        )
        init_actions = adapter.get_default_initial_actions(n_seed=3, seed=random_state)
        engine.initialize(init_actions)
        engine_initialization_sec = time.perf_counter() - t0

        # 6. First proposal timing
        t0 = time.perf_counter()
        proposal = engine.propose_next_experiment()
        first_proposal_sec = time.perf_counter() - t0

        total_pipeline_sec = time.perf_counter() - t_start
        mem_mb = round(float(cands_chunk.memory_usage(deep=True).sum() / (1024 * 1024)), 2)

        runs.append({
            "candidate_pool_size": int(N),
            "screened_working_set_size": len(working_set),
            "pool_load_sec": round(pool_load_sec, 4),
            "candidate_identity_sec": round(candidate_identity_sec, 4),
            "screening_sec": round(screening_sec, 4),
            "adapter_construction_sec": round(adapter_construction_sec, 4),
            "engine_initialization_sec": round(engine_initialization_sec, 4),
            "first_proposal_sec": round(first_proposal_sec, 4),
            "total_pipeline_sec": round(total_pipeline_sec, 4),
            "first_proposed_candidate_id": proposal.action.candidate_id,
            "first_proposal_total_value": round(float(proposal.total_value), 4),
            "memory_mb_estimate": mem_mb,
        })

    result = {
        "benchmark_title": "Large-Pool End-to-End Decision Pipeline Benchmark",
        "description": "Measures wall-clock time from raw candidate pool loading to first scientific recommendation.",
        "evaluated_pool_sizes": [int(N) for N in pool_sizes],
        "target_electrolyte_subspace": "LiFSI Salt Subspace (333,333 candidate scope)",
        "feature_count": len(f_cols),
        "working_set_size": working_set_size,
        "runs": runs,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_surrogate_simulation(
    derived_path: str = DEFAULT_COMPATIBLE_DERIVED_PATH,
    virtual_csv_path: str = DEFAULT_VIRTUAL_1M_PATH,
    simulation_candidates_count: int = 50000,
    seeds: Sequence[int] = (42, 101, 2024),
    steps: int = 15,
) -> dict[str, Any]:
    """Runs a genuine closed-loop discovery screening simulation over unmeasured candidates."""
    df_hist = load_derived_historical_outcomes(derived_path)
    f_cols = list(ELECTROLYTE_SOLVENT_FEATURES)

    # Load candidate slice from LiFSI discovery space
    cands_df = load_lifsi_virtual_candidate_chunk(
        virtual_csv_path=virtual_csv_path,
        nrows=simulation_candidates_count,
        feature_cols=f_cols,
    )

    t0 = time.perf_counter()
    working_set = screen_large_pool_candidates(
        candidates_df=cands_df,
        working_set_size=200,
        feature_cols=f_cols,
        random_state=42,
    )
    screen_time = time.perf_counter() - t0

    sim_policies = (
        "RANDOM",
        "BOTORCH_EI_DIRECT",
        "BOTORCH_GPUCB_DIRECT",
        "PURE_FALSIFICATION",
        "HYBRID_DEFAULT",
        "DISCOVERY_ONLY",
    )

    # Compute omniscient oracle truth on working set for offline regret
    oracle_truth_pool = SurrogateElectrolyteOracle(df_train=df_hist, feature_cols=f_cols)
    oracle_values = [
        oracle_truth_pool.predict_capacity_loss(row[f_cols].to_numpy(dtype=np.float64))
        for _, row in working_set.iterrows()
    ]
    oracle_max = float(np.max(oracle_values))

    policy_seed_runs: dict[str, list[dict[str, Any]]] = {p: [] for p in sim_policies}

    for pol_name in sim_policies:
        for s in seeds:
            # Independent frozen surrogate oracle per policy and seed
            surrogate_oracle = SurrogateElectrolyteOracle(df_train=df_hist, feature_cols=f_cols)
            adapter = ElectrolyteDomainAdapter(
                candidate_pool_df=working_set,
                oracle=surrogate_oracle,
            )

            is_random = (pol_name == "RANDOM")
            is_botorch = pol_name in ("BOTORCH_EI_DIRECT", "BOTORCH_GPUCB_DIRECT")

            if pol_name == "DISCOVERY_ONLY":
                pol = FalsificationFirstPolicy(mode=FalsificationPolicyMode.DISCOVERY_ONLY, w_hig=0.0, w_disc=1.0, w_cost=0.0)
            elif pol_name == "PURE_FALSIFICATION":
                pol = FalsificationFirstPolicy(mode=FalsificationPolicyMode.PURE_FALSIFICATION, w_hig=1.0, w_disc=0.0, w_cost=0.0)
            else:
                pol = FalsificationFirstPolicy(mode=FalsificationPolicyMode.HYBRID, w_hig=1.0, w_disc=0.8, w_cost=0.0)

            opt_backend = BoTorchBackend()
            engine = ScientificDecisionEngine(
                domain=adapter,
                policy=pol,
                optimizer_backend=opt_backend,
                seed=s,
            )
            init_actions = adapter.get_default_initial_actions(n_seed=3, seed=s)
            engine.initialize(init_actions)
            init_entropy = float(engine.ensemble.get_entropy())

            queried_cids = []
            revealed_vals = []
            cum_hig_nats = 0.0
            rng = np.random.default_rng(s)

            for step_idx in range(steps):
                valid = adapter.list_valid_actions(engine.get_state())
                if not valid:
                    break

                if is_random:
                    chosen_action = rng.choice(valid)
                    cand_f = adapter.get_candidate_features(chosen_action.candidate_id)
                    comp = np.array([cand_f[f] for f in f_cols], dtype=np.float64)
                    eval_res = pol.hig_estimator.evaluate_action_discrimination(
                        candidate_id=chosen_action.candidate_id,
                        action_type=chosen_action.action_type,
                        composition=comp,
                        ensemble=engine.ensemble,
                        seed=s,
                    )
                    raw_hig = float(eval_res.hypothesis_information_gain)
                    outcome = engine.execute_external_action(chosen_action, metadata={"policy": "RANDOM"})

                elif is_botorch:
                    strat = "expected_improvement" if pol_name == "BOTORCH_EI_DIRECT" else "gp_ucb"
                    obs_dict = engine.observations_by_modality.get("CAPACITY_TEST", {})
                    obs_rows = [{"candidate_id": cid, "C_norm_20": float(val), **adapter.get_candidate_features(cid)} for cid, val in obs_dict.items()]
                    obs_df = pd.DataFrame(obs_rows)

                    cand_rows = [{"candidate_id": a.candidate_id, **adapter.get_candidate_features(a.candidate_id)} for a in valid]
                    cand_pool_df = pd.DataFrame(cand_rows)

                    proposals = opt_backend.propose(
                        observations=obs_df,
                        candidate_pool=cand_pool_df,
                        objective="C_norm_20",
                        feature_columns=f_cols,
                        candidate_id_column="candidate_id",
                        strategy=strat,
                        seed=s,
                    )
                    top_cid = proposals[0].candidate_id
                    chosen_action = next(a for a in valid if a.candidate_id == top_cid)

                    cand_f = adapter.get_candidate_features(chosen_action.candidate_id)
                    comp = np.array([cand_f[f] for f in f_cols], dtype=np.float64)
                    eval_res = pol.hig_estimator.evaluate_action_discrimination(
                        candidate_id=chosen_action.candidate_id,
                        action_type=chosen_action.action_type,
                        composition=comp,
                        ensemble=engine.ensemble,
                        seed=s,
                    )
                    raw_hig = float(eval_res.hypothesis_information_gain)
                    outcome = engine.execute_external_action(chosen_action, metadata={"policy": pol_name})

                else:
                    rec = engine.propose_next_experiment()
                    chosen_action = rec.action
                    raw_hig = float(rec.uncertainty_summary.get("raw_hig_nats", rec.scientific_information_value))
                    outcome = engine.execute_recommendation(rec)

                cum_hig_nats += raw_hig
                c_val = float(outcome.revealed_data["C_norm_20"])
                queried_cids.append(chosen_action.candidate_id)
                revealed_vals.append(c_val)

            final_entropy = float(engine.ensemble.get_entropy())
            best_val = float(max(revealed_vals)) if revealed_vals else 0.0
            mean_val = float(np.mean(revealed_vals)) if revealed_vals else 0.0
            n_acts = max(1, len(revealed_vals))

            policy_seed_runs[pol_name].append({
                "seed": s,
                "best_simulated_capacity": round(best_val, 4),
                "mean_simulated_capacity": round(mean_val, 4),
                "cumulative_raw_hig_nats": round(cum_hig_nats, 4),
                "mean_raw_hig_nats_per_action": round(cum_hig_nats / n_acts, 4),
                "realized_entropy_reduction": round(init_entropy - final_entropy, 4),
                "regret_vs_oracle_max": round(max(0.0, oracle_max - best_val), 4),
                "queried_count": len(queried_cids),
            })

    # Summarize across seeds
    policy_summaries = {}
    for pol_name, r_list in policy_seed_runs.items():
        b_list = [r["best_simulated_capacity"] for r in r_list]
        m_list = [r["mean_simulated_capacity"] for r in r_list]
        hig_list = [r["cumulative_raw_hig_nats"] for r in r_list]
        per_act_hig = [r["mean_raw_hig_nats_per_action"] for r in r_list]
        ent_red = [r["realized_entropy_reduction"] for r in r_list]
        reg_list = [r["regret_vs_oracle_max"] for r in r_list]

        policy_summaries[pol_name] = {
            "best_simulated_capacity_mean": round(float(np.mean(b_list)), 4),
            "best_simulated_capacity_std": round(float(np.std(b_list)), 4),
            "mean_simulated_capacity_mean": round(float(np.mean(m_list)), 4),
            "mean_simulated_capacity_std": round(float(np.std(m_list)), 4),
            "cumulative_raw_hig_nats_mean": round(float(np.mean(hig_list)), 4),
            "cumulative_raw_hig_nats_std": round(float(np.std(hig_list)), 4),
            "mean_raw_hig_nats_per_action_mean": round(float(np.mean(per_act_hig)), 4),
            "mean_raw_hig_nats_per_action_std": round(float(np.std(per_act_hig)), 4),
            "realized_entropy_reduction_mean": round(float(np.mean(ent_red)), 4),
            "regret_vs_oracle_max_mean": round(float(np.mean(reg_list)), 4),
            "queried_count": r_list[0]["queried_count"] if r_list else 0,
            # Backwards-compatible aliases
            "best_simulated_capacity": round(float(np.mean(b_list)), 4),
            "mean_simulated_capacity": round(float(np.mean(m_list)), 4),
            "cumulative_hig_nats": round(float(np.mean(hig_list)), 4),
        }

    return {
        "simulation_label": "SIMULATED SURROGATE ORACLE - In-Silico Computational Approximation Only",
        "oracle_kind": "SIMULATED_SURROGATE",
        "physical_synthesis": False,
        "search_space_slice": f"{len(cands_df):,} LiFSI Virtual Candidates",
        "screened_working_set_size": len(working_set),
        "screening_time_sec": round(screen_time, 4),
        "surrogate_model_family": "ExtraTreesRegressor (100 trees, max_depth=8)",
        "evaluated_seeds": list(seeds),
        "oracle_pool_maximum": round(oracle_max, 4),
        "simulation_policies": policy_summaries,
        "detailed_policy_seed_runs": policy_seed_runs,
        "disclaimer": "Computational simulation under frozen surrogate. Not physical experimental validation.",
        "notice": (
            "This simulation evaluates algorithmic screening throughput and working-set information capture. "
            "It does not represent wet-lab physical experimental synthesis."
        ),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-surrogate", action="store_true", help="Run only Benchmark 6 (surrogate simulation)")
    args = parser.parse_args()

    print("=" * 80)
    print("STARTING ELECTROLYTE BENCHMARK EXECUTION")
    print("=" * 80)
    os.makedirs(OUT_BENCHMARK_DIR, exist_ok=True)

    if not args.only_surrogate:
        # 1. Historical Benchmark (Phases G, 3, 4, 5, 10)
        print("\n[BENCHMARK 1] Running Retrospective Finite Historical Benchmark (5 Seeds, 6 Policies)...")
        hist_results = run_comprehensive_historical_benchmark(
            derived_outcomes_path=DEFAULT_COMPATIBLE_DERIVED_PATH,
            seeds=(42, 101, 2024, 314, 7),
            policies=(
                "RANDOM",
                "DISCOVERY_ONLY",
                "PURE_FALSIFICATION",
                "HYBRID",
                "BOTORCH_EI_DIRECT",
                "BOTORCH_GPUCB_DIRECT",
            ),
            max_steps=15,
        )

        with open(os.path.join(OUT_BENCHMARK_DIR, "historical_policy_comparison.json"), "w") as f:
            json.dump(hist_results, f, indent=2)

        hist_md = render_historical_markdown(hist_results)
        with open(os.path.join(OUT_BENCHMARK_DIR, "historical_policy_comparison.md"), "w", encoding="utf-8") as f:
            f.write(hist_md)
        print("Saved historical_policy_comparison.json and .md")

        # Natural Wow scenario
        with open(os.path.join(OUT_BENCHMARK_DIR, "natural_wow_scenario.json"), "w") as f:
            json.dump(hist_results["natural_wow_scenario"], f, indent=2)
        print(f"Natural Wow Scenario: {hist_results['natural_wow_scenario'].get('scenario_found')}")

        # Policy Equivalence Diagnostics
        equiv_diag = hist_results.get("policy_equivalence_diagnostics", {})
        with open(os.path.join(OUT_BENCHMARK_DIR, "policy_equivalence_diagnostics.json"), "w") as f:
            json.dump(equiv_diag, f, indent=2)
        print("Saved policy_equivalence_diagnostics.json")

        # 2. Retrospective Next-Batch Ranking (Separating All Policies)
        print("\n[BENCHMARK 2] Running Temporal Forward Next-Batch Ranking across All Policies...")
        run_retrospective_next_batch_ranking(
            derived_outcomes_path=DEFAULT_COMPATIBLE_DERIVED_PATH,
            feature_cols=ELECTROLYTE_SOLVENT_FEATURES,
            out_dir=OUT_BENCHMARK_DIR,
        )
        print("Saved aicoscientist_temporal_next_batch.json and rf_temporal_baseline.json")

        # 3. Hypothesis Calibration & Sensitivity (With HIG Ranking Stability)
        print("\n[BENCHMARK 3] Running Predictive Hypothesis Calibration & Variance-Floor Sensitivity...")
        df_hist = load_derived_historical_outcomes(DEFAULT_COMPATIBLE_DERIVED_PATH)
        calib_res = evaluate_hypothesis_calibration(
            df_historical=df_hist,
            feature_cols=ELECTROLYTE_SOLVENT_FEATURES,
        )
        with open(os.path.join(OUT_BENCHMARK_DIR, "hypothesis_calibration.json"), "w") as f:
            json.dump(calib_res, f, indent=2)
        print("Saved hypothesis_calibration.json")

        # 4. Large-Pool Scale & Screening Benchmark
        print("\n[BENCHMARK 4] Running Large-Pool Two-Stage Screening Scalability Benchmark (10k, 100k, 333k, 999k)...")
        scale_results = benchmark_large_pool_screening(
            virtual_csv_path=DEFAULT_VIRTUAL_1M_PATH,
            sample_sizes=(10000, 100000, 333333, 999999),
            working_set_size=200,
            feature_cols=ELECTROLYTE_SOLVENT_FEATURES,
            random_state=42,
        )
        with open(os.path.join(OUT_BENCHMARK_DIR, "large_pool_scale.json"), "w") as f:
            json.dump(scale_results, f, indent=2)
        with open(os.path.join(OUT_BENCHMARK_DIR, "large_pool_screening_benchmarks.json"), "w") as f:
            json.dump(scale_results, f, indent=2)
        print("Saved large_pool_scale.json and large_pool_screening_benchmarks.json")

        # 5. Large-Pool End-to-End Decision Pipeline Benchmark
        print("\n[BENCHMARK 5] Running Large-Pool End-to-End Decision Benchmark (10k, 100k, 333k LiFSI)...")
        e2e_results = run_large_pool_end_to_end_decision_benchmark(
            virtual_csv_path=DEFAULT_VIRTUAL_1M_PATH,
            pool_sizes=(10000, 100000, 333333),
            working_set_size=200,
            feature_cols=ELECTROLYTE_SOLVENT_FEATURES,
            random_state=42,
            out_path=os.path.join(OUT_BENCHMARK_DIR, "large_pool_end_to_end.json"),
        )
        print("Saved large_pool_end_to_end.json")

    # 6. Surrogate Simulation Benchmark
    print("\n[BENCHMARK 6] Running In-Silico Surrogate Closed-Loop Simulation Benchmark...")
    surr_results = run_surrogate_simulation(
        derived_path=DEFAULT_COMPATIBLE_DERIVED_PATH,
        virtual_csv_path=DEFAULT_VIRTUAL_1M_PATH,
        simulation_candidates_count=50000,
        seeds=(42, 101, 2024),
        steps=15,
    )
    with open(os.path.join(OUT_BENCHMARK_DIR, "surrogate_simulation.json"), "w") as f:
        json.dump(surr_results, f, indent=2)

    surr_rows = []
    for pol_k, s_dict in surr_results["simulation_policies"].items():
        surr_rows.append(
            f"| **{pol_k}** | {s_dict['best_simulated_capacity_mean']:.4f} ± {s_dict['best_simulated_capacity_std']:.4f} | "
            f"{s_dict['mean_simulated_capacity_mean']:.4f} ± {s_dict['mean_simulated_capacity_std']:.4f} | "
            f"{s_dict['cumulative_raw_hig_nats_mean']:.4f} ± {s_dict['cumulative_raw_hig_nats_std']:.4f} | "
            f"{s_dict['mean_raw_hig_nats_per_action_mean']:.4f} ± {s_dict['mean_raw_hig_nats_per_action_std']:.4f} | "
            f"{s_dict['realized_entropy_reduction_mean']:.4f} | "
            f"{s_dict['regret_vs_oracle_max_mean']:.4f} | "
            f"{s_dict['queried_count']} |"
        )
    surr_table = "\n".join(surr_rows)

    surr_md = f"""# In-Silico Surrogate Simulation Benchmark Report

**Status:** `{surr_results['simulation_label']}`  
**Oracle Kind:** `{surr_results['oracle_kind']}`  
**Physical Synthesis:** `{surr_results['physical_synthesis']}`  
**Search Space Slice:** {surr_results['search_space_slice']}  
**Working Set Size:** {surr_results['screened_working_set_size']}  
**Screening Runtime:** {surr_results['screening_time_sec']} seconds  
**Surrogate Model Family:** {surr_results['surrogate_model_family']}  
**Oracle Working-Set Max Capacity:** {surr_results['oracle_pool_maximum']:.4f}  

## Policy Closed-Loop Performance (Mean ± Std over Seeds {', '.join(map(str, surr_results['evaluated_seeds']))})
| Policy | Best Simulated Cap | Mean Simulated Cap | Cum. HIG (nats) | HIG / action (nats) | Entropy reduction | Mean Regret | Steps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{surr_table}

> [!IMPORTANT]
> {surr_results['notice']}  
> **Disclaimer:** {surr_results['disclaimer']}
"""
    with open(os.path.join(OUT_BENCHMARK_DIR, "surrogate_simulation.md"), "w", encoding="utf-8") as f:
        f.write(surr_md)
    print("Saved surrogate_simulation.json and .md")

    print("\nALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
