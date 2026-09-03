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
from typing import Any

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
        rows.append(
            f"| **{s['policy_name']}** | {s['best_found_mean']:.4f} ± {s['best_found_std']:.4f} | "
            f"{s['improvement_mean']:.4f} ± {s['improvement_std']:.4f} | "
            f"{s['auc_mean']:.2f} ± {s['auc_std']:.2f} | "
            f"{s['top_decile_hit_rate']*100:.1f}% | {s['near_zero_rate']*100:.1f}% | "
            f"{s['mean_hig_nats']:.4f} | {s['mean_hig_normalized']:.4f} | {s['runtime_sec_mean']:.2f}s |"
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

| Policy | Best Found ($C_{{\\text{{norm}}}}^{{20}}$) | Autonomous Imprv | AUC Best Curve | Top-Decile Hit % | Near-Zero % | Mean HIG (nats) | Mean HIG (norm) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
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


def run_surrogate_simulation(
    derived_path: str = DEFAULT_COMPATIBLE_DERIVED_PATH,
    virtual_csv_path: str = DEFAULT_VIRTUAL_1M_PATH,
    simulation_candidates_count: int = 50000,
    steps: int = 15,
) -> dict[str, Any]:
    """Runs a genuine closed-loop discovery screening simulation over unmeasured candidates.

    Phase 9 mandate:
    1. Screen candidates into a bounded working set.
    2. Wrap into ElectrolyteDomainAdapter with SurrogateElectrolyteOracle.
    3. Run genuine closed loop with ScientificDecisionEngine for multiple policies.
    4. Explicitly label all outputs with SIMULATED SURROGATE ORACLE.
    """
    df_hist = load_derived_historical_outcomes(derived_path)
    surrogate_oracle = SurrogateElectrolyteOracle(df_train=df_hist, feature_cols=ELECTROLYTE_SOLVENT_FEATURES)

    # Load candidate slice from LiFSI discovery space
    cands_df = load_lifsi_virtual_candidate_chunk(
        virtual_csv_path=virtual_csv_path,
        nrows=simulation_candidates_count,
        feature_cols=ELECTROLYTE_SOLVENT_FEATURES,
    )

    t0 = time.perf_counter()
    working_set = screen_large_pool_candidates(
        candidates_df=cands_df,
        working_set_size=200,
        feature_cols=ELECTROLYTE_SOLVENT_FEATURES,
        random_state=42,
    )
    screen_time = time.perf_counter() - t0

    sim_policies = ("HYBRID", "DISCOVERY_ONLY", "RANDOM")
    policy_results = {}
    rng = np.random.default_rng(42)

    for pol_name in sim_policies:
        adapter = ElectrolyteDomainAdapter(
            candidate_pool_df=working_set,
            oracle=surrogate_oracle,
        )

        if pol_name == "DISCOVERY_ONLY":
            pol = FalsificationFirstPolicy(mode=FalsificationPolicyMode.DISCOVERY_ONLY, w_hig=0.0, w_disc=1.0, w_cost=0.0)
        elif pol_name == "HYBRID":
            pol = FalsificationFirstPolicy(mode=FalsificationPolicyMode.HYBRID, w_hig=1.0, w_disc=0.8, w_cost=0.0)
        else:
            pol = FalsificationFirstPolicy(mode=FalsificationPolicyMode.HYBRID, w_hig=0.5, w_disc=0.5, w_cost=0.0)

        engine = ScientificDecisionEngine(
            domain=adapter,
            policy=pol,
            optimizer_backend=BoTorchBackend(),
            seed=42,
        )
        init_actions = adapter.get_default_initial_actions(n_seed=3, seed=42)
        engine.initialize(init_actions)

        queried_cids = []
        revealed_vals = []
        cum_hig = 0.0

        for s in range(steps):
            valid = adapter.list_valid_actions(engine.get_state())
            if not valid:
                break
            if pol_name == "RANDOM":
                act = rng.choice(valid)
                outcome = engine.execute_external_action(act, metadata={"policy": "RANDOM"})
            else:
                rec = engine.propose_next_experiment()
                act = rec.action
                outcome = engine.execute_recommendation(rec)
                cum_hig += float(rec.uncertainty_summary.get("raw_hig_nats", 0.0))

            c_val = float(outcome.revealed_data["C_norm_20"])
            queried_cids.append(act.candidate_id)
            revealed_vals.append(c_val)

        policy_results[pol_name] = {
            "best_simulated_capacity": round(float(max(revealed_vals)), 4) if revealed_vals else 0.0,
            "mean_simulated_capacity": round(float(np.mean(revealed_vals)), 4) if revealed_vals else 0.0,
            "cumulative_hig_nats": round(cum_hig, 4),
            "queried_count": len(queried_cids),
        }

    return {
        "simulation_label": "SIMULATED SURROGATE ORACLE - In-Silico Computational Approximation Only",
        "oracle_kind": "SIMULATED_SURROGATE",
        "physical_synthesis": False,
        "search_space_slice": f"{len(cands_df):,} LiFSI Virtual Candidates",
        "screened_working_set_size": len(working_set),
        "screening_time_sec": round(screen_time, 4),
        "surrogate_model_family": "ExtraTreesRegressor (100 trees, max_depth=8)",
        "simulation_policies": policy_results,
        "disclaimer": "Computational simulation under frozen surrogate. Not physical experimental validation.",
        "notice": (
            "This simulation evaluates algorithmic screening throughput and working-set information capture. "
            "It does not represent wet-lab physical experimental synthesis."
        ),
    }


def main():
    print("=" * 80)
    print("STARTING FULL ELECTROLYTE BENCHMARK EXECUTION")
    print("=" * 80)
    os.makedirs(OUT_BENCHMARK_DIR, exist_ok=True)

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

    # 2. Retrospective Next-Batch Ranking (Phase 12: Separate AIcoScientist and RF)
    print("\n[BENCHMARK 2] Running Temporal Forward Next-Batch Ranking (AIcoScientist vs RF Baseline)...")
    run_retrospective_next_batch_ranking(
        derived_outcomes_path=DEFAULT_COMPATIBLE_DERIVED_PATH,
        feature_cols=ELECTROLYTE_SOLVENT_FEATURES,
        out_dir=OUT_BENCHMARK_DIR,
    )
    print("Saved aicoscientist_temporal_next_batch.json and rf_temporal_baseline.json")

    # 3. Hypothesis Calibration & Sensitivity (Phase 11)
    print("\n[BENCHMARK 3] Running Predictive Hypothesis Calibration & Variance-Floor Sensitivity...")
    df_hist = load_derived_historical_outcomes(DEFAULT_COMPATIBLE_DERIVED_PATH)
    calib_res = evaluate_hypothesis_calibration(
        df_historical=df_hist,
        feature_cols=ELECTROLYTE_SOLVENT_FEATURES,
    )
    with open(os.path.join(OUT_BENCHMARK_DIR, "hypothesis_calibration.json"), "w") as f:
        json.dump(calib_res, f, indent=2)
    print("Saved hypothesis_calibration.json")

    # 4. Large-Pool Scale Benchmark (Phase F & 8)
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
    print("Saved large_pool_scale.json")

    # 5. Surrogate Simulation Benchmark (Phase 9)
    print("\n[BENCHMARK 5] Running In-Silico Surrogate Closed-Loop Simulation Benchmark...")
    surr_results = run_surrogate_simulation(
        derived_path=DEFAULT_COMPATIBLE_DERIVED_PATH,
        virtual_csv_path=DEFAULT_VIRTUAL_1M_PATH,
        simulation_candidates_count=50000,
        steps=15,
    )
    with open(os.path.join(OUT_BENCHMARK_DIR, "surrogate_simulation.json"), "w") as f:
        json.dump(surr_results, f, indent=2)

    surr_md = f"""# In-Silico Surrogate Simulation Benchmark Report

**Status:** `{surr_results['simulation_label']}`  
**Oracle Kind:** `{surr_results['oracle_kind']}`  
**Physical Synthesis:** `{surr_results['physical_synthesis']}`  
**Search Space Slice:** {surr_results['search_space_slice']}  
**Working Set Size:** {surr_results['screened_working_set_size']}  
**Screening Runtime:** {surr_results['screening_time_sec']} seconds  
**Surrogate Model Family:** {surr_results['surrogate_model_family']}  

## Policy Closed-Loop Performance
| Policy | Best Simulated Capacity | Mean Simulated Capacity | Cumulative HIG (nats) | Queried Steps |
| :--- | :---: | :---: | :---: | :---: |
| **HYBRID** | {surr_results['simulation_policies']['HYBRID']['best_simulated_capacity']:.4f} | {surr_results['simulation_policies']['HYBRID']['mean_simulated_capacity']:.4f} | {surr_results['simulation_policies']['HYBRID']['cumulative_hig_nats']:.4f} | {surr_results['simulation_policies']['HYBRID']['queried_count']} |
| **DISCOVERY_ONLY** | {surr_results['simulation_policies']['DISCOVERY_ONLY']['best_simulated_capacity']:.4f} | {surr_results['simulation_policies']['DISCOVERY_ONLY']['mean_simulated_capacity']:.4f} | {surr_results['simulation_policies']['DISCOVERY_ONLY']['cumulative_hig_nats']:.4f} | {surr_results['simulation_policies']['DISCOVERY_ONLY']['queried_count']} |
| **RANDOM** | {surr_results['simulation_policies']['RANDOM']['best_simulated_capacity']:.4f} | {surr_results['simulation_policies']['RANDOM']['mean_simulated_capacity']:.4f} | {surr_results['simulation_policies']['RANDOM']['cumulative_hig_nats']:.4f} | {surr_results['simulation_policies']['RANDOM']['queried_count']} |

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
