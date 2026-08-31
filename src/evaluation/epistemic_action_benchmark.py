from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.datasets.auirh_actions import AuIrRhMultimodalOracle
from src.science.actions import ExperimentActionType, ScientificAction
from src.science.discovery_engine import AutonomousDiscoveryEngine

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("outputs/epistemic_actions/auirh_demo")


def run_single_campaign(
    policy_name: str,
    seed: int,
    budget_limit: float = 60.0,
    cost_xrd: float = 1.0,
    cost_property: float = 5.0,
) -> dict[str, Any]:
    """Runs a single simulated scientific discovery campaign under a specific policy."""
    engine = AutonomousDiscoveryEngine(
        seed=seed,
        cost_xrd=cost_xrd,
        cost_property=cost_property,
    )
    engine.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=seed)

    cost_history: list[float] = [engine.total_budget_spent]
    best_k0_history: list[float] = [engine.oracle.get_revealed_state_summary()["best_observed_k0"] or 0.0]
    action_type_history: list[str] = ["INIT"]
    global_max_k0 = engine.oracle.global_best_k0

    rng = np.random.default_rng(seed + 100)

    step = 0
    while engine.total_budget_spent < budget_limit and step < 40:
        cand_df = engine.oracle.get_candidate_pool()
        observed_xrd = set(engine.oracle._revealed_xrd.keys())
        observed_prop = set(engine.oracle._revealed_property.keys())

        unobserved_xrd = [cid for cid in cand_df["candidate_id"] if cid not in observed_xrd]
        unobserved_prop = [cid for cid in cand_df["candidate_id"] if cid not in observed_prop]

        if not unobserved_xrd and not unobserved_prop:
            break

        act: ScientificAction | None = None

        if policy_name == "property_only":
            # Always choose property on highest predicted candidate
            if unobserved_prop:
                prop_means, _ = engine.property_model.predict(cand_df[["Au", "Ir", "Rh"]].to_numpy())
                # filter to unobserved prop
                unobs_indices = [i for i, cid in enumerate(cand_df["candidate_id"]) if cid in unobserved_prop]
                best_idx = unobs_indices[int(np.argmax(prop_means[unobs_indices]))]
                best_cid = cand_df["candidate_id"].iloc[best_idx]
                act = ScientificAction(
                    action_id=f"prop_only_{step}_{best_cid}",
                    candidate_id=best_cid,
                    action_type=ExperimentActionType.PROPERTY,
                    estimated_cost=cost_property,
                )
            elif unobserved_xrd:
                act = ScientificAction(
                    action_id=f"xrd_fallback_{step}_{unobserved_xrd[0]}",
                    candidate_id=unobserved_xrd[0],
                    action_type=ExperimentActionType.XRD,
                    estimated_cost=cost_xrd,
                )

        elif policy_name == "random_action":
            # Random candidate and random action
            choices = []
            if unobserved_xrd:
                choices.append(ExperimentActionType.XRD)
            if unobserved_prop:
                choices.append(ExperimentActionType.PROPERTY)
            chosen_type = rng.choice(choices)
            if chosen_type == ExperimentActionType.XRD:
                chosen_cid = rng.choice(unobserved_xrd)
                act = ScientificAction(
                    action_id=f"rand_{step}_{chosen_cid}",
                    candidate_id=chosen_cid,
                    action_type=ExperimentActionType.XRD,
                    estimated_cost=cost_xrd,
                )
            else:
                chosen_cid = rng.choice(unobserved_prop)
                act = ScientificAction(
                    action_id=f"rand_{step}_{chosen_cid}",
                    candidate_id=chosen_cid,
                    action_type=ExperimentActionType.PROPERTY,
                    estimated_cost=cost_property,
                )

        elif policy_name == "fixed_ratio":
            # Fixed 2:1 XRD to Property ratio
            if step % 3 < 2 and unobserved_xrd:
                chosen_cid = rng.choice(unobserved_xrd)
                act = ScientificAction(
                    action_id=f"fixed_{step}_{chosen_cid}",
                    candidate_id=chosen_cid,
                    action_type=ExperimentActionType.XRD,
                    estimated_cost=cost_xrd,
                )
            elif unobserved_prop:
                prop_means, _ = engine.property_model.predict(cand_df[["Au", "Ir", "Rh"]].to_numpy())
                unobs_indices = [i for i, cid in enumerate(cand_df["candidate_id"]) if cid in unobserved_prop]
                best_idx = unobs_indices[int(np.argmax(prop_means[unobs_indices]))]
                best_cid = cand_df["candidate_id"].iloc[best_idx]
                act = ScientificAction(
                    action_id=f"fixed_{step}_{best_cid}",
                    candidate_id=best_cid,
                    action_type=ExperimentActionType.PROPERTY,
                    estimated_cost=cost_property,
                )
            elif unobserved_xrd:
                act = ScientificAction(
                    action_id=f"fixed_xrd_{step}_{unobserved_xrd[0]}",
                    candidate_id=unobserved_xrd[0],
                    action_type=ExperimentActionType.XRD,
                    estimated_cost=cost_xrd,
                )

        elif policy_name == "scientific_action":
            # Our adaptive policy
            rec, _ = engine.propose_next_experiment()
            act = rec.action

        if act is None:
            break

        if engine.total_budget_spent + act.estimated_cost > budget_limit:
            break

        engine.execute_experiment(act)
        step += 1

        rev = engine.oracle.get_revealed_state_summary()
        cost_history.append(engine.total_budget_spent)
        best_k0_history.append(rev["best_observed_k0"] or 0.0)
        action_type_history.append(act.action_type.value)

    # Compute regret
    final_best_k0 = best_k0_history[-1]
    simple_regret = max(0.0, global_max_k0 - final_best_k0)
    relative_regret = simple_regret / (global_max_k0 + 1e-12)

    return {
        "policy_name": policy_name,
        "seed": seed,
        "total_budget_spent": engine.total_budget_spent,
        "num_xrd": len(engine.oracle._revealed_xrd),
        "num_property": len(engine.oracle._revealed_property),
        "final_best_k0": final_best_k0,
        "global_max_k0": global_max_k0,
        "relative_regret": relative_regret,
        "cost_history": cost_history,
        "best_k0_history": best_k0_history,
        "action_type_history": action_type_history,
        "hypothesis_beliefs": {hid: h.belief_score for hid, h in engine.hypothesis_engine.hypotheses.items()},
    }


def run_epistemic_action_benchmark(
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    budget_limit: float = 60.0,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """Runs the 4-policy comparison benchmark across multiple random seeds."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = ["property_only", "random_action", "fixed_ratio", "scientific_action"]
    records = []
    full_runs = []

    for pol in policies:
        for seed in seeds:
            res = run_single_campaign(policy_name=pol, seed=seed, budget_limit=budget_limit)
            full_runs.append(res)
            records.append(
                {
                    "policy": pol,
                    "seed": seed,
                    "budget_spent": res["total_budget_spent"],
                    "num_xrd": res["num_xrd"],
                    "num_property": res["num_property"],
                    "final_best_k0": res["final_best_k0"],
                    "relative_regret": res["relative_regret"],
                    "belief_H2": res["hypothesis_beliefs"].get("H2", 0.0),
                }
            )

    df = pd.DataFrame(records)

    # Save outputs
    summary_path = out_dir / "benchmark_summary.csv"
    df.to_csv(summary_path, index=False)

    runs_path = out_dir / "benchmark_runs.json"
    with open(runs_path, "w", encoding="utf-8") as f:
        json.dump(full_runs, f, indent=2)

    # Generate Markdown Summary
    agg_df = df.groupby("policy").agg(
        mean_best_k0=("final_best_k0", "mean"),
        mean_regret=("relative_regret", "mean"),
        mean_xrd=("num_xrd", "mean"),
        mean_prop=("num_property", "mean"),
        mean_budget=("budget_spent", "mean"),
    ).reset_index()

    report_md = f"""# Au-Ir-Rh Multimodal Epistemic Action Benchmark

**Evaluation Target**: 4 Discovery Policies on Real Au-Ir-Rh Multimodal Dataset  
**Budget Limit**: {budget_limit} Normalized Cost Units  
**Random Seeds**: {list(seeds)}  

## Policy Comparison Summary

| Policy | Mean Best $k^0$ [cm/s] | Mean Rel Regret | Mean XRD Tests | Mean Property Tests | Mean Cost |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in agg_df.iterrows():
        report_md += f"| **{row['policy']}** | {row['mean_best_k0']:.6f} | {row['mean_regret']:.4f} | {row['mean_xrd']:.1f} | {row['mean_prop']:.1f} | {row['mean_budget']:.1f} |\n"

    report_md += "\n*Note: Illustrative development benchmark on 5 seeds demonstrating adaptive multi-action exploration under normalized cost constraints.*\n"

    report_path = out_dir / "benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info(f"Benchmark results saved to: {out_dir.resolve()}")
    return df


if __name__ == "__main__":
    run_epistemic_action_benchmark()
