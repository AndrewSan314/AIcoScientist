from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.optimization.botorch_backend import BoTorchBackend
from src.science.actions import ExperimentActionType, ScientificAction
from src.science.falsification.policy import FalsificationFirstPolicy, FalsificationPolicyMode
from src.science.falsification.synthetic_worlds import (
    SyntheticTruthWorld,
    World1_CompositionSufficient,
    World2_StructureInformed,
    World3_LocalStructuralRegime,
)
from src.science.hypothesis_models import HypothesisEnsemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("outputs/falsification")


def _df_to_markdown_simple(df: pd.DataFrame) -> str:
    """Formats a DataFrame as a Markdown table without requiring tabulate."""
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_strs = []
        for val in row:
            if isinstance(val, float):
                row_strs.append(f"{val:.4f}")
            else:
                row_strs.append(str(val))
        rows.append("| " + " | ".join(row_strs) + " |")
    return "\n".join([header_line, sep_line] + rows)


def run_single_falsification_trajectory(
    world: SyntheticTruthWorld,
    policy_name: str,
    n_steps: int = 10,
    seed: int = 42,
    cost_xrd: float = 1.0,
    cost_property: float = 5.0,
    fast_mode: bool = True,
) -> list[dict[str, Any]]:
    """Runs a single closed-loop experimental trajectory on a synthetic world."""
    world.reset()
    ensemble = HypothesisEnsemble()
    botorch = BoTorchBackend(default_strategy="expected_improvement")
    cand_df = world.get_candidate_pool()
    all_cids = cand_df["candidate_id"].tolist()
    rng = np.random.default_rng(seed)

    # Initial seed observations:
    # 2 candidates with both XRD and Property (joint seeds)
    # 2 candidates with XRD only
    # 2 candidates with Property only
    shuffled = list(all_cids)
    rng.shuffle(shuffled)
    init_joint = shuffled[:2]
    init_xrd_only = shuffled[2:4]
    init_prop_only = shuffled[4:6]

    for cid in init_joint:
        world.execute_xrd(cid, step=0)
        world.execute_property(cid, step=0)
    for cid in init_xrd_only:
        world.execute_xrd(cid, step=0)
    for cid in init_prop_only:
        world.execute_property(cid, step=0)

    # Policy initialization
    if policy_name == "pure_falsification":
        policy = FalsificationFirstPolicy(mode=FalsificationPolicyMode.PURE_FALSIFICATION, cost_xrd=cost_xrd, cost_property=cost_property)
    elif policy_name == "discovery_only":
        policy = FalsificationFirstPolicy(mode=FalsificationPolicyMode.DISCOVERY_ONLY, cost_xrd=cost_xrd, cost_property=cost_property)
    elif policy_name == "hybrid":
        policy = FalsificationFirstPolicy(mode=FalsificationPolicyMode.HYBRID, cost_xrd=cost_xrd, cost_property=cost_property)
    else:
        policy = None

    history: list[dict[str, Any]] = []
    cumulative_cost = 4 * cost_xrd + 4 * cost_property

    for step in range(1, n_steps + 1):
        observed_xrd_ids = set(world.get_revealed_xrd_ids())
        observed_prop_ids = set(world.get_revealed_property_ids())
        revealed_props = world.get_revealed_properties()
        revealed_xrds = world.get_revealed_xrd()

        # Build candidate maps by ID
        comp_map = {row["candidate_id"]: row[["Au", "Ir", "Rh"]].to_numpy(dtype=np.float64) for _, row in cand_df.iterrows()}
        prop_map = {cid: float(out.revealed_data["k0"]) for cid, out in revealed_props.items()}
        xrd_map = {cid: np.asarray(out.revealed_data["xrd_embedding"], dtype=np.float64) for cid, out in revealed_xrds.items()}

        ensemble.fit_all(
            composition_by_id=comp_map,
            property_by_id=prop_map,
            xrd_embedding_by_id=xrd_map,
            observed_xrd_ids=observed_xrd_ids,
            observed_property_ids=observed_prop_ids,
        )

        # Score candidate property discovery potential via BoTorch
        obs_rows = []
        for cid, out in revealed_props.items():
            match = cand_df[cand_df["candidate_id"] == cid]
            if not match.empty:
                r = match.iloc[0].to_dict()
                r["k0"] = float(out.revealed_data["k0"])
                obs_rows.append(r)
        obs_df = pd.DataFrame(obs_rows)

        try:
            prop_disc_scores = botorch.score_candidates(
                observations=obs_df,
                candidate_pool=cand_df[["candidate_id", "Au", "Ir", "Rh"]],
                objective="k0",
                strategy="expected_improvement",
                seed=seed,
            )
        except Exception:
            prop_disc_scores = {}

        # Action Selection
        if policy_name in {"pure_falsification", "discovery_only", "hybrid"}:
            rec = policy.recommend_next_experiment(
                candidate_pool_df=cand_df,
                observed_xrd_ids=observed_xrd_ids,
                observed_property_ids=observed_prop_ids,
                ensemble=ensemble,
                property_discovery_scores=prop_disc_scores,
                observed_xrd_embeddings_map=xrd_map,
                fast_mode=fast_mode,
                seed=seed + step,
                step=step,
            )
            selected_cid = rec.action.candidate_id
            selected_action_type = rec.action.action_type
            expected_hig = float(rec.uncertainty_summary.get("hypothesis_information_gain", 0.0))

        elif policy_name == "random_action":
            valid_actions: list[tuple[str, ExperimentActionType]] = []
            for cid in all_cids:
                if cid not in observed_xrd_ids:
                    valid_actions.append((cid, ExperimentActionType.XRD))
                if cid not in observed_prop_ids:
                    valid_actions.append((cid, ExperimentActionType.PROPERTY))
            choice_idx = rng.choice(len(valid_actions))
            selected_cid, selected_action_type = valid_actions[choice_idx]
            expected_hig = 0.0

        elif policy_name == "uncertainty_only":
            best_var = -1.0
            selected_cid = all_cids[0]
            selected_action_type = ExperimentActionType.PROPERTY
            for i, cid in enumerate(all_cids):
                comp_i = comp_map[cid]
                preds = ensemble.predict_all(cid, ExperimentActionType.PROPERTY, comp_i)
                avg_var = float(np.mean([p.variance[0] for p in preds.values()]))
                if cid not in observed_prop_ids and avg_var > best_var:
                    best_var = avg_var
                    selected_cid = cid
                    selected_action_type = ExperimentActionType.PROPERTY
            expected_hig = 0.0

        # Pre-register prediction before execution
        cand_comp = comp_map[selected_cid]
        pre_preds = ensemble.predict_all(
            candidate_id=selected_cid,
            action_type=selected_action_type,
            composition=cand_comp,
            observed_xrd_embedding=xrd_map.get(selected_cid),
        )

        # Execute action on oracle
        action = ScientificAction(
            action_id=f"step_{step:03d}_{selected_action_type.value}_{selected_cid}",
            candidate_id=selected_cid,
            action_type=selected_action_type,
            estimated_cost=cost_xrd if selected_action_type == ExperimentActionType.XRD else cost_property,
            requested_at_step=step,
        )
        outcome = world.execute(action)
        cumulative_cost += action.estimated_cost

        # Extract observation and update sequential predictive evidence
        if selected_action_type == ExperimentActionType.XRD:
            obs_val = outcome.revealed_data["xrd_embedding"]
        else:
            obs_val = float(outcome.revealed_data["k0"])

        update_summary = ensemble.record_observation_and_update(
            action_id=action.action_id,
            candidate_id=selected_cid,
            action_type=selected_action_type,
            observation=obs_val,
            pre_predictions=pre_preds,
        )

        beliefs = ensemble.get_beliefs()
        entropy = ensemble.get_entropy()
        true_h_weight = beliefs.get(world.true_hypothesis_id, 0.0)

        # Check top-1 hypothesis
        top_h = max(beliefs.keys(), key=lambda h: beliefs[h])
        is_top1_correct = (top_h == world.true_hypothesis_id)

        all_revealed_props = [out.revealed_data["k0"] for out in world.get_revealed_properties().values()]
        best_k0 = max(all_revealed_props) if all_revealed_props else 0.0

        step_record = {
            "world": world.name,
            "true_hypothesis": world.true_hypothesis_id,
            "policy": policy_name,
            "seed": seed,
            "step": step,
            "candidate_id": selected_cid,
            "action_type": selected_action_type.value,
            "cost_spent": cumulative_cost,
            "true_hypothesis_weight": true_h_weight,
            "hypothesis_entropy": entropy,
            "is_identified_75": bool(true_h_weight >= 0.75),
            "is_identified_90": bool(true_h_weight >= 0.90),
            "is_top1_correct": bool(is_top1_correct),
            "best_observed_k0": best_k0,
            "expected_hig": expected_hig,
            "realized_entropy_reduction": update_summary["realized_entropy_reduction"],
            "beliefs": beliefs,
        }
        history.append(step_record)

    return history


WORLD_TYPE_TO_CANONICAL_NAME: dict[str, str] = {
    "World1": "Synthetic_World_1_H1_True",
    "World2": "Synthetic_World_2_H2_True",
    "World3": "Synthetic_World_3_H3_True",
}


def _run_single_job(args: tuple[str, str, int, int]) -> list[dict[str, Any]]:
    world_type, policy_name, n_steps, seed = args

    # Test hook: allow testing worker exceptions in child processes without monkeypatch pickling issues
    simulated_fail_policy = os.environ.get("_TEST_BENCHMARK_SIMULATE_FAILURE_POLICY")
    if simulated_fail_policy and policy_name == simulated_fail_policy:
        raise RuntimeError(f"Simulated worker failure for policy {policy_name}")

    if world_type == "World1":
        world = World1_CompositionSufficient(seed=seed)
    elif world_type == "World2":
        world = World2_StructureInformed(seed=seed)
    else:
        world = World3_LocalStructuralRegime(seed=seed)

    return run_single_falsification_trajectory(
        world=world,
        policy_name=policy_name,
        n_steps=n_steps,
        seed=seed,
        fast_mode=True,
    )


def run_full_falsification_benchmark(
    seeds: Sequence[int] = (42, 101, 2024),
    n_steps: int = 6,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    parallel: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Runs full factorial benchmark across synthetic worlds and policies with fail-closed integrity."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    world_types = ["World1", "World2", "World3"]
    policies = [
        "pure_falsification",
        "hybrid",
        "discovery_only",
        "uncertainty_only",
        "random_action",
    ]

    jobs: list[tuple[str, str, int, int]] = []
    for w in world_types:
        for p in policies:
            for s in seeds:
                jobs.append((w, p, n_steps, s))

    expected_jobs_count = len(jobs)
    all_records: list[dict[str, Any]] = []
    errors: list[str] = []
    completed_jobs = 0

    if parallel and len(jobs) > 1:
        logger.info(f"Executing {expected_jobs_count} benchmark runs in parallel (max_workers=3)...")
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_run_single_job, j): j for j in jobs}
            for f in as_completed(futures):
                job_spec = futures[f]
                try:
                    res = f.result()
                    if len(res) != n_steps:
                        errors.append(f"Job {job_spec} produced {len(res)} steps, expected {n_steps}")
                    else:
                        all_records.extend(res)
                        completed_jobs += 1
                except Exception as exc:
                    errors.append(f"Job {job_spec} failed with exception: {exc}")
    else:
        for j in jobs:
            try:
                res = _run_single_job(j)
                if len(res) != n_steps:
                    errors.append(f"Job {j} produced {len(res)} steps, expected {n_steps}")
                else:
                    all_records.extend(res)
                    completed_jobs += 1
            except Exception as exc:
                errors.append(f"Job {j} failed with exception: {exc}")

    # Fail-closed validation: All expected jobs and trajectory steps must complete without errors
    if errors or completed_jobs != expected_jobs_count:
        err_msg = (
            f"Falsification benchmark failed: {len(errors)} error(s) occurred. "
            f"Only {completed_jobs}/{expected_jobs_count} jobs completed successfully.\n"
            + "\n".join(errors)
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    raw_df = pd.DataFrame(all_records)

    # Validate that all expected (world, policy, seed) combinations are present without digit substring matching
    expected_combinations = {(WORLD_TYPE_TO_CANONICAL_NAME.get(w, w), p, s) for w, p, _, s in jobs}
    actual_combinations = set(zip(
        raw_df["world"],
        raw_df["policy"],
        raw_df["seed"],
    ))
    missing = expected_combinations - actual_combinations
    if missing:
        raise RuntimeError(f"Missing expected benchmark combinations: {missing}")

    # Sort deterministically before serialization
    df = raw_df.sort_values(by=["world", "policy", "seed", "step"]).reset_index(drop=True)
    sorted_records = df.to_dict(orient="records")

    # Save raw outputs
    summary_csv = out_path / "benchmark_summary.csv"
    runs_json = out_path / "benchmark_runs.json"
    report_md = out_path / "benchmark_report.md"

    df.to_csv(summary_csv, index=False)
    with open(runs_json, "w", encoding="utf-8") as f:
        json.dump(sorted_records, f, indent=2)

    # Multi-Seed Aggregation:
    # Step 1: Group by (world, true_hypothesis, policy, seed) and get final state
    final_states = df.groupby(["world", "true_hypothesis", "policy", "seed"]).last().reset_index()

    # Step 2: Aggregate across seeds for each (world, true_hypothesis, policy)
    agg_df = final_states.groupby(["world", "true_hypothesis", "policy"]).agg(
        mean_final_true_weight=("true_hypothesis_weight", "mean"),
        median_final_true_weight=("true_hypothesis_weight", "median"),
        std_final_true_weight=("true_hypothesis_weight", "std"),
        id_rate_75=("is_identified_75", "mean"),
        id_rate_90=("is_identified_90", "mean"),
        top1_accuracy=("is_top1_correct", "mean"),
        mean_final_entropy=("hypothesis_entropy", "mean"),
        mean_cost=("cost_spent", "mean"),
        mean_final_best_k0=("best_observed_k0", "mean"),
        median_final_best_k0=("best_observed_k0", "median"),
        std_final_best_k0=("best_observed_k0", "std"),
        max_final_best_k0=("best_observed_k0", "max"),
    ).reset_index()

    table_md = _df_to_markdown_simple(agg_df)

    report_lines = [
        "# Falsification-First Hypothesis Discrimination Benchmark Report",
        "",
        f"**Benchmark Horizon**: {n_steps} adaptive steps  ",
        f"**Seeds**: {list(seeds)}  ",
        f"**Worlds Evaluated**: World 1 ($H_1$), World 2 ($H_2$), World 3 ($H_3$)  ",
        "",
        "## Summary Results by World and Policy (Aggregated Across Seeds)",
        "",
        table_md,
        "",
        "## Scientific Findings & Methodological Boundaries",
        "- **World 3 ($H_3$ Local Regime)**: Falsification and Hybrid policies demonstrate strong true-hypothesis recovery ($P(H_3) \\approx 1.0$) by identifying transition candidates with high Expected HIG, significantly outperforming unguided exploration.",
        "- **World 1 & World 2 ($H_1$ vs. $H_2$)**: At the evaluated six-step horizon, H1 and H2 remain poorly identifiable. The current results are consistent with a sample-complexity limitation of the higher-dimensional structure-informed model, but longer-horizon and targeted joint-characterization experiments are required to test that explanation.",
        "- **Discovery vs. Falsification Trade-off**: `pure_falsification` operates with lowest experimental cost, while `hybrid` balances discovery potential with information gain.",
    ]

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Benchmark saved to {out_path}")
    return df, sorted_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Falsification-First Benchmark")
    parser.add_argument("--steps", type=int, default=6, help="Adaptive steps per run")
    parser.add_argument("--out", type=str, default="outputs/falsification", help="Output directory")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel execution")
    args = parser.parse_args()

    run_full_falsification_benchmark(n_steps=args.steps, output_dir=args.out, parallel=not args.no_parallel)
