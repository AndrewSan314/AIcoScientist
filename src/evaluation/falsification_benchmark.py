from __future__ import annotations

import argparse
import json
import logging
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
) -> list[dict[str, Any]]:
    """Runs a single closed-loop experimental trajectory on a synthetic world."""
    world.reset()
    ensemble = HypothesisEnsemble()
    botorch = BoTorchBackend(default_strategy="expected_improvement")
    cand_df = world.get_candidate_pool()
    all_cids = cand_df["candidate_id"].tolist()
    rng = np.random.default_rng(seed)

    # Initial seed observations (2 XRD, 2 Property) for baseline context without fake evidence
    shuffled = list(all_cids)
    rng.shuffle(shuffled)
    init_xrd = shuffled[:2]
    init_prop = shuffled[2:4]

    for cid in init_xrd:
        world.execute_xrd(cid, step=0)
    for cid in init_prop:
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
    cumulative_cost = 2 * cost_xrd + 2 * cost_property

    for step in range(1, n_steps + 1):
        observed_xrd_ids = set(world.get_revealed_xrd_ids())
        observed_prop_ids = set(world.get_revealed_property_ids())
        revealed_props = world.get_revealed_properties()
        revealed_xrds = world.get_revealed_xrd()

        # 1. Fit hypothesis models strictly on revealed data
        rev_prop_cids = list(revealed_props.keys())
        rev_xrd_cids = list(revealed_xrds.keys())

        comps = cand_df[cand_df["candidate_id"].isin(rev_prop_cids)][["Au", "Ir", "Rh"]].to_numpy()
        props = np.array([revealed_props[cid].revealed_data["k0"] for cid in rev_prop_cids])

        xrd_comps = cand_df[cand_df["candidate_id"].isin(rev_xrd_cids)][["Au", "Ir", "Rh"]].to_numpy()
        xrd_embs = np.array([revealed_xrds[cid].revealed_data["xrd_embedding"] for cid in rev_xrd_cids])

        xrd_embs_map = {cid: revealed_xrds[cid].revealed_data["xrd_embedding"] for cid in rev_xrd_cids}

        ensemble.fit_all(
            compositions=comps,
            property_targets=props,
            xrd_embeddings=xrd_embs,
            xrd_compositions=xrd_comps,
            candidate_ids=rev_prop_cids,
            observed_xrd_ids=observed_xrd_ids,
            observed_property_ids=observed_prop_ids,
        )

        # 2. Score candidate property discovery potential via BoTorch
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

        # 3. Action Selection
        if policy_name in {"pure_falsification", "discovery_only", "hybrid"}:
            rec = policy.recommend_next_experiment(
                candidate_pool_df=cand_df,
                observed_xrd_ids=observed_xrd_ids,
                observed_property_ids=observed_prop_ids,
                ensemble=ensemble,
                property_discovery_scores=prop_disc_scores,
                observed_xrd_embeddings_map=xrd_embs_map,
                fast_mode=False,
                seed=seed + step,
                step=step,
            )
            selected_cid = rec.action.candidate_id
            selected_action_type = rec.action.action_type
            expected_hig = float(rec.uncertainty_summary.get("hypothesis_information_gain", 0.0))

        elif policy_name == "random_action":
            # Pick random valid action from unobserved
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
            # Select point with highest epistemic variance
            best_var = -1.0
            selected_cid = all_cids[0]
            selected_action_type = ExperimentActionType.PROPERTY
            for i, cid in enumerate(all_cids):
                comp_i = cand_df[cand_df["candidate_id"] == cid][["Au", "Ir", "Rh"]].iloc[0].to_numpy()
                preds = ensemble.predict_all(cid, ExperimentActionType.PROPERTY, comp_i)
                avg_var = float(np.mean([p.variance[0] for p in preds.values()]))
                if cid not in observed_prop_ids and avg_var > best_var:
                    best_var = avg_var
                    selected_cid = cid
                    selected_action_type = ExperimentActionType.PROPERTY
            expected_hig = 0.0

        elif policy_name == "fixed_schedule":
            # Alternate XRD and Property
            target_type = ExperimentActionType.XRD if step % 2 == 1 else ExperimentActionType.PROPERTY
            unseen = [c for c in all_cids if (c not in observed_xrd_ids if target_type == ExperimentActionType.XRD else c not in observed_prop_ids)]
            selected_cid = unseen[rng.choice(len(unseen))]
            selected_action_type = target_type
            expected_hig = 0.0

        # 4. Pre-register prediction before execution
        cand_comp = cand_df[cand_df["candidate_id"] == selected_cid][["Au", "Ir", "Rh"]].iloc[0].to_numpy()
        pre_preds = ensemble.predict_all(
            candidate_id=selected_cid,
            action_type=selected_action_type,
            composition=cand_comp,
            observed_xrd_embedding=xrd_embs_map.get(selected_cid),
        )

        # 5. Execute action on oracle
        action = ScientificAction(
            action_id=f"step_{step:03d}_{selected_action_type.value}_{selected_cid}",
            candidate_id=selected_cid,
            action_type=selected_action_type,
            estimated_cost=cost_xrd if selected_action_type == ExperimentActionType.XRD else cost_property,
            requested_at_step=step,
        )
        outcome = world.execute(action)
        cumulative_cost += action.estimated_cost

        # 6. Extract observation and update sequential predictive evidence
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

        # Current best property
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
            "best_observed_k0": best_k0,
            "expected_hig": expected_hig,
            "realized_entropy_reduction": update_summary["realized_entropy_reduction"],
            "beliefs": beliefs,
        }
        history.append(step_record)

    return history


def run_full_falsification_benchmark(
    seeds: Sequence[int] = (42, 101, 2024),
    n_steps: int = 8,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Runs full factorial benchmark across synthetic worlds and policies."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    worlds: list[SyntheticTruthWorld] = [
        World1_CompositionSufficient(),
        World2_StructureInformed(),
        World3_LocalStructuralRegime(),
    ]

    policies = [
        "pure_falsification",
        "hybrid",
        "discovery_only",
        "uncertainty_only",
        "random_action",
    ]

    all_records: list[dict[str, Any]] = []

    for world in worlds:
        for policy in policies:
            for seed in seeds:
                logger.info(f"Running World='{world.name}', Policy='{policy}', Seed={seed}")
                traj = run_single_falsification_trajectory(
                    world=world,
                    policy_name=policy,
                    n_steps=n_steps,
                    seed=seed,
                )
                all_records.extend(traj)

    df = pd.DataFrame(all_records)

    # Save outputs
    summary_csv = out_path / "benchmark_summary.csv"
    runs_json = out_path / "benchmark_runs.json"
    report_md = out_path / "benchmark_report.md"

    df.to_csv(summary_csv, index=False)
    with open(runs_json, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    # Generate Markdown Summary Report
    grouped = df.groupby(["world", "true_hypothesis", "policy"]).agg(
        final_true_weight=("true_hypothesis_weight", "last"),
        final_entropy=("hypothesis_entropy", "last"),
        best_k0=("best_observed_k0", "max"),
        mean_cost=("cost_spent", "last"),
    ).reset_index()

    table_md = _df_to_markdown_simple(grouped)

    report_lines = [
        "# Falsification-First Hypothesis Discrimination Benchmark Report",
        "",
        f"**Benchmark Horizon**: {n_steps} adaptive steps  ",
        f"**Seeds**: {list(seeds)}  ",
        f"**Worlds Evaluated**: World 1 ($H_1$), World 2 ($H_2$), World 3 ($H_3$)  ",
        "",
        "## Summary Results by World and Policy",
        "",
        table_md,
        "",
        "## Scientific Interpretation",
        "- **Pure Falsification (HIG)** preferentially selects experiments maximizing hypothesis entropy reduction.",
        "- **Hybrid Policy** balances hypothesis discrimination with property discovery.",
        "- **Discovery Only (BoTorch BO)** finds high property values rapidly but allocates zero budget to structural characterization.",
    ]

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Benchmark saved to {out_path}")
    return df, all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Falsification-First Benchmark")
    parser.add_argument("--steps", type=int, default=6, help="Adaptive steps per run")
    parser.add_argument("--out", type=str, default="outputs/falsification", help="Output directory")
    args = parser.parse_args()

    run_full_falsification_benchmark(n_steps=args.steps, output_dir=args.out)
