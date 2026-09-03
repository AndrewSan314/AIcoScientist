from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.domains.electrolyte.adapter import ElectrolyteDomainAdapter
from src.domains.electrolyte.config import (
    ELECTROLYTE_DOMAIN_CONFIG,
    ELECTROLYTE_SOLVENT_FEATURES,
)
from src.domains.electrolyte.data import (
    DEFAULT_COMPATIBLE_DERIVED_PATH,
    load_derived_historical_outcomes,
)
from src.domains.electrolyte.screening import (
    benchmark_large_pool_screening,
    screen_large_pool_candidates,
)
from src.optimization.botorch_backend import BoTorchBackend
from src.science.actions import ScientificAction
from src.science.decision_engine import ScientificDecisionEngine
from src.science.falsification.policy import (
    FalsificationFirstPolicy,
    FalsificationPolicyMode,
)

logger = logging.getLogger(__name__)


@dataclass
class PolicyRunResult:
    policy_name: str
    seed: int
    bootstrap_candidate_ids: list[str]
    bootstrap_best: float
    autonomous_actions: list[str]
    autonomous_observations: list[float]
    best_so_far_curve: list[float]
    best_autonomous_found: float
    improvement_over_bootstrap: float
    area_under_best_curve: float
    mean_selected_capacity: float
    top_decile_hits: int
    near_zero_query_count: int
    initial_entropy: float
    final_entropy: float
    cumulative_hig_nats: float
    cumulative_hig_normalized: float
    final_beliefs: dict[str, float]
    steps_count: int
    elapsed_time_sec: float
    cumulative_hig: float = 0.0
    step_diagnostics: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PolicyAggregateSummary:
    policy_name: str
    num_seeds: int
    bootstrap_best: float
    objective_saturated: bool
    best_found_mean: float
    best_found_std: float
    improvement_mean: float
    improvement_std: float
    auc_mean: float
    auc_std: float
    top_decile_hit_rate: float
    near_zero_rate: float
    mean_hig_nats: float
    mean_hig_normalized: float
    mean_entropy_reduction: float
    runtime_sec_mean: float
    mean_hig: float = 0.0


def evaluate_historical_policy(
    policy_name: str,
    derived_outcomes_path: str = DEFAULT_COMPATIBLE_DERIVED_PATH,
    seed: int = 42,
    max_steps: int = 15,
    top_decile_threshold: float = 0.70,
) -> PolicyRunResult:
    """Executes a retrospective finite historical replay run for a specific policy and seed."""
    t0 = time.perf_counter()
    adapter = ElectrolyteDomainAdapter(derived_outcomes_path=derived_outcomes_path)

    # Configure policy parameters
    is_external = False
    is_random = (policy_name == "RANDOM")
    is_botorch = policy_name in ("BOTORCH_EI_DIRECT", "BOTORCH_GPUCB_DIRECT")

    if policy_name == "PURE_FALSIFICATION":
        mode = FalsificationPolicyMode.PURE_FALSIFICATION
        w_hig = 1.0
        w_disc = 0.0
    elif policy_name == "DISCOVERY_ONLY":
        mode = FalsificationPolicyMode.DISCOVERY_ONLY
        w_hig = 0.0
        w_disc = 1.0
    else:
        # Phase 4 mandate: Primary Hybrid uses core defaults w_hig = 1.0, w_disc = 0.8, w_cost = 0.0
        mode = FalsificationPolicyMode.HYBRID
        w_hig = 1.0
        w_disc = 0.8

    policy = FalsificationFirstPolicy(
        mode=mode,
        w_hig=w_hig,
        w_disc=w_disc,
        w_cost=0.0,
    )

    optimizer_backend = BoTorchBackend()

    engine = ScientificDecisionEngine(
        domain=adapter,
        policy=policy,
        optimizer_backend=optimizer_backend,
        seed=seed,
    )

    # Initialize strictly with Batch-0 compatible seed actions (N=3)
    init_actions = adapter.get_default_initial_actions(n_seed=3, seed=seed)
    engine.initialize(init_actions)

    init_beliefs = engine.ensemble.get_beliefs()
    init_entropy = engine.ensemble.get_entropy()

    # Measure bootstrap values
    bootstrap_vals = [
        float(engine.observations_by_modality["CAPACITY_TEST"][act.candidate_id])
        for act in init_actions
        if act.candidate_id in engine.observations_by_modality.get("CAPACITY_TEST", {})
    ]
    bootstrap_best = max(bootstrap_vals) if bootstrap_vals else 0.0
    bootstrap_cids = [act.candidate_id for act in init_actions]

    autonomous_actions = []
    autonomous_obs = []
    best_curve = [bootstrap_best]
    running_best = bootstrap_best
    cumulative_hig_nats = 0.0
    cumulative_hig_norm = 0.0
    step_diagnostics: list[dict[str, Any]] = []

    rng = np.random.default_rng(seed)
    f_cols = list(adapter.get_config().candidate_features)

    # Run autonomous exploration steps
    for step in range(max_steps):
        valid_actions = adapter.list_valid_actions(engine.get_state())
        if not valid_actions:
            break

        beliefs_before = dict(engine.ensemble.get_beliefs())
        rec = None

        if is_random:
            chosen_action = rng.choice(valid_actions)
            disc_score = 0.0
            outcome = engine.execute_external_action(chosen_action, metadata={"policy": "RANDOM"})
            c_val = float(outcome.revealed_data["C_norm_20"])
            # Evaluate post-hoc HIG in nats
            comp = np.array([adapter._feature_cache[chosen_action.candidate_id][f] for f in f_cols], dtype=np.float64)
            eval_res = policy.hig_estimator.evaluate_action_discrimination(
                candidate_id=chosen_action.candidate_id,
                action_type=chosen_action.action_type,
                composition=comp,
                ensemble=engine.ensemble,
                seed=seed,
            )
            raw_hig = float(eval_res.hypothesis_information_gain)
            norm_hig = float(np.clip(raw_hig / np.log(3.0), 0.0, 1.0))
            tot_val = 0.0

        elif is_botorch:
            strat = "expected_improvement" if policy_name == "BOTORCH_EI_DIRECT" else "gp_ucb"
            obs_dict = engine.observations_by_modality.get("CAPACITY_TEST", {})
            obs_rows = []
            for cid, val in obs_dict.items():
                r = {"candidate_id": cid, "C_norm_20": float(val)}
                for f in f_cols:
                    r[f] = adapter._feature_cache[cid][f]
                obs_rows.append(r)
            obs_df = pd.DataFrame(obs_rows)

            valid_cids = [a.candidate_id for a in valid_actions]
            cand_rows = []
            for cid in valid_cids:
                r = {"candidate_id": cid}
                for f in f_cols:
                    r[f] = adapter._feature_cache[cid][f]
                cand_rows.append(r)
            cand_pool_df = pd.DataFrame(cand_rows)

            proposals = optimizer_backend.propose(
                observations=obs_df,
                candidate_pool=cand_pool_df,
                objective="C_norm_20",
                feature_columns=f_cols,
                candidate_id_column="candidate_id",
                strategy=strat,
                seed=seed,
            )
            top_cid = proposals[0].candidate_id
            disc_score = float(proposals[0].acquisition_value)
            chosen_action = next(a for a in valid_actions if a.candidate_id == top_cid)

            outcome = engine.execute_external_action(chosen_action, metadata={"policy": policy_name})
            c_val = float(outcome.revealed_data["C_norm_20"])

            # Evaluate post-hoc HIG in nats
            comp = np.array([adapter._feature_cache[chosen_action.candidate_id][f] for f in f_cols], dtype=np.float64)
            eval_res = policy.hig_estimator.evaluate_action_discrimination(
                candidate_id=chosen_action.candidate_id,
                action_type=chosen_action.action_type,
                composition=comp,
                ensemble=engine.ensemble,
                seed=seed,
            )
            raw_hig = float(eval_res.hypothesis_information_gain)
            norm_hig = float(np.clip(raw_hig / np.log(3.0), 0.0, 1.0))
            tot_val = disc_score

        else:
            rec = engine.propose_next_experiment()
            chosen_action = rec.action
            outcome = engine.execute_recommendation(rec)
            c_val = float(outcome.revealed_data["C_norm_20"])
            raw_hig = float(rec.uncertainty_summary.get("raw_hig_nats", rec.scientific_information_value))
            norm_hig = float(rec.scientific_information_value)
            disc_score = float(rec.discovery_value)
            tot_val = float(rec.total_value)

        cumulative_hig_nats += raw_hig
        cumulative_hig_norm += norm_hig

        beliefs_after = dict(engine.ensemble.get_beliefs())
        max_shift = float(max(abs(beliefs_after[k] - beliefs_before[k]) for k in beliefs_before))

        step_diagnostics.append({
            "step": step,
            "candidate_id": chosen_action.candidate_id,
            "revealed_capacity": round(c_val, 4),
            "predicted_discovery_score": round(disc_score, 4),
            "raw_hig_nats": round(raw_hig, 4),
            "normalized_hig": round(norm_hig, 4),
            "total_value": round(tot_val, 4),
            "beliefs_before": {k: round(v, 4) for k, v in beliefs_before.items()},
            "beliefs_after": {k: round(v, 4) for k, v in beliefs_after.items()},
            "max_belief_shift": round(max_shift, 4),
        })

        autonomous_actions.append(chosen_action.candidate_id)
        autonomous_obs.append(c_val)
        if c_val > running_best:
            running_best = c_val
        best_curve.append(running_best)

    elapsed = time.perf_counter() - t0
    final_entropy = engine.ensemble.get_entropy()
    final_beliefs = engine.ensemble.get_beliefs()

    best_autonomous = max(autonomous_obs) if autonomous_obs else bootstrap_best
    improvement = max(0.0, best_autonomous - bootstrap_best)
    if len(best_curve) > 1:
        auc = float(np.sum(0.5 * (np.array(best_curve[:-1]) + np.array(best_curve[1:]))))
    else:
        auc = float(best_curve[0])
    mean_sel = float(np.mean(autonomous_obs)) if autonomous_obs else 0.0
    top_hits = int(sum(1 for v in autonomous_obs if v >= top_decile_threshold))
    near_zeros = int(sum(1 for v in autonomous_obs if v <= 0.05))

    return PolicyRunResult(
        policy_name=policy_name,
        seed=seed,
        bootstrap_candidate_ids=bootstrap_cids,
        bootstrap_best=round(bootstrap_best, 4),
        autonomous_actions=autonomous_actions,
        autonomous_observations=[round(v, 4) for v in autonomous_obs],
        best_so_far_curve=[round(v, 4) for v in best_curve],
        best_autonomous_found=round(best_autonomous, 4),
        improvement_over_bootstrap=round(improvement, 4),
        area_under_best_curve=round(auc, 4),
        mean_selected_capacity=round(mean_sel, 4),
        top_decile_hits=top_hits,
        near_zero_query_count=near_zeros,
        initial_entropy=round(init_entropy, 4),
        final_entropy=round(final_entropy, 4),
        cumulative_hig_nats=round(cumulative_hig_nats, 4),
        cumulative_hig_normalized=round(cumulative_hig_norm, 4),
        final_beliefs={k: round(v, 4) for k, v in final_beliefs.items()},
        steps_count=len(autonomous_obs),
        elapsed_time_sec=round(elapsed, 4),
        cumulative_hig=round(cumulative_hig_nats, 4),
        step_diagnostics=step_diagnostics,
    )


def run_comprehensive_historical_benchmark(
    derived_outcomes_path: str = DEFAULT_COMPATIBLE_DERIVED_PATH,
    seeds: Sequence[int] = (42, 101, 2024, 314, 7),
    policies: Sequence[str] = (
        "RANDOM",
        "DISCOVERY_ONLY",
        "PURE_FALSIFICATION",
        "HYBRID",
        "BOTORCH_EI_DIRECT",
        "BOTORCH_GPUCB_DIRECT",
    ),
    max_steps: int = 15,
) -> dict[str, Any]:
    """Runs multi-seed evaluation across all candidate policies on the retrospective finite pool."""
    df_historical = load_derived_historical_outcomes(derived_outcomes_path)
    global_max = float(df_historical["C_norm_20"].max())
    p90_threshold = float(np.percentile(df_historical["C_norm_20"], 90))

    b0_pool = df_historical[df_historical["batch"] == 0]
    b0_best = float(b0_pool["C_norm_20"].max()) if len(b0_pool) > 0 else 0.0
    saturation_ratio = b0_best / global_max if global_max > 0 else 0.0
    is_saturated = bool(saturation_ratio >= 0.95)

    all_results: dict[str, list[PolicyRunResult]] = {p: [] for p in policies}

    for p in policies:
        for s in seeds:
            res = evaluate_historical_policy(
                policy_name=p,
                derived_outcomes_path=derived_outcomes_path,
                seed=s,
                max_steps=max_steps,
                top_decile_threshold=p90_threshold,
            )
            all_results[p].append(res)

    summaries = []
    for p in policies:
        runs = all_results[p]
        best_founds = [r.best_autonomous_found for r in runs]
        improvements = [r.improvement_over_bootstrap for r in runs]
        aucs = [r.area_under_best_curve for r in runs]
        top_hits = [r.top_decile_hits / max(r.steps_count, 1) for r in runs]
        near_zeros = [r.near_zero_query_count / max(r.steps_count, 1) for r in runs]
        higs_nats = [r.cumulative_hig_nats for r in runs]
        higs_norm = [r.cumulative_hig_normalized for r in runs]
        ent_reds = [r.initial_entropy - r.final_entropy for r in runs]
        times = [r.elapsed_time_sec for r in runs]

        summaries.append(
            PolicyAggregateSummary(
                policy_name=p,
                num_seeds=len(runs),
                bootstrap_best=round(b0_best, 4),
                objective_saturated=is_saturated,
                best_found_mean=round(float(np.mean(best_founds)), 4),
                best_found_std=round(float(np.std(best_founds)), 4),
                improvement_mean=round(float(np.mean(improvements)), 4),
                improvement_std=round(float(np.std(improvements)), 4),
                auc_mean=round(float(np.mean(aucs)), 4),
                auc_std=round(float(np.std(aucs)), 4),
                top_decile_hit_rate=round(float(np.mean(top_hits)), 4),
                near_zero_rate=round(float(np.mean(near_zeros)), 4),
                mean_hig_nats=round(float(np.mean(higs_nats)), 4),
                mean_hig_normalized=round(float(np.mean(higs_norm)), 4),
                mean_entropy_reduction=round(float(np.mean(ent_reds)), 4),
                runtime_sec_mean=round(float(np.mean(times)), 4),
                mean_hig=round(float(np.mean(higs_nats)), 4),
            )
        )

    # Search for natural "Wow" scenario under strict preregistered divergence criteria
    natural_wow = find_natural_wow_scenario(all_results.get("DISCOVERY_ONLY", []), all_results.get("HYBRID", []))

    return {
        "benchmark_metadata": {
            "title": "Retrospective Finite Historical Label-Pool Replay",
            "historical_pool_size": len(df_historical),
            "global_pool_maximum": round(global_max, 4),
            "top_decile_p90_threshold": round(p90_threshold, 4),
            "bootstrap_seed_count": 3,
            "bootstrap_best_capacity": round(b0_best, 4),
            "objective_saturation_status": is_saturated,
            "saturation_ratio": round(saturation_ratio, 4),
            "evaluated_seeds": list(seeds),
            "max_autonomous_steps": max_steps,
        },
        "policy_summaries": [asdict(s) for s in summaries],
        "detailed_runs": {p: [asdict(r) for r in runs] for p, runs in all_results.items()},
        "natural_wow_scenario": natural_wow,
    }


def find_natural_wow_scenario(
    discovery_runs: list[PolicyRunResult],
    hybrid_runs: list[PolicyRunResult],
) -> dict[str, Any]:
    """Deterministically identifies whether a natural Wow scenario exists under strict preregistered divergence criteria.

    Preregistered Criteria (Phase 10):
    1. Candidate divergence: c_disc != c_hyb at step t
    2. Discovery priority: EI(c_disc) > EI(c_hyb)
    3. Falsification priority: HIG_nats(c_hyb) > HIG_nats(c_disc)
    4. Net scientific valuation: V_hyb(c_hyb) > V_hyb(c_disc)
    5. Material epistemic shift: max(|P_post - P_prior|) >= 0.01

    If no candidate/step across all seeds satisfies all 5 criteria, reports honest scientific outcome:
    'NO NATURAL ELECTROLYTE WOW SCENARIO FOUND UNDER PREREGISTERED SETTINGS'.
    """
    total_steps = 0
    divergence_candidates_checked = 0

    for disc_run, hyb_run in zip(discovery_runs, hybrid_runs):
        if disc_run.seed != hyb_run.seed:
            continue
        min_len = min(len(disc_run.step_diagnostics), len(hyb_run.step_diagnostics))
        for step_idx in range(min_len):
            total_steps += 1
            d_diag = disc_run.step_diagnostics[step_idx]
            h_diag = hyb_run.step_diagnostics[step_idx]

            disc_cand = d_diag["candidate_id"]
            hyb_cand = h_diag["candidate_id"]

            if disc_cand == hyb_cand:
                continue

            divergence_candidates_checked += 1
            # Check 5 preregistered criteria
            cond1_diff = (disc_cand != hyb_cand)
            cond2_ei = (d_diag["predicted_discovery_score"] > h_diag["predicted_discovery_score"])
            cond3_hig = (h_diag["raw_hig_nats"] > d_diag["raw_hig_nats"])
            cond4_val = (h_diag["total_value"] > d_diag["total_value"])
            cond5_shift = (h_diag["max_belief_shift"] >= 0.01)

            if cond1_diff and cond2_ei and cond3_hig and cond4_val and cond5_shift:
                return {
                    "scenario_found": True,
                    "seed": disc_run.seed,
                    "step_index": step_idx,
                    "discovery_only_choice": {
                        "candidate_id": disc_cand,
                        "revealed_C_norm_20": d_diag["revealed_capacity"],
                        "predicted_discovery_score": d_diag["predicted_discovery_score"],
                        "raw_hig_nats": d_diag["raw_hig_nats"],
                    },
                    "hybrid_choice": {
                        "candidate_id": hyb_cand,
                        "revealed_C_norm_20": h_diag["revealed_capacity"],
                        "predicted_discovery_score": h_diag["predicted_discovery_score"],
                        "raw_hig_nats": h_diag["raw_hig_nats"],
                        "max_belief_shift": h_diag["max_belief_shift"],
                    },
                    "scientific_divergence_analysis": (
                        f"At autonomous step {step_idx + 1} (seed {disc_run.seed}), Discovery-Only selected "
                        f"{disc_cand} (yielded C_norm_20={d_diag['revealed_capacity']:.4f}) with higher pure discovery score "
                        f"({d_diag['predicted_discovery_score']:.4f} vs {h_diag['predicted_discovery_score']:.4f}), whereas "
                        f"Hybrid selected {hyb_cand} (yielded C_norm_20={h_diag['revealed_capacity']:.4f}) which possessed "
                        f"higher hypothesis information gain ({h_diag['raw_hig_nats']:.4f} vs {d_diag['raw_hig_nats']:.4f} nats), "
                        f"inducing a material Bayesian belief shift of {h_diag['max_belief_shift']:.4f} across H1-H3."
                    ),
                    "preregistered_criteria_met": {
                        "candidate_divergence": True,
                        "discovery_score_favoring_discovery_policy": True,
                        "hig_nats_favoring_hybrid_policy": True,
                        "net_valuation_favoring_hybrid": True,
                        "material_belief_shift_ge_0_01": True,
                    },
                }

    return {
        "scenario_found": False,
        "message": "NO NATURAL ELECTROLYTE WOW SCENARIO FOUND UNDER PREREGISTERED SETTINGS",
        "preregistered_criteria": [
            "c_disc != c_hyb",
            "EI(c_disc) > EI(c_hyb)",
            "HIG(c_hyb) > HIG(c_disc)",
            "V_hyb(c_hyb) > V_hyb(c_disc)",
            "max_belief_shift >= 0.01",
        ],
        "total_steps_evaluated": total_steps,
        "divergent_candidate_steps": divergence_candidates_checked,
    }


def run_retrospective_next_batch_ranking(
    derived_outcomes_path: str = DEFAULT_COMPATIBLE_DERIVED_PATH,
    feature_cols: Sequence[str] = ELECTROLYTE_SOLVENT_FEATURES,
    out_dir: str = "outputs/electrolyte/benchmark",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates forward temporal batch prediction: batches 0..k-1 -> batch k.

    Phase 12 mandate: Strictly separates:
    1. AIcoScientist Bayesian Hypothesis Ensemble (H1, H2, H3) -> aicoscientist_temporal_next_batch.json
    2. Standard Random Forest Baseline Reference Model -> rf_temporal_baseline.json
    """
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import spearmanr
    from src.domains.electrolyte.hypotheses import ElectrolyteHypothesisProvider
    from src.science.domain import HypothesisTrainingContext

    df = load_derived_historical_outcomes(derived_outcomes_path)
    f_cols = list(feature_cols)
    batches = sorted(df["batch"].unique())

    aico_batches = []
    rf_batches = []

    for k in batches:
        if k == 0:
            continue
        train_df = df[df["batch"] < k]
        test_df = df[df["batch"] == k]

        if len(train_df) < 3 or len(test_df) == 0:
            continue

        y_test = test_df["C_norm_20"].to_numpy(dtype=np.float64)
        k_top = min(5, len(test_df))
        true_top_indices = set(np.argsort(-y_test)[:k_top])
        true_best_idx = int(np.argmax(y_test))

        # 1. AIcoScientist Bayesian Ensemble
        train_feats = {
            str(row["candidate_id"]): row[f_cols].to_numpy(dtype=np.float64)
            for _, row in train_df.iterrows()
        }
        train_obs = {
            str(row["candidate_id"]): float(row["C_norm_20"])
            for _, row in train_df.iterrows()
        }
        ctx = HypothesisTrainingContext(
            candidate_features_by_id=train_feats,
            observations_by_modality={"CAPACITY_TEST": train_obs},
        )
        provider = ElectrolyteHypothesisProvider()
        hyps = provider.build_hypotheses()
        for h in hyps.values():
            h.fit_context(ctx)

        aico_preds = []
        for _, row in test_df.iterrows():
            cid = str(row["candidate_id"])
            comp = row[f_cols].to_numpy(dtype=np.float64)
            pred_means = [float(h.predict_observation(cid, "CAPACITY_TEST", comp).mean[0]) for h in hyps.values()]
            aico_preds.append(float(np.mean(pred_means)))
        aico_preds = np.array(aico_preds)

        aico_top_indices = set(np.argsort(-aico_preds)[:k_top])
        aico_p5 = len(true_top_indices.intersection(aico_top_indices)) / float(k_top)
        aico_top1_hit = bool(true_best_idx in aico_top_indices)
        aico_rho = float(spearmanr(aico_preds, y_test).statistic) if len(y_test) > 2 else 0.0

        aico_batches.append({
            "target_batch": int(k),
            "train_sample_count": len(train_df),
            "test_sample_count": len(test_df),
            "top_5_precision": round(aico_p5, 4),
            "top_1_in_top_5": aico_top1_hit,
            "spearman_rank_correlation": round(0.0 if np.isnan(aico_rho) else aico_rho, 4),
        })

        # 2. Random Forest Baseline Reference Model
        X_train = train_df[f_cols].to_numpy(dtype=np.float64)
        y_train = train_df["C_norm_20"].to_numpy(dtype=np.float64)
        X_test = test_df[f_cols].to_numpy(dtype=np.float64)

        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)

        rf_top_indices = set(np.argsort(-rf_preds)[:k_top])
        rf_p5 = len(true_top_indices.intersection(rf_top_indices)) / float(k_top)
        rf_top1_hit = bool(true_best_idx in rf_top_indices)
        rf_rho = float(spearmanr(rf_preds, y_test).statistic) if len(y_test) > 2 else 0.0

        rf_batches.append({
            "target_batch": int(k),
            "train_sample_count": len(train_df),
            "test_sample_count": len(test_df),
            "top_5_precision": round(rf_p5, 4),
            "top_1_in_top_5": rf_top1_hit,
            "spearman_rank_correlation": round(0.0 if np.isnan(rf_rho) else rf_rho, 4),
        })

    aico_summary = {
        "model_architecture": "AIcoScientist Multi-Hypothesis Bayesian Decision Ensemble (H1, H2, H3)",
        "evaluation_protocol": "Temporal Next-Batch Forward Ranking (train batches < k -> rank batch k)",
        "mean_top_5_precision": round(float(np.mean([b["top_5_precision"] for b in aico_batches])), 4),
        "mean_spearman_rank_correlation": round(float(np.mean([b["spearman_rank_correlation"] for b in aico_batches])), 4),
        "top_1_recovery_rate": round(float(np.mean([1.0 if b["top_1_in_top_5"] else 0.0 for b in aico_batches])), 4),
        "per_batch_results": aico_batches,
    }

    rf_summary = {
        "model_architecture": "Standard RandomForestRegressor Baseline (50 trees, max_depth=5) - REFERENCE ONLY",
        "evaluation_protocol": "Temporal Next-Batch Forward Ranking (train batches < k -> rank batch k)",
        "mean_top_5_precision": round(float(np.mean([b["top_5_precision"] for b in rf_batches])), 4),
        "mean_spearman_rank_correlation": round(float(np.mean([b["spearman_rank_correlation"] for b in rf_batches])), 4),
        "top_1_recovery_rate": round(float(np.mean([1.0 if b["top_1_in_top_5"] else 0.0 for b in rf_batches])), 4),
        "per_batch_results": rf_batches,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "aicoscientist_temporal_next_batch.json"), "w") as f:
        json.dump(aico_summary, f, indent=2)
    with open(os.path.join(out_dir, "rf_temporal_baseline.json"), "w") as f:
        json.dump(rf_summary, f, indent=2)

    return aico_summary, rf_summary
