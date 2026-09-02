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
    cumulative_hig: float
    final_beliefs: dict[str, float]
    steps_count: int
    elapsed_time_sec: float


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
    mean_hig: float
    mean_entropy_reduction: float
    runtime_sec_mean: float


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
    all_candidates = adapter.get_candidate_pool()

    # Configure policy
    is_random = (policy_name == "RANDOM")
    if policy_name == "PURE_FALSIFICATION":
        mode = FalsificationPolicyMode.PURE_FALSIFICATION
        w_hig = 1.0
        w_disc = 0.0
    elif policy_name == "DISCOVERY_ONLY":
        mode = FalsificationPolicyMode.DISCOVERY_ONLY
        w_hig = 0.0
        w_disc = 1.0
    else:
        mode = FalsificationPolicyMode.HYBRID
        w_hig = 0.5
        w_disc = 1.0

    policy = FalsificationFirstPolicy(
        mode=mode,
        w_hig=w_hig,
        w_disc=w_disc,
        w_cost=0.0,  # All CAPACITY_TEST actions have equal unit cost
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
    cumulative_hig = 0.0

    rng = np.random.default_rng(seed)

    # Run autonomous exploration steps
    for step in range(max_steps):
        valid_actions = adapter.list_valid_actions(engine.get_state())
        if not valid_actions:
            break

        if is_random:
            chosen_action = rng.choice(valid_actions)
            outcome = adapter.execute_or_reveal(chosen_action)
            engine.revealed_outcomes[chosen_action.action_id] = outcome
            if "CAPACITY_TEST" not in engine.observations_by_modality:
                engine.observations_by_modality["CAPACITY_TEST"] = {}
            c_val = float(outcome.revealed_data["C_norm_20"])
            engine.observations_by_modality["CAPACITY_TEST"][chosen_action.candidate_id] = c_val
        else:
            rec = engine.propose_next_experiment()
            chosen_action = rec.action
            outcome = engine.execute_recommendation(rec)
            c_val = float(outcome.revealed_data["C_norm_20"])
            cumulative_hig += float(rec.scientific_information_value)

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
        cumulative_hig=round(cumulative_hig, 4),
        final_beliefs={k: round(v, 4) for k, v in final_beliefs.items()},
        steps_count=len(autonomous_obs),
        elapsed_time_sec=round(elapsed, 4),
    )


def run_comprehensive_historical_benchmark(
    derived_outcomes_path: str = DEFAULT_COMPATIBLE_DERIVED_PATH,
    seeds: Sequence[int] = (42, 101, 2024, 314, 7),
    policies: Sequence[str] = (
        "RANDOM",
        "DISCOVERY_ONLY",
        "PURE_FALSIFICATION",
        "HYBRID",
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
        higs = [r.cumulative_hig for r in runs]
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
                mean_hig=round(float(np.mean(higs)), 4),
                mean_entropy_reduction=round(float(np.mean(ent_reds)), 4),
                runtime_sec_mean=round(float(np.mean(times)), 4),
            )
        )

    # Search for natural "Wow" scenario: earliest step where Hybrid diverges from Discovery-Only
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
    """Deterministically identifies the earliest divergence between Discovery-Only and Hybrid."""
    for disc_run, hyb_run in zip(discovery_runs, hybrid_runs):
        if disc_run.seed != hyb_run.seed:
            continue
        min_len = min(len(disc_run.autonomous_actions), len(hyb_run.autonomous_actions))
        for step_idx in range(min_len):
            disc_cand = disc_run.autonomous_actions[step_idx]
            hyb_cand = hyb_run.autonomous_actions[step_idx]
            if disc_cand != hyb_cand:
                disc_val = disc_run.autonomous_observations[step_idx]
                hyb_val = hyb_run.autonomous_observations[step_idx]
                return {
                    "scenario_found": True,
                    "seed": disc_run.seed,
                    "step_index": step_idx,
                    "discovery_only_choice": {
                        "candidate_id": disc_cand,
                        "revealed_C_norm_20": disc_val,
                    },
                    "hybrid_choice": {
                        "candidate_id": hyb_cand,
                        "revealed_C_norm_20": hyb_val,
                    },
                    "scientific_divergence_analysis": (
                        f"At autonomous step {step_idx + 1} (seed {disc_run.seed}), Discovery-Only chose "
                        f"candidate {disc_cand} (yielded C_norm_20={disc_val:.4f}) prioritizing predicted mean capacity, "
                        f"whereas Hybrid chose candidate {hyb_cand} (yielded C_norm_20={hyb_val:.4f}) balancing expected "
                        f"improvement with hypothesis falsification information gain (HIG). "
                        f"Revealing this measurement updated belief weights across H1, H2, and H3."
                    ),
                }

    return {
        "scenario_found": False,
        "message": "NO NATURAL ELECTROLYTE WOW SCENARIO FOUND UNDER PREREGISTERED SETTINGS",
    }
