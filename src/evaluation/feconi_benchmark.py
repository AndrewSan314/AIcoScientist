from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.datasets.feconi import (
    FECONI_CANDIDATE_ID_COLUMN,
    FECONI_FEATURE_COLUMNS,
    FeCoNiAdapter,
    FeCoNiExperimentOracle,
)
from src.optimization.botorch_backend import BoTorchBackend
from src.optimization.objective import OptimizationObjective
from src.optimization.search_space import ContinuousVariable, SearchSpace

logger = logging.getLogger(__name__)


def compute_bootstrap_ci_95(
    data: np.ndarray | Sequence[float],
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes non-parametric bootstrap 95% confidence interval for the sample mean."""
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    if n <= 1 or np.all(arr == arr[0]):
        val = float(arr[0]) if n > 0 else 0.0
        return val, val

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))
    boot_means = np.mean(arr[boot_indices], axis=1)

    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))
    return lower, upper


def get_feconi_search_space() -> SearchSpace:
    """Returns SearchSpace definition for Fe-Co-Ni 2D free composition variables."""
    return SearchSpace(
        variables=[
            ContinuousVariable("Co", 0.0, 100.0),
            ContinuousVariable("Fe", 0.0, 100.0),
        ]
    )


def run_single_aicoscientist_trajectory(
    candidate_pool: pd.DataFrame,
    oracle: FeCoNiExperimentOracle,
    target_name: str,
    strategy: str,  # "random", "greedy", "gp_ucb", "expected_improvement", "noisy_expected_improvement", "turbo_nei", "thompson"
    init_sample_index: int,
    total_budget: int = 50,
    seed: int = 42,
    beta: float = 2.0,
) -> list[dict[str, Any]]:
    """Runs a single closed-loop optimization trajectory delegating acquisition to BoTorchBackend."""
    oracle.reset()
    global_optimum = oracle.global_best_value
    feature_cols = list(FECONI_FEATURE_COLUMNS)

    # Initial warm-up observation
    init_cand = candidate_pool.iloc[init_sample_index]
    init_cid = str(init_cand[FECONI_CANDIDATE_ID_COLUMN])
    init_res = oracle.query(init_cid)

    observed_records: list[dict[str, Any]] = [
        {
            FECONI_CANDIDATE_ID_COLUMN: init_cid,
            "sample_index": int(init_cand["sample_index"]),
            "Co": float(init_cand["Co"]),
            "Fe": float(init_cand["Fe"]),
            "Ni": float(init_cand["Ni"]),
            target_name: float(init_res[target_name]),
        }
    ]
    seen_cids: set[str] = {init_cid}

    trajectory: list[dict[str, Any]] = []

    current_best = float(init_res[target_name])
    regret = float(global_optimum - current_best)
    rel_regret = float(regret / global_optimum) if global_optimum != 0 else 0.0
    pct_deviation = float(regret / global_optimum * 100.0) if global_optimum != 0 else 0.0

    trajectory.append(
        {
            "run_seed": seed,
            "target": target_name,
            "strategy": strategy,
            "iteration": 1,
            "selected_sample_id": init_cid,
            "sample_index": int(init_cand["sample_index"]),
            "Co": float(init_cand["Co"]),
            "Fe": float(init_cand["Fe"]),
            "Ni": float(init_cand["Ni"]),
            "observed_target": float(init_res[target_name]),
            "best_observed": float(current_best),
            "global_best": float(global_optimum),
            "regret": float(regret),
            "relative_regret": float(rel_regret),
            "percent_deviation": float(pct_deviation),
            "acquisition_score": 0.0,
        }
    )

    # Map strategy name to BoTorchBackend strategy
    strat_key = strategy.lower().strip()
    if strat_key == "noisy_expected_improvement":
        backend_strat = "nei"
    elif strat_key == "expected_improvement":
        backend_strat = "ei"
    elif strat_key in {"thompson_sampling", "thompson"}:
        backend_strat = "thompson"
    elif strat_key in {"gp_ucb_1", "gp_ucb_2", "gp_ucb", "ucb"}:
        backend_strat = "gp_ucb"
    elif strat_key in {"turbo_nei", "turbo_ei"}:
        backend_strat = "nei"
    else:
        backend_strat = strat_key

    b_val = 1.0 if strategy == "gp_ucb_1" else beta

    opt_backend = BoTorchBackend(default_strategy=backend_strat)
    objective = OptimizationObjective(target_name=target_name, minimize=False)

    for step in range(2, total_budget + 1):
        unseen_pool = candidate_pool[~candidate_pool[FECONI_CANDIDATE_ID_COLUMN].isin(seen_cids)].copy().reset_index(drop=True)
        if unseen_pool.empty:
            break

        step_seed = seed * 1000 + step * 10 + 7
        obs_df = pd.DataFrame(observed_records)

        proposals = opt_backend.propose(
            observations=obs_df,
            candidate_pool=candidate_pool,
            objective=objective,
            feature_columns=feature_cols,
            candidate_id_column=FECONI_CANDIDATE_ID_COLUMN,
            seed=step_seed,
            strategy=backend_strat,
            beta=b_val,
            n=1,
        )

        prop = proposals[0]
        cid = prop.candidate_id
        selected_row = candidate_pool[candidate_pool[FECONI_CANDIDATE_ID_COLUMN] == cid].iloc[0]
        acq_score_val = prop.acquisition_value

        # Query Oracle
        res = oracle.query(cid)
        obs_val = float(res[target_name])
        seen_cids.add(cid)

        observed_records.append(
            {
                FECONI_CANDIDATE_ID_COLUMN: cid,
                "sample_index": int(selected_row["sample_index"]),
                "Co": float(selected_row["Co"]),
                "Fe": float(selected_row["Fe"]),
                "Ni": float(selected_row["Ni"]),
                target_name: obs_val,
            }
        )

        if obs_val > current_best:
            current_best = obs_val

        regret = float(global_optimum - current_best)
        rel_regret = float(regret / global_optimum) if global_optimum != 0 else 0.0
        pct_deviation = float(regret / global_optimum * 100.0) if global_optimum != 0 else 0.0

        trajectory.append(
            {
                "run_seed": seed,
                "target": target_name,
                "strategy": strategy,
                "iteration": step,
                "selected_sample_id": cid,
                "sample_index": int(selected_row["sample_index"]),
                "Co": float(selected_row["Co"]),
                "Fe": float(selected_row["Fe"]),
                "Ni": float(selected_row["Ni"]),
                "observed_target": obs_val,
                "best_observed": float(current_best),
                "global_best": float(global_optimum),
                "regret": float(regret),
                "relative_regret": float(rel_regret),
                "percent_deviation": float(pct_deviation),
                "acquisition_score": float(acq_score_val),
            }
        )

    return trajectory


def run_feconi_benchmark_suite(
    target_name: str = "Kerr",
    n_seeds: int = 30,
    total_budget: int = 50,
    output_dir: Path | str = "outputs/feconi",
) -> dict[str, Any]:
    """Runs full Fe-Co-Ni benchmark comparing strategies across n_seeds with summary statistics and statistical test."""
    adapter = FeCoNiAdapter()
    df = adapter.load()
    oracle = FeCoNiExperimentOracle(df, target_column=target_name, allow_duplicate_queries=False)
    cand_pool = adapter.candidate_space(observed=None)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    strategies = [
        "random",
        "greedy",
        "gp_ucb",
        "expected_improvement",
        "noisy_expected_improvement",
        "turbo_nei",
        "thompson",
    ]

    all_trajectories: list[dict[str, Any]] = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        init_idx = int(rng.integers(0, len(cand_pool)))
        for strat in strategies:
            traj = run_single_aicoscientist_trajectory(
                candidate_pool=cand_pool,
                oracle=oracle,
                target_name=target_name,
                strategy=strat,
                init_sample_index=init_idx,
                total_budget=total_budget,
                seed=seed,
            )
            all_trajectories.extend(traj)

    trajs_df = pd.DataFrame(all_trajectories)
    trajs_df.to_csv(out_p / f"feconi_{target_name}_trajectories.csv", index=False)

    summary_rows = []
    for strat in strategies:
        sub = trajs_df[(trajs_df["strategy"] == strat) & (trajs_df["iteration"] == total_budget)]
        final_devs = sub["percent_deviation"].to_numpy()
        mean_dev = float(np.mean(final_devs))
        std_dev = float(np.std(final_devs, ddof=1)) if len(final_devs) > 1 else 0.0
        ci_low, ci_high = compute_bootstrap_ci_95(final_devs)
        summary_rows.append(
            {
                "strategy": strat,
                "target": target_name,
                "n_seeds": n_seeds,
                "final_pct_deviation_mean": mean_dev,
                "final_pct_deviation_std": std_dev,
                "final_pct_deviation_ci_95_lower": ci_low,
                "final_pct_deviation_ci_95_upper": ci_high,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_p / f"feconi_{target_name}_summary.csv", index=False)

    report = {
        "dataset": "FeCoNi",
        "target": target_name,
        "n_seeds": n_seeds,
        "total_budget": total_budget,
        "backend": "botorch",
        "summary": summary_rows,
    }
    with open(out_p / f"feconi_{target_name}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
