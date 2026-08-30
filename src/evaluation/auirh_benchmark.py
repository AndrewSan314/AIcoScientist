from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.datasets.auirh import (
    AUIRH_CANDIDATE_ID_COLUMN,
    AUIRH_FEATURE_COLUMNS,
    AUIRH_LIBRARIES,
    AuIrRhAdapter,
    AuIrRhExperimentOracle,
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


def get_auirh_search_space() -> SearchSpace:
    """Returns SearchSpace definition for Au-Ir-Rh 2D free composition variables (Au, Ir)."""
    return SearchSpace(
        variables=[
            ContinuousVariable("Au", 0.0, 100.0),
            ContinuousVariable("Ir", 0.0, 100.0),
        ]
    )


def run_single_aicoscientist_trajectory(
    candidate_pool: pd.DataFrame,
    oracle: AuIrRhExperimentOracle,
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
    feature_cols = list(AUIRH_FEATURE_COLUMNS)

    # Initial warm-up observation
    init_cand = candidate_pool.iloc[init_sample_index]
    init_cid = str(init_cand[AUIRH_CANDIDATE_ID_COLUMN])
    init_res = oracle.query(init_cid)

    observed_records: list[dict[str, Any]] = [
        {
            AUIRH_CANDIDATE_ID_COLUMN: init_cid,
            "Library": str(init_cand["Library"]),
            "Area": int(init_cand["Area"]),
            "Au": float(init_cand["Au"]),
            "Ir": float(init_cand["Ir"]),
            "Rh": float(init_cand["Rh"]),
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
            "library": str(init_cand["Library"]),
            "area": int(init_cand["Area"]),
            "Au": float(init_cand["Au"]),
            "Ir": float(init_cand["Ir"]),
            "Rh": float(init_cand["Rh"]),
            "observed_target": float(init_res[target_name]),
            "best_observed": float(current_best),
            "global_best": float(global_optimum),
            "regret": float(regret),
            "relative_regret": float(rel_regret),
            "percent_deviation": float(pct_deviation),
            "acquisition_score": 0.0,
            "turbo_length": np.nan,
        }
    )

    # Map strategy name to BoTorchBackend strategy
    strat_key = strategy.lower().strip()
    if strat_key in {"noisy_expected_improvement", "true_nei", "nei"}:
        backend_strat = "nei"
    elif strat_key in {"expected_improvement", "ei"}:
        backend_strat = "ei"
    elif strat_key in {"thompson_sampling", "thompson"}:
        backend_strat = "thompson"
    elif strat_key in {"gp_ucb", "ucb"}:
        backend_strat = "gp_ucb"
    elif strat_key in {"turbo_nei", "turbo"}:
        backend_strat = "nei"
    else:
        backend_strat = strat_key

    opt_backend = BoTorchBackend(default_strategy=backend_strat)
    objective = OptimizationObjective(target_name=target_name, minimize=False)

    for step in range(2, total_budget + 1):
        unseen_pool = candidate_pool[~candidate_pool[AUIRH_CANDIDATE_ID_COLUMN].isin(seen_cids)].copy().reset_index(drop=True)
        if unseen_pool.empty:
            break

        step_seed = seed * 1000 + step * 10 + 7
        obs_df = pd.DataFrame(observed_records)

        proposals = opt_backend.propose(
            observations=obs_df,
            candidate_pool=candidate_pool,
            objective=objective,
            feature_columns=feature_cols,
            candidate_id_column=AUIRH_CANDIDATE_ID_COLUMN,
            seed=step_seed,
            strategy=backend_strat,
            beta=beta,
            n=1,
        )

        prop = proposals[0]
        cid = prop.candidate_id
        selected_row = candidate_pool[candidate_pool[AUIRH_CANDIDATE_ID_COLUMN] == cid].iloc[0]
        acq_score_val = prop.acquisition_value

        # Query Oracle
        res = oracle.query(cid)
        obs_val = float(res[target_name])
        seen_cids.add(cid)

        observed_records.append(
            {
                AUIRH_CANDIDATE_ID_COLUMN: cid,
                "Library": str(selected_row["Library"]),
                "Area": int(selected_row["Area"]),
                "Au": float(selected_row["Au"]),
                "Ir": float(selected_row["Ir"]),
                "Rh": float(selected_row["Rh"]),
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
                "library": str(selected_row["Library"]),
                "area": int(selected_row["Area"]),
                "Au": float(selected_row["Au"]),
                "Ir": float(selected_row["Ir"]),
                "Rh": float(selected_row["Rh"]),
                "observed_target": obs_val,
                "best_observed": float(current_best),
                "global_best": float(global_optimum),
                "regret": float(regret),
                "relative_regret": float(rel_regret),
                "percent_deviation": float(pct_deviation),
                "acquisition_score": float(acq_score_val),
                "turbo_length": np.nan,
            }
        )

    return trajectory


def run_cross_library_diagnostic(
    target_name: str = "k0",
    n_seeds: int = 10,
    prior_samples: int = 5,
    budget_in_dest: int = 15,
    base_seed: int = 42,
    output_dir: Path | str = "outputs/auirh/diagnostics",
) -> dict[str, Any]:
    """Runs cross-library prior transfer diagnostic to assess composition-only transferability."""
    pairs = [
        ("Au-rich", "Ir-rich"),
        ("Au-rich", "Rh-rich"),
        ("Ir-rich", "Rh-rich"),
        ("Rh-rich", "Ir-rich"),
    ]

    results: dict[str, Any] = {}
    per_seed_rows: list[dict[str, Any]] = []
    feature_cols = list(AUIRH_FEATURE_COLUMNS)
    opt_backend = BoTorchBackend(default_strategy="gp_ucb")
    objective = OptimizationObjective(target_name=target_name, minimize=False)

    for src_lib, dst_lib in pairs:
        pair_key = f"{src_lib}->{dst_lib}"
        src_adapter = AuIrRhAdapter(target=target_name, library=src_lib)
        dst_adapter = AuIrRhAdapter(target=target_name, library=dst_lib)

        src_pool = src_adapter.get_candidate_pool()
        dst_pool = dst_adapter.get_candidate_pool()
        dst_oracle = dst_adapter.create_oracle()
        src_oracle = src_adapter.create_oracle()
        dst_global_best = dst_oracle.global_best_value

        regrets_with_prior: list[float] = []
        regrets_without_prior: list[float] = []

        for s_idx in range(n_seeds):
            seed = base_seed + s_idx
            seed_rng = np.random.default_rng(seed)

            # Pick 1 init sample in destination
            init_dst_idx = int(seed_rng.integers(0, len(dst_pool)))
            init_dst_row = dst_pool.iloc[init_dst_idx]
            dst_oracle.reset()
            init_res = dst_oracle.query(str(init_dst_row[AUIRH_CANDIDATE_ID_COLUMN]))

            # Run baseline on destination alone (cold start)
            cold_records = [
                {
                    AUIRH_CANDIDATE_ID_COLUMN: str(init_dst_row[AUIRH_CANDIDATE_ID_COLUMN]),
                    "Au": float(init_dst_row["Au"]),
                    "Ir": float(init_dst_row["Ir"]),
                    target_name: float(init_res[target_name]),
                }
            ]
            seen_dst = {str(init_dst_row[AUIRH_CANDIDATE_ID_COLUMN])}
            cold_best = float(init_res[target_name])

            for step in range(2, budget_in_dest + 1):
                obs_df = pd.DataFrame(cold_records)
                props = opt_backend.propose(
                    observations=obs_df,
                    candidate_pool=dst_pool,
                    objective=objective,
                    feature_columns=feature_cols,
                    candidate_id_column=AUIRH_CANDIDATE_ID_COLUMN,
                    strategy="gp_ucb",
                    beta=2.0,
                    seed=seed + step,
                )
                if not props:
                    break
                p = props[0]
                cid = p.candidate_id
                row_sel = dst_pool[dst_pool[AUIRH_CANDIDATE_ID_COLUMN] == cid].iloc[0]
                res = dst_oracle.query(cid)
                obs_val = float(res[target_name])
                seen_dst.add(cid)
                cold_records.append({
                    AUIRH_CANDIDATE_ID_COLUMN: cid,
                    "Au": float(row_sel["Au"]),
                    "Ir": float(row_sel["Ir"]),
                    target_name: obs_val,
                })
                cold_best = max(cold_best, obs_val)

            regrets_without_prior.append(float(dst_global_best - cold_best))

            # Run with source prior observations (warm start)
            src_oracle.reset()
            dst_oracle.reset()
            src_indices = seed_rng.choice(len(src_pool), size=prior_samples, replace=False)
            prior_records = []
            for s_i in src_indices:
                src_row = src_pool.iloc[s_i]
                s_cid = str(src_row[AUIRH_CANDIDATE_ID_COLUMN])
                s_res = src_oracle.query(s_cid)
                prior_records.append({
                    AUIRH_CANDIDATE_ID_COLUMN: s_cid,
                    "Au": float(src_row["Au"]),
                    "Ir": float(src_row["Ir"]),
                    target_name: float(s_res[target_name]),
                })

            init_res_warm = dst_oracle.query(str(init_dst_row[AUIRH_CANDIDATE_ID_COLUMN]))
            warm_records = list(prior_records)
            warm_records.append({
                AUIRH_CANDIDATE_ID_COLUMN: str(init_dst_row[AUIRH_CANDIDATE_ID_COLUMN]),
                "Au": float(init_dst_row["Au"]),
                "Ir": float(init_dst_row["Ir"]),
                target_name: float(init_res_warm[target_name]),
            })
            seen_dst_warm = {str(init_dst_row[AUIRH_CANDIDATE_ID_COLUMN])}
            warm_best = float(init_res_warm[target_name])

            for step in range(2, budget_in_dest + 1):
                obs_df = pd.DataFrame(warm_records)
                props = opt_backend.propose(
                    observations=obs_df,
                    candidate_pool=dst_pool,
                    objective=objective,
                    feature_columns=feature_cols,
                    candidate_id_column=AUIRH_CANDIDATE_ID_COLUMN,
                    strategy="gp_ucb",
                    beta=2.0,
                    seed=seed + step,
                )
                if not props:
                    break
                p = props[0]
                cid = p.candidate_id
                row_sel = dst_pool[dst_pool[AUIRH_CANDIDATE_ID_COLUMN] == cid].iloc[0]
                res = dst_oracle.query(cid)
                obs_val = float(res[target_name])
                seen_dst_warm.add(cid)
                warm_records.append({
                    AUIRH_CANDIDATE_ID_COLUMN: cid,
                    "Au": float(row_sel["Au"]),
                    "Ir": float(row_sel["Ir"]),
                    target_name: obs_val,
                })
                warm_best = max(warm_best, obs_val)

            regrets_with_prior.append(float(dst_global_best - warm_best))

        regrets_cold_arr = np.array(regrets_without_prior, dtype=float)
        regrets_warm_arr = np.array(regrets_with_prior, dtype=float)
        paired_deltas = regrets_warm_arr - regrets_cold_arr

        delta_mean = float(np.mean(paired_deltas))
        delta_ci_lower, delta_ci_upper = compute_bootstrap_ci_95(paired_deltas, seed=base_seed)

        if delta_ci_upper < 0.0:
            transfer_impact = "helpful"
        elif delta_ci_lower > 0.0:
            transfer_impact = "harmful"
        else:
            transfer_impact = "inconclusive / neutral"

        results[pair_key] = {
            "source_library": src_lib,
            "destination_library": dst_lib,
            "prior_samples_from_source": prior_samples,
            "budget_in_destination": budget_in_dest,
            "n_seeds": n_seeds,
            "mean_regret_cold_start": float(np.mean(regrets_cold_arr)),
            "std_regret_cold_start": float(np.std(regrets_cold_arr)),
            "mean_regret_warm_prior": float(np.mean(regrets_warm_arr)),
            "std_regret_warm_prior": float(np.std(regrets_warm_arr)),
            "paired_regret_delta_mean": delta_mean,
            "paired_regret_delta_95ci": [delta_ci_lower, delta_ci_upper],
            "transfer_impact": transfer_impact,
        }

        for s_i in range(n_seeds):
            per_seed_rows.append({
                "pair": pair_key,
                "source_library": src_lib,
                "destination_library": dst_lib,
                "seed_index": s_i,
                "run_seed": base_seed + s_i,
                "cold_start_regret": float(regrets_cold_arr[s_i]),
                "warm_prior_regret": float(regrets_warm_arr[s_i]),
                "paired_delta": float(paired_deltas[s_i]),
            })

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    df_per_seed = pd.DataFrame(per_seed_rows)
    per_seed_file = out_p / "per_seed.csv"
    df_per_seed.to_csv(per_seed_file, index=False)

    out_file = out_p / "summary.json"
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)

    logger.info("Saved cross-library diagnostic summary to %s and per-seed to %s", out_file, per_seed_file)
    return results
