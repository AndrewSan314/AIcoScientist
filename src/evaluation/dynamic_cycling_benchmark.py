from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, RepeatedKFold
from sklearn.preprocessing import StandardScaler

from src.datasets.dynamic_cycling import (
    DYNAMIC_CYCLING_FEATURE_COLUMNS,
    DynamicCyclingAdapter,
    compute_replicate_feature_differences,
)
from src.evaluation.oracle import OfflineOracle


def evaluate_surrogate_prediction(
    adapter: DynamicCyclingAdapter,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluates surrogate model performance under strict protocol-grouped splitting and protocol-level CV."""
    cells_df = adapter.load_cells()
    protocols_df = adapter.load_protocols()
    spec = adapter.spec

    feature_cols = list(spec.feature_columns)

    # 1. Cell-Level Evaluation (Grouped Split)
    target_col_cell = "efc_lifetime"
    group_col = "protocol_id"

    groups = cells_df[group_col].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(cells_df, groups=groups))

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    if train_groups & test_groups:
        raise RuntimeError(f"Group leakage detected in dynamic cycling split: {train_groups & test_groups}")

    train_df = cells_df.iloc[train_idx]
    test_df = cells_df.iloc[test_idx]

    X_train_cell = train_df[feature_cols].to_numpy(dtype=float)
    y_train_cell = train_df[target_col_cell].to_numpy(dtype=float)
    X_test_cell = test_df[feature_cols].to_numpy(dtype=float)
    y_test_cell = test_df[target_col_cell].to_numpy(dtype=float)

    # Cell RF
    rf_cell = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=random_state)
    rf_cell.fit(X_train_cell, y_train_cell)
    rf_pred_cell = rf_cell.predict(X_test_cell)

    # Cell GP
    scaler_cell = StandardScaler()
    X_train_cell_scaled = scaler_cell.fit_transform(X_train_cell)
    X_test_cell_scaled = scaler_cell.transform(X_test_cell)

    kernel_cell = ConstantKernel(1.0, (1e-2, 1e4)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
        noise_level=1.0, noise_level_bounds=(1e-5, 1e2)
    )
    gp_cell = GaussianProcessRegressor(
        kernel=kernel_cell,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=random_state,
    )
    gp_cell.fit(X_train_cell_scaled, y_train_cell)
    gp_pred_cell = gp_cell.predict(X_test_cell_scaled)

    def _get_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        return {
            "mae": float(mean_absolute_error(true, pred)),
            "rmse": float(np.sqrt(mean_squared_error(true, pred))),
            "r2": float(r2_score(true, pred)),
        }

    cell_level_results = {
        "n_train_cells": len(train_df),
        "n_test_cells": len(test_df),
        "n_train_protocols": len(train_groups),
        "n_test_protocols": len(test_groups),
        "random_forest": _get_metrics(y_test_cell, rf_pred_cell),
        "gaussian_process": _get_metrics(y_test_cell, gp_pred_cell),
    }

    # 2. Protocol-Level Evaluation (Repeated 5-Fold CV with 10 repeats, Out-Of-Fold metrics per repeat)
    X_proto = protocols_df[feature_cols].to_numpy(dtype=float)
    y_proto = protocols_df["target_mean"].to_numpy(dtype=float)
    n_protocols = len(protocols_df)
    n_splits = 5
    n_repeats = 10

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    splits = list(rkf.split(X_proto))

    rf_repeats: list[dict[str, float]] = []
    gp_repeats: list[dict[str, float]] = []

    for rep_idx in range(n_repeats):
        oof_rf_pred = np.full(n_protocols, np.nan)
        oof_gp_pred = np.full(n_protocols, np.nan)
        rep_splits = splits[rep_idx * n_splits : (rep_idx + 1) * n_splits]

        for fold_train_idx, fold_test_idx in rep_splits:
            X_p_tr, y_p_tr = X_proto[fold_train_idx], y_proto[fold_train_idx]
            X_p_te, y_p_te = X_proto[fold_test_idx], y_proto[fold_test_idx]

            # Protocol RF
            rf_fold = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=random_state + rep_idx)
            rf_fold.fit(X_p_tr, y_p_tr)
            oof_rf_pred[fold_test_idx] = rf_fold.predict(X_p_te)

            # Protocol GP
            sc_p = StandardScaler()
            X_p_tr_sc = sc_p.fit_transform(X_p_tr)
            X_p_te_sc = sc_p.transform(X_p_te)
            kernel_proto = ConstantKernel(1.0, (1e-2, 1e4)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
                noise_level=1.0, noise_level_bounds=(1e-5, 1e2)
            )
            gp_fold = GaussianProcessRegressor(
                kernel=kernel_proto,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=random_state + rep_idx,
            )
            gp_fold.fit(X_p_tr_sc, y_p_tr)
            oof_gp_pred[fold_test_idx] = gp_fold.predict(X_p_te_sc)

        if np.isnan(oof_rf_pred).any() or np.isnan(oof_gp_pred).any():
            raise RuntimeError(f"OOF prediction in repeat {rep_idx} contains unpredicted protocols")

        rf_repeats.append(
            {
                "mae": float(mean_absolute_error(y_proto, oof_rf_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_proto, oof_rf_pred))),
                "r2": float(r2_score(y_proto, oof_rf_pred)),
            }
        )
        gp_repeats.append(
            {
                "mae": float(mean_absolute_error(y_proto, oof_gp_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_proto, oof_gp_pred))),
                "r2": float(r2_score(y_proto, oof_gp_pred)),
            }
        )

    rf_maes = [r["mae"] for r in rf_repeats]
    rf_rmses = [r["rmse"] for r in rf_repeats]
    rf_r2s = [r["r2"] for r in rf_repeats]

    gp_maes = [r["mae"] for r in gp_repeats]
    gp_rmses = [r["rmse"] for r in gp_repeats]
    gp_r2s = [r["r2"] for r in gp_repeats]

    protocol_level_results = {
        "n_protocols": n_protocols,
        "cv_method": f"RepeatedKFold(n_splits={n_splits}, n_repeats={n_repeats}, random_state={random_state})",
        "random_state": random_state,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_forest": {
            "mean_mae": float(np.mean(rf_maes)),
            "std_mae": float(np.std(rf_maes)),
            "mean_rmse": float(np.mean(rf_rmses)),
            "std_rmse": float(np.std(rf_rmses)),
            "mean_r2": float(np.mean(rf_r2s)),
            "std_r2": float(np.std(rf_r2s)),
            "per_repeat": rf_repeats,
        },
        "gaussian_process": {
            "mean_mae": float(np.mean(gp_maes)),
            "std_mae": float(np.std(gp_maes)),
            "mean_rmse": float(np.mean(gp_rmses)),
            "std_rmse": float(np.std(gp_rmses)),
            "mean_r2": float(np.mean(gp_r2s)),
            "std_r2": float(np.std(gp_r2s)),
            "per_repeat": gp_repeats,
        },
    }

    return {
        "cell_level": cell_level_results,
        "protocol_level": protocol_level_results,
        # Top-level backwards compatibility
        "n_train_cells": cell_level_results["n_train_cells"],
        "n_test_cells": cell_level_results["n_test_cells"],
        "n_train_protocols": cell_level_results["n_train_protocols"],
        "n_test_protocols": cell_level_results["n_test_protocols"],
        "random_forest": cell_level_results["random_forest"],
        "gaussian_process": cell_level_results["gaussian_process"],
    }


def run_single_optimization_trajectory(
    candidate_pool: pd.DataFrame,
    oracle: OfflineOracle,
    feature_cols: list[str],
    strategy: str,  # "random", "greedy", "gp_ucb"
    init_indices: list[int],
    total_queries: int,
    evaluator_meta: dict[str, float],
    rng: np.random.Generator,
    beta: float = 1.0,
) -> list[dict[str, Any]]:
    """Runs a single closed-loop optimization trajectory.

    Visibility guarantee:
    - candidate_pool contains ONLY protocol_id and design features.
    - Ground truth targets are revealed ONLY through oracle.query().
    """
    global_max = evaluator_meta["global_max"]
    top_10_pct_val = evaluator_meta["top_10_pct_val"]
    top_5_pct_val = evaluator_meta["top_5_pct_val"]

    # Reveal initial protocols
    observed_protocols: list[dict[str, Any]] = []
    for idx in init_indices:
        cand_row = candidate_pool.iloc[idx]
        query_dict = {col: cand_row[col] for col in feature_cols}
        query_dict["protocol_id"] = cand_row["protocol_id"]
        response = oracle.query(query_dict)
        observed_protocols.append(
            {
                "protocol_id": cand_row["protocol_id"],
                "target": response.target,
                **query_dict,
            }
        )

    history: list[dict[str, Any]] = []
    seen_targets = [row["target"] for row in observed_protocols]
    current_best = max(seen_targets)

    # Initial state record (step 0, initial protocols evaluated)
    history.append(
        {
            "step": 0,
            "strategy": strategy,
            "best_seen": current_best,
            "simple_regret": global_max - current_best,
            "hit_top_10_pct": int(current_best >= top_10_pct_val),
            "hit_top_5_pct": int(current_best >= top_5_pct_val),
            "queried_protocol_id": None,
        }
    )

    observed_df = pd.DataFrame(observed_protocols)

    # Closed-loop BO loop for total_queries additional experiments
    for step in range(1, total_queries + 1):
        observed_ids = set(observed_df["protocol_id"].astype(str))
        # Form unseen candidate space from candidate pool
        unseen_mask = ~candidate_pool["protocol_id"].astype(str).isin(observed_ids)
        unseen_cands = candidate_pool[unseen_mask].copy()

        if unseen_cands.empty:
            break

        if strategy == "random":
            chosen_cand_pos = int(rng.integers(0, len(unseen_cands)))
            chosen_row = unseen_cands.iloc[chosen_cand_pos]
        elif strategy in {"greedy", "gp_ucb"}:
            X_obs = observed_df[feature_cols].to_numpy(dtype=float)
            y_obs = observed_df["target"].to_numpy(dtype=float)

            scaler = StandardScaler()
            X_obs_scaled = scaler.fit_transform(X_obs)
            X_cands = unseen_cands[feature_cols].to_numpy(dtype=float)
            X_cands_scaled = scaler.transform(X_cands)

            kernel = ConstantKernel(1.0, (1e-2, 1e4)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
                noise_level=1.0, noise_level_bounds=(1e-5, 1e2)
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=int(rng.integers(0, 1000000)),
            )
            gp.fit(X_obs_scaled, y_obs)
            mean, std = gp.predict(X_cands_scaled, return_std=True)

            if strategy == "greedy":
                score = mean
            else:  # gp_ucb
                score = mean + beta * std

            best_cand_pos = int(np.argmax(score))
            chosen_row = unseen_cands.iloc[best_cand_pos]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Query Oracle
        query_dict = {col: chosen_row[col] for col in feature_cols}
        query_dict["protocol_id"] = chosen_row["protocol_id"]
        response = oracle.query(query_dict)

        new_obs = {
            "protocol_id": chosen_row["protocol_id"],
            "target": response.target,
            **query_dict,
        }
        observed_df = pd.concat([observed_df, pd.DataFrame([new_obs])], ignore_index=True)

        if response.target > current_best:
            current_best = response.target

        history.append(
            {
                "step": step,
                "strategy": strategy,
                "best_seen": current_best,
                "simple_regret": global_max - current_best,
                "hit_top_10_pct": int(current_best >= top_10_pct_val),
                "hit_top_5_pct": int(current_best >= top_5_pct_val),
                "queried_protocol_id": chosen_row["protocol_id"],
            }
        )

    return history


def compute_bootstrap_ci(data: np.ndarray, n_bootstraps: int = 2000, ci: float = 0.95, random_state: int = 42) -> tuple[float, float]:

    """Computes percentile bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(random_state)
    boot_means = np.empty(n_bootstraps)
    n = len(data)
    for b in range(n_bootstraps):
        resample = rng.choice(data, size=n, replace=True)
        boot_means[b] = np.mean(resample)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return low, high


def compute_paired_comparison(

    traj_df: pd.DataFrame,
    total_queries: int,
) -> dict[str, Any]:
    """Computes paired seed-by-seed comparison between Greedy and GP-UCB."""
    greedy_traj = traj_df[traj_df["strategy"] == "greedy"]
    ucb_traj = traj_df[traj_df["strategy"] == "gp_ucb"]

    seeds = sorted(traj_df["seed"].unique())
    diff_regret: list[float] = []
    diff_auc: list[float] = []
    diff_top5_queries: list[float] = []

    for seed in seeds:
        g_seed = greedy_traj[greedy_traj["seed"] == seed].sort_values("step")
        u_seed = ucb_traj[ucb_traj["seed"] == seed].sort_values("step")

        # Final simple regret difference (UCB - Greedy; negative means UCB has lower regret)
        g_final_regret = float(g_seed[g_seed["step"] == total_queries]["simple_regret"].iloc[0])
        u_final_regret = float(u_seed[u_seed["step"] == total_queries]["simple_regret"].iloc[0])
        diff_regret.append(u_final_regret - g_final_regret)

        # Regret AUC difference (mean regret across steps 0..total_queries)
        g_auc = float(g_seed["simple_regret"].mean())
        u_auc = float(u_seed["simple_regret"].mean())
        diff_auc.append(u_auc - g_auc)

        # Queries to top 5%
        g_hit = g_seed[g_seed["hit_top_5_pct"] == 1]["step"]
        g_q = int(g_hit.min()) if not g_hit.empty else total_queries + 1

        u_hit = u_seed[u_seed["hit_top_5_pct"] == 1]["step"]
        u_q = int(u_hit.min()) if not u_hit.empty else total_queries + 1

        diff_top5_queries.append(float(u_q - g_q))

    diff_regret_arr = np.array(diff_regret)
    diff_auc_arr = np.array(diff_auc)
    diff_top5_arr = np.array(diff_top5_queries)

    def _safe_wilcoxon(diffs: np.ndarray) -> float | None:
        if np.all(diffs == 0):
            return 1.0
        try:
            from scipy import stats
            res = stats.wilcoxon(diffs, zero_method="pratt")
            return float(res.pvalue)
        except Exception:
            return None

    return {
        "n_paired_seeds": len(seeds),
        "regret_diff_ucb_minus_greedy": {
            "mean": float(np.mean(diff_regret_arr)),
            "median": float(np.median(diff_regret_arr)),
            "std": float(np.std(diff_regret_arr)),
            "p_value": _safe_wilcoxon(diff_regret_arr),
        },
        "auc_diff_ucb_minus_greedy": {
            "mean": float(np.mean(diff_auc_arr)),
            "median": float(np.median(diff_auc_arr)),
            "std": float(np.std(diff_auc_arr)),
            "p_value": _safe_wilcoxon(diff_auc_arr),
        },
        "queries_to_top5_diff_ucb_minus_greedy": {
            "mean": float(np.mean(diff_top5_arr)),
            "median": float(np.median(diff_top5_arr)),
            "std": float(np.std(diff_top5_arr)),
            "p_value": _safe_wilcoxon(diff_top5_arr),
        },
    }


def run_dynamic_cycling_benchmark(
    adapter: DynamicCyclingAdapter | None = None,
    output_dir: Path | None = None,
    initial_protocols: int = 5,
    total_budget: int = 20,
    n_seeds: int = 50,
    budgets_to_sweep: list[int] | None = None,
) -> dict[str, Any]:
    """Runs surrogate evaluation and closed-loop BO comparison with paired seeds and bootstrap statistics.

    Budget parameters:
    - initial_protocols: Number of randomly selected protocols before closed-loop BO (default: 5)
    - total_budget: Total number of evaluated protocols for primary benchmark (default: 20)
    - total_queries = total_budget - initial_protocols (default: 15 additional queries)
    - budgets_to_sweep: Optional list of total budgets to sweep (default: [8, 10, 12, 15, 20])
    """
    if adapter is None:
        adapter = DynamicCyclingAdapter()

    project_root = Path(__file__).resolve().parent.parent.parent
    if output_dir is None:
        output_dir = project_root / "outputs" / "dynamic_cycling"
    output_dir.mkdir(parents=True, exist_ok=True)

    if budgets_to_sweep is None:
        budgets_to_sweep = [8, 10, 12, 15, 20]

    # 1. Surrogate prediction benchmark
    surrogate_metrics = evaluate_surrogate_prediction(adapter)

    # 2. Setup candidate pool (47 protocol rows) and hidden cell-level oracle (92 rows)
    candidate_pool = adapter.load_candidate_pool()
    hidden_oracle_df = adapter.load_hidden_oracle()
    oracle = OfflineOracle(hidden_oracle_df, adapter.spec, replicate_policy="mean")
    feature_cols = list(adapter.spec.feature_columns)
    protocols_df = adapter.load_protocols()

    n_candidates = len(candidate_pool)
    if initial_protocols <= 0:
        raise ValueError("initial_protocols must be a positive integer")

    # Evaluator metrics computed from protocol ground truth
    ground_truth_targets = protocols_df["target_mean"].to_numpy(dtype=float)
    global_max = float(np.max(ground_truth_targets))
    sorted_targets = np.sort(ground_truth_targets)[::-1]
    top_10_pct_val = float(sorted_targets[max(0, int(np.ceil(0.10 * n_candidates)) - 1)])
    top_5_pct_val = float(sorted_targets[max(0, int(np.ceil(0.05 * n_candidates)) - 1)])

    evaluator_meta = {
        "global_max": global_max,
        "top_10_pct_val": top_10_pct_val,
        "top_5_pct_val": top_5_pct_val,
    }

    strategies = ["random", "greedy", "gp_ucb"]

    # Run budget sweep
    sweep_summary_list: list[dict[str, Any]] = []
    sweep_seed_rows: list[dict[str, Any]] = []

    # Keep track of primary benchmark trajectories (for total_budget)
    primary_trajectories: list[dict[str, Any]] = []

    for budget in budgets_to_sweep:
        if budget < initial_protocols:
            continue
        queries = budget - initial_protocols
        budget_trajectories: list[dict[str, Any]] = []

        for seed in range(n_seeds):
            seed_rng = np.random.default_rng(seed + 1000)
            init_indices = list(seed_rng.choice(len(candidate_pool), size=initial_protocols, replace=False))

            for strat in strategies:
                strat_rng = np.random.default_rng(seed + 2000 + strategies.index(strat))
                traj = run_single_optimization_trajectory(
                    candidate_pool=candidate_pool,
                    oracle=oracle,
                    feature_cols=feature_cols,
                    strategy=strat,
                    init_indices=init_indices,
                    total_queries=queries,
                    evaluator_meta=evaluator_meta,
                    rng=strat_rng,
                    beta=1.0,
                )
                # Compute seed-level summary metrics
                steps_regret = [p["simple_regret"] for p in traj]
                auc_val = float(np.mean(steps_regret))
                final_regret = float(traj[-1]["simple_regret"])
                final_best = float(traj[-1]["best_seen"])
                hit_10 = int(traj[-1]["hit_top_10_pct"])
                hit_5 = int(traj[-1]["hit_top_5_pct"])

                hit_10_steps = [p["step"] for p in traj if p["hit_top_10_pct"] == 1]
                q_10 = int(min(hit_10_steps)) if hit_10_steps else queries + 1

                hit_5_steps = [p["step"] for p in traj if p["hit_top_5_pct"] == 1]
                q_5 = int(min(hit_5_steps)) if hit_5_steps else queries + 1

                sweep_seed_rows.append(
                    {
                        "budget": budget,
                        "initial_protocols": initial_protocols,
                        "queries": queries,
                        "seed": seed,
                        "strategy": strat,
                        "final_best_seen": final_best,
                        "final_simple_regret": final_regret,
                        "regret_auc": auc_val,
                        "hit_top_10_pct": hit_10,
                        "hit_top_5_pct": hit_5,
                        "queries_to_top_10_pct": q_10,
                        "queries_to_top_5_pct": q_5,
                    }
                )

                for point in traj:
                    point["seed"] = seed
                    point["budget"] = budget
                    budget_trajectories.append(point)

        budget_traj_df = pd.DataFrame(budget_trajectories)
        if budget == total_budget:
            primary_trajectories = budget_trajectories

        # Summarize this budget
        budget_strat_summary: dict[str, Any] = {}
        for strat in strategies:
            strat_pts = [r for r in sweep_seed_rows if r["budget"] == budget and r["strategy"] == strat]
            regrets = np.array([r["final_simple_regret"] for r in strat_pts])
            aucs = np.array([r["regret_auc"] for r in strat_pts])
            best_seens = np.array([r["final_best_seen"] for r in strat_pts])
            hit10s = np.array([r["hit_top_10_pct"] for r in strat_pts])
            hit5s = np.array([r["hit_top_5_pct"] for r in strat_pts])
            q10s = np.array([r["queries_to_top_10_pct"] for r in strat_pts])
            q5s = np.array([r["queries_to_top_5_pct"] for r in strat_pts])

            ci_low, ci_high = compute_bootstrap_ci(regrets, n_bootstraps=2000, ci=0.95)

            budget_strat_summary[strat] = {
                "mean_best_seen": float(np.mean(best_seens)),
                "std_best_seen": float(np.std(best_seens)),
                "median_best_seen": float(np.median(best_seens)),
                "mean_simple_regret": float(np.mean(regrets)),
                "std_simple_regret": float(np.std(regrets)),
                "median_simple_regret": float(np.median(regrets)),
                "simple_regret_95_ci": [ci_low, ci_high],
                "mean_regret_auc": float(np.mean(aucs)),
                "median_regret_auc": float(np.median(aucs)),
                "top_10_pct_hit_rate": float(np.mean(hit10s)),
                "top_5_pct_hit_rate": float(np.mean(hit5s)),
                "mean_queries_to_top_10_pct": float(np.mean(q10s)),
                "median_queries_to_top_10_pct": float(np.median(q10s)),
                "mean_queries_to_top_5_pct": float(np.mean(q5s)),
                "median_queries_to_top_5_pct": float(np.median(q5s)),
            }

        paired_comp = compute_paired_comparison(budget_traj_df, total_queries=queries)

        sweep_summary_list.append(
            {
                "budget": budget,
                "initial_protocols": initial_protocols,
                "queries": queries,
                "strategies": budget_strat_summary,
                "paired_comparison_ucb_vs_greedy": paired_comp,
            }
        )

    # Save budget sweep artifacts
    sweep_df = pd.DataFrame(sweep_seed_rows)
    sweep_df.to_csv(output_dir / "budget_sweep.csv", index=False)

    budget_sweep_summary = {
        "benchmark": "Dynamic Cycling Low-Budget Optimization Sweep",
        "universe_protocols": n_candidates,
        "n_seeds": n_seeds,
        "budgets_evaluated": budgets_to_sweep,
        "results_by_budget": sweep_summary_list,
    }
    (output_dir / "budget_sweep_summary.json").write_text(
        json.dumps(budget_sweep_summary, indent=2), encoding="utf-8"
    )

    # Primary trajectory history
    history_df = pd.DataFrame(primary_trajectories)
    history_df.to_csv(output_dir / "optimization_history.csv", index=False)

    # Primary budget summary
    primary_budget_summary = next(
        (s for s in sweep_summary_list if s["budget"] == total_budget),
        sweep_summary_list[-1],
    )

    benchmark_summary = {
        "benchmark": "Dynamic Cycling 2024 Protocol Optimization Benchmark",
        "universe_protocols": n_candidates,
        "total_cells": len(hidden_oracle_df),
        "design_features": feature_cols,
        "evaluator_thresholds": {
            "global_max": global_max,
            "top_10_pct_threshold": top_10_pct_val,
            "top_5_pct_threshold": top_5_pct_val,
        },
        "surrogate_evaluation": surrogate_metrics,
        "optimization_parameters": {
            "initial_protocols": initial_protocols,
            "total_budget": total_budget,
            "total_queries": total_budget - initial_protocols,
            "n_seeds": n_seeds,
        },
        "strategy_comparison": primary_budget_summary["strategies"],
        "paired_comparison_ucb_vs_greedy": primary_budget_summary["paired_comparison_ucb_vs_greedy"],
        "budget_sweep": sweep_summary_list,
    }

    # Save outputs
    compute_replicate_feature_differences(
        hidden_oracle_df, output_path=output_dir / "replicate_feature_differences.csv"
    )
    (output_dir / "model_metrics.json").write_text(json.dumps(surrogate_metrics, indent=2), encoding="utf-8")
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(benchmark_summary, indent=2), encoding="utf-8"
    )

    return benchmark_summary


def main() -> None:
    summary = run_dynamic_cycling_benchmark()
    print("=" * 80)
    print("DYNAMIC CYCLING 2024 BENCHMARK RESULTS")
    print("=" * 80)
    surr = summary["surrogate_evaluation"]
    params = summary["optimization_parameters"]
    print(f"Protocols Universe: {summary['universe_protocols']} protocols ({summary['total_cells']} replicate cells)")
    print(f"Primary Budget: {params['initial_protocols']} initial + {params['total_queries']} queries = {params['total_budget']} total experiments ({params['n_seeds']} paired seeds)")
    print(f"Surrogate RF (Cell Test)   -> MAE: {surr['cell_level']['random_forest']['mae']:.2f}, RMSE: {surr['cell_level']['random_forest']['rmse']:.2f}, R2: {surr['cell_level']['random_forest']['r2']:.3f}")
    print(f"Surrogate GP (Cell Test)   -> MAE: {surr['cell_level']['gaussian_process']['mae']:.2f}, RMSE: {surr['cell_level']['gaussian_process']['rmse']:.2f}, R2: {surr['cell_level']['gaussian_process']['r2']:.3f}")
    if "protocol_level" in surr:
        rf_p = surr["protocol_level"]["random_forest"]
        gp_p = surr["protocol_level"]["gaussian_process"]
        print(f"Surrogate RF (Protocol CV) -> MAE: {rf_p['mean_mae']:.2f} ± {rf_p['std_mae']:.2f}, RMSE: {rf_p['mean_rmse']:.2f} ± {rf_p['std_rmse']:.2f}, R2: {rf_p['mean_r2']:.3f} ± {rf_p['std_r2']:.3f}")
        print(f"Surrogate GP (Protocol CV) -> MAE: {gp_p['mean_mae']:.2f} ± {gp_p['std_mae']:.2f}, RMSE: {gp_p['mean_rmse']:.2f} ± {gp_p['std_rmse']:.2f}, R2: {gp_p['mean_r2']:.3f} ± {gp_p['std_r2']:.3f}")
    print("\nPrimary Budget (20) Strategy Comparison (Paired Seeds, 95% Bootstrap CI):")
    print(f"{'Strategy':<10} {'Mean Best':<12} {'Med Best':<10} {'Mean Regret':<14} {'95% CI':<18} {'Top 10%':<10} {'Top 5%'}")
    print("-" * 88)
    for strat, res in summary["strategy_comparison"].items():
        ci_str = f"[{res['simple_regret_95_ci'][0]:.1f}, {res['simple_regret_95_ci'][1]:.1f}]"
        print(
            f"{strat.upper():<10} {res['mean_best_seen']:<12.2f} {res['median_best_seen']:<10.2f} "
            f"{res['mean_simple_regret']:<14.2f} {ci_str:<18} "
            f"{res['top_10_pct_hit_rate']*100:<9.1f}% {res['top_5_pct_hit_rate']*100:.1f}%"
        )

    print("\nBudget Sweep Summary (Budgets 8, 10, 12, 15, 20):")
    print(f"{'Budget':<8} {'Strategy':<10} {'Mean Regret':<14} {'Regret AUC':<14} {'Top 10% Hit':<14} {'Top 5% Hit'}")
    print("-" * 75)
    for b_item in summary["budget_sweep"]:
        b = b_item["budget"]
        for s_name, s_res in b_item["strategies"].items():
            print(
                f"{b:<8} {s_name.upper():<10} {s_res['mean_simple_regret']:<14.2f} "
                f"{s_res['mean_regret_auc']:<14.2f} {s_res['top_10_pct_hit_rate']*100:<13.1f}% "
                f"{s_res['top_5_pct_hit_rate']*100:.1f}%"
            )
    print("=" * 80)


if __name__ == "__main__":
    main()


