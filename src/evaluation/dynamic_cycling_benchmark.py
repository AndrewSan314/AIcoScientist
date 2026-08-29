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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.datasets.dynamic_cycling import (
    DYNAMIC_CYCLING_FEATURE_COLUMNS,
    DynamicCyclingAdapter,
)
from src.evaluation.oracle import OfflineOracle


def evaluate_surrogate_prediction(
    adapter: DynamicCyclingAdapter,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluates surrogate model performance under strict protocol-grouped splitting."""
    cells_df = adapter.load_cells()
    spec = adapter.spec

    feature_cols = list(spec.feature_columns)
    target_col = "efc_lifetime"
    group_col = "protocol_id"

    # Protocol-group-safe split
    groups = cells_df[group_col].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(cells_df, groups=groups))

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    if train_groups & test_groups:
        raise RuntimeError(f"Group leakage detected in dynamic cycling split: {train_groups & test_groups}")

    train_df = cells_df.iloc[train_idx]
    test_df = cells_df.iloc[test_idx]

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)

    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=random_state)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # 2. Gaussian Process
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kernel = ConstantKernel(1.0, (1e-2, 1e4)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
        noise_level=1.0, noise_level_bounds=(1e-5, 1e2)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=random_state,
    )
    gp.fit(X_train_scaled, y_train)
    gp_pred = gp.predict(X_test_scaled)

    def _get_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        return {
            "mae": float(mean_absolute_error(true, pred)),
            "rmse": float(np.sqrt(mean_squared_error(true, pred))),
            "r2": float(r2_score(true, pred)),
        }

    return {
        "n_train_cells": len(train_df),
        "n_test_cells": len(test_df),
        "n_train_protocols": len(train_groups),
        "n_test_protocols": len(test_groups),
        "random_forest": _get_metrics(y_test, rf_pred),
        "gaussian_process": _get_metrics(y_test, gp_pred),
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


def run_dynamic_cycling_benchmark(
    adapter: DynamicCyclingAdapter | None = None,
    output_dir: Path | None = None,
    initial_protocols: int = 5,
    total_budget: int = 20,
    n_seeds: int = 50,
) -> dict[str, Any]:
    """Runs surrogate evaluation and closed-loop BO comparison with paired seeds and bootstrap statistics.

    Budget parameters:
    - initial_protocols: Number of randomly selected protocols before closed-loop BO (default: 5)
    - total_budget: Total number of evaluated protocols (default: 20)
    - total_queries = total_budget - initial_protocols (default: 15 additional queries)
    """
    if adapter is None:
        adapter = DynamicCyclingAdapter()

    project_root = Path(__file__).resolve().parent.parent.parent
    if output_dir is None:
        output_dir = project_root / "outputs" / "dynamic_cycling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Surrogate prediction benchmark
    surrogate_metrics = evaluate_surrogate_prediction(adapter)

    # 2. Setup candidate pool and hidden oracle
    candidate_pool = adapter.load_candidate_pool()
    hidden_oracle_df = adapter.load_hidden_oracle()
    oracle = OfflineOracle(hidden_oracle_df, adapter.spec, replicate_policy="mean")
    feature_cols = list(adapter.spec.feature_columns)

    n_candidates = len(candidate_pool)
    if initial_protocols <= 0:
        raise ValueError("initial_protocols must be a positive integer")
    if total_budget < initial_protocols:
        raise ValueError(f"total_budget ({total_budget}) must be >= initial_protocols ({initial_protocols})")
    if total_budget > n_candidates:
        raise ValueError(f"total_budget ({total_budget}) cannot exceed total candidate protocols ({n_candidates})")

    total_queries = total_budget - initial_protocols

    # Evaluator metrics computed from hidden oracle
    ground_truth_targets = hidden_oracle_df["target_mean"].to_numpy(dtype=float)
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
    all_trajectories: list[dict[str, Any]] = []

    # Run paired trajectories across seeds
    for seed in range(n_seeds):
        seed_rng = np.random.default_rng(seed + 1000)
        # Generate paired initial indices shared identically across all strategies for this seed
        init_indices = list(seed_rng.choice(len(candidate_pool), size=initial_protocols, replace=False))

        for strat in strategies:
            strat_rng = np.random.default_rng(seed + 2000 + strategies.index(strat))
            traj = run_single_optimization_trajectory(
                candidate_pool=candidate_pool,
                oracle=oracle,
                feature_cols=feature_cols,
                strategy=strat,
                init_indices=init_indices,
                total_queries=total_queries,
                evaluator_meta=evaluator_meta,
                rng=strat_rng,
            )
            for point in traj:
                point["seed"] = seed
                all_trajectories.append(point)

    history_df = pd.DataFrame(all_trajectories)
    history_df.to_csv(output_dir / "optimization_history.csv", index=False)

    # Compute trajectory-level statistics and final-step metrics
    summary_by_strat: dict[str, Any] = {}

    for strat in strategies:
        strat_traj = history_df[history_df["strategy"] == strat]
        final_step = strat_traj[strat_traj["step"] == total_queries]

        best_seen_vals = final_step["best_seen"].to_numpy(dtype=float)
        regret_vals = final_step["simple_regret"].to_numpy(dtype=float)
        hit_10_vals = final_step["hit_top_10_pct"].to_numpy(dtype=float)
        hit_5_vals = final_step["hit_top_5_pct"].to_numpy(dtype=float)

        # Calculate experiments to top 10% and top 5%
        steps_to_10: list[int] = []
        steps_to_5: list[int] = []

        for seed, group in strat_traj.groupby("seed"):
            hit_10_steps = group[group["hit_top_10_pct"] == 1]["step"]
            steps_to_10.append(int(hit_10_steps.min()) if not hit_10_steps.empty else total_queries + 1)

            hit_5_steps = group[group["hit_top_5_pct"] == 1]["step"]
            steps_to_5.append(int(hit_5_steps.min()) if not hit_5_steps.empty else total_queries + 1)

        ci_low, ci_high = compute_bootstrap_ci(regret_vals, n_bootstraps=2000, ci=0.95)

        summary_by_strat[strat] = {
            "mean_best_seen": float(np.mean(best_seen_vals)),
            "std_best_seen": float(np.std(best_seen_vals)),
            "median_best_seen": float(np.median(best_seen_vals)),
            "mean_simple_regret": float(np.mean(regret_vals)),
            "std_simple_regret": float(np.std(regret_vals)),
            "median_simple_regret": float(np.median(regret_vals)),
            "simple_regret_95_ci": [ci_low, ci_high],
            "top_10_pct_hit_rate": float(np.mean(hit_10_vals)),
            "top_5_pct_hit_rate": float(np.mean(hit_5_vals)),
            "mean_queries_to_top_10_pct": float(np.mean(steps_to_10)),
            "median_queries_to_top_10_pct": float(np.median(steps_to_10)),
            "mean_queries_to_top_5_pct": float(np.mean(steps_to_5)),
            "median_queries_to_top_5_pct": float(np.median(steps_to_5)),
        }

    benchmark_summary = {
        "benchmark": "Dynamic Cycling 2024 Protocol Optimization Benchmark",
        "universe_protocols": n_candidates,
        "total_cells": len(adapter.load_cells()),
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
            "total_queries": total_queries,
            "n_seeds": n_seeds,
        },
        "strategy_comparison": summary_by_strat,
    }

    # Save outputs
    (output_dir / "model_metrics.json").write_text(json.dumps(surrogate_metrics, indent=2), encoding="utf-8")
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(benchmark_summary, indent=2), encoding="utf-8"
    )

    return benchmark_summary


def main() -> None:
    summary = run_dynamic_cycling_benchmark()
    print("=" * 75)
    print("DYNAMIC CYCLING 2024 BENCHMARK RESULTS")
    print("=" * 75)
    surr = summary["surrogate_evaluation"]
    params = summary["optimization_parameters"]
    print(f"Protocols Universe: {summary['universe_protocols']} protocols ({summary['total_cells']} replicate cells)")
    print(f"Budget: {params['initial_protocols']} initial + {params['total_queries']} queries = {params['total_budget']} total experiments ({params['n_seeds']} paired seeds)")
    print(f"Surrogate RF  -> Test MAE: {surr['random_forest']['mae']:.2f}, RMSE: {surr['random_forest']['rmse']:.2f}, R2: {surr['random_forest']['r2']:.3f}")
    print(f"Surrogate GP  -> Test MAE: {surr['gaussian_process']['mae']:.2f}, RMSE: {surr['gaussian_process']['rmse']:.2f}, R2: {surr['gaussian_process']['r2']:.3f}")
    print("\nOffline Closed-Loop BO Comparison (Paired Seeds, 95% Bootstrap CI):")
    print(f"{'Strategy':<10} {'Mean Best':<12} {'Med Best':<10} {'Mean Regret':<14} {'95% CI':<18} {'Top 10%':<10} {'Top 5%'}")
    print("-" * 88)
    for strat, res in summary["strategy_comparison"].items():
        ci_str = f"[{res['simple_regret_95_ci'][0]:.1f}, {res['simple_regret_95_ci'][1]:.1f}]"
        print(
            f"{strat.upper():<10} {res['mean_best_seen']:<12.2f} {res['median_best_seen']:<10.2f} "
            f"{res['mean_simple_regret']:<14.2f} {ci_str:<18} "
            f"{res['top_10_pct_hit_rate']*100:<9.1f}% {res['top_5_pct_hit_rate']*100:.1f}%"
        )
    print("=" * 75)


if __name__ == "__main__":
    main()

