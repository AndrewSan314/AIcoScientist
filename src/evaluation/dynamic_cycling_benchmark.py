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
    protocols_df: pd.DataFrame,
    oracle: OfflineOracle,
    feature_cols: list[str],
    strategy: str,  # "random", "greedy", "gp_ucb"
    initial_protocols: int,
    budget: int,
    rng: np.random.Generator,
    beta: float = 1.0,
) -> list[dict[str, Any]]:
    """Runs a single closed-loop optimization trajectory without target leakage."""
    n_total = len(protocols_df)
    all_indices = list(range(n_total))
    global_max = float(protocols_df["target_mean"].max())

    # Calculate top-k thresholds
    sorted_targets = np.sort(protocols_df["target_mean"].to_numpy(dtype=float))[::-1]
    top_10_pct_val = sorted_targets[max(0, int(np.ceil(0.10 * n_total)) - 1)]
    top_5_pct_val = sorted_targets[max(0, int(np.ceil(0.05 * n_total)) - 1)]

    # Initial random sample of protocols
    init_indices = list(rng.choice(all_indices, size=initial_protocols, replace=False))
    observed_protocols: list[dict[str, Any]] = []

    for idx in init_indices:
        cand_row = protocols_df.iloc[idx]
        cand_dict = {col: cand_row[col] for col in feature_cols}
        cand_dict["protocol_id"] = cand_row["protocol_id"]
        response = oracle.query(cand_dict)
        observed_protocols.append(
            {
                "protocol_id": cand_row["protocol_id"],
                "target": response.target,
                **cand_dict,
            }
        )

    history: list[dict[str, Any]] = []
    seen_targets = [row["target"] for row in observed_protocols]
    current_best = max(seen_targets)

    # Initial state record (step 0)
    history.append(
        {
            "step": 0,
            "strategy": strategy,
            "best_seen": current_best,
            "simple_regret": global_max - current_best,
            "hit_top_10_pct": int(current_best >= top_10_pct_val),
            "hit_top_5_pct": int(current_best >= top_5_pct_val),
        }
    )

    observed_df = pd.DataFrame(observed_protocols)

    # Closed-loop BO loop for remaining budget
    for step in range(1, budget + 1):
        observed_ids = set(observed_df["protocol_id"].astype(str))
        # Form candidate pool containing strictly unseen protocols
        unseen_mask = ~protocols_df["protocol_id"].astype(str).isin(observed_ids)
        unseen_cands = protocols_df[unseen_mask].copy()

        if unseen_cands.empty:
            break

        if strategy == "random":
            chosen_cand_idx = rng.choice(unseen_cands.index)
            chosen_row = protocols_df.loc[chosen_cand_idx]
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
            }
        )

    return history


def run_dynamic_cycling_benchmark(
    adapter: DynamicCyclingAdapter | None = None,
    output_dir: Path | None = None,
    initial_protocols: int = 5,
    budget: int = 15,
    n_seeds: int = 50,
) -> dict[str, Any]:
    """Runs surrogate evaluation and closed-loop BO comparison across Random, Greedy, and GP-UCB."""
    if adapter is None:
        adapter = DynamicCyclingAdapter()

    project_root = Path(__file__).resolve().parent.parent.parent
    if output_dir is None:
        output_dir = project_root / "outputs" / "dynamic_cycling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Surrogate prediction benchmark
    surrogate_metrics = evaluate_surrogate_prediction(adapter)

    # 2. Closed-loop BO benchmark
    protocols_df = adapter.load_protocols()
    oracle = OfflineOracle(protocols_df, adapter.spec, replicate_policy="mean")
    feature_cols = list(adapter.spec.feature_columns)

    strategies = ["random", "greedy", "gp_ucb"]
    all_trajectories: list[dict[str, Any]] = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 1000)
        for strat in strategies:
            # Re-seed RNG with fixed sequence per seed/strategy for fair initialization comparison
            strat_rng = np.random.default_rng(seed + 1000)
            traj = run_single_optimization_trajectory(
                protocols_df=protocols_df,
                oracle=oracle,
                feature_cols=feature_cols,
                strategy=strat,
                initial_protocols=initial_protocols,
                budget=budget,
                rng=strat_rng,
            )
            for point in traj:
                point["seed"] = seed
                all_trajectories.append(point)

    history_df = pd.DataFrame(all_trajectories)
    history_df.to_csv(output_dir / "optimization_history.csv", index=False)

    # Aggregate optimization metrics per strategy at final step
    final_step = history_df[history_df["step"] == budget]
    summary_by_strat: dict[str, Any] = {}

    for strat in strategies:
        strat_df = final_step[final_step["strategy"] == strat]
        summary_by_strat[strat] = {
            "mean_best_seen": float(strat_df["best_seen"].mean()),
            "std_best_seen": float(strat_df["best_seen"].std()),
            "mean_simple_regret": float(strat_df["simple_regret"].mean()),
            "top_10_pct_hit_rate": float(strat_df["hit_top_10_pct"].mean()),
            "top_5_pct_hit_rate": float(strat_df["hit_top_5_pct"].mean()),
        }

    benchmark_summary = {
        "benchmark": "Dynamic Cycling 2024 Protocol Optimization Benchmark",
        "universe_protocols": len(protocols_df),
        "total_cells": len(adapter.load_cells()),
        "design_features": feature_cols,
        "surrogate_evaluation": surrogate_metrics,
        "optimization_parameters": {
            "initial_protocols": initial_protocols,
            "budget": budget,
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
    print("=" * 70)
    print("DYNAMIC CYCLING 2024 BENCHMARK RESULTS")
    print("=" * 70)
    surr = summary["surrogate_evaluation"]
    print(f"Protocols Universe: {summary['universe_protocols']} protocols ({summary['total_cells']} replicate cells)")
    print(f"Surrogate RF  -> Test MAE: {surr['random_forest']['mae']:.2f}, RMSE: {surr['random_forest']['rmse']:.2f}, R2: {surr['random_forest']['r2']:.3f}")
    print(f"Surrogate GP  -> Test MAE: {surr['gaussian_process']['mae']:.2f}, RMSE: {surr['gaussian_process']['rmse']:.2f}, R2: {surr['gaussian_process']['r2']:.3f}")
    print("\nOffline Closed-Loop BO Comparison (Budget=15, 50 Seeds):")
    print(f"{'Strategy':<12} {'Mean Best Lifetime':<20} {'Mean Simple Regret':<20} {'Top 10% Hit Rate':<18} {'Top 5% Hit Rate'}")
    print("-" * 85)
    for strat, res in summary["strategy_comparison"].items():
        print(
            f"{strat.upper():<12} {res['mean_best_seen']:<20.2f} {res['mean_simple_regret']:<20.2f} {res['top_10_pct_hit_rate']*100:<17.1f}% {res['top_5_pct_hit_rate']*100:.1f}%"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
