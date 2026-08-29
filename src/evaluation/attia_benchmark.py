from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from src.datasets.attia import ATTIA_FEATURE_COLUMNS, AttiaAdapter
from src.evaluation.attia_oracle import (
    AttiaSimulatorOracle,
    compute_or_load_reference_landscape,
)


def run_single_attia_optimization_trajectory(
    candidate_pool: pd.DataFrame,
    oracle: AttiaSimulatorOracle,
    feature_cols: list[str],
    strategy: str,  # "random", "greedy", "gp_ucb"
    init_indices: list[int],
    total_queries: int,
    evaluator_meta: dict[str, Any],
    optimizer_seed: int,
    beta: float = 1.0,
) -> list[dict[str, Any]]:
    """Runs a single closed-loop optimization trajectory on the Attia candidate space.

    Visibility & Anti-Leakage Guarantees:
    - candidate_pool contains ONLY policy_id, C1, C2, C3, C4.
    - Ground truth/evaluator reference table is strictly isolated from model training & acquisition.
    - Oracle returns noisy simulated_lifetime.
    - Evaluator-only regret is computed against the pre-calculated expected reference optimum.
    - Paired seed schedule ensures identical stochastic simulation noise across strategies.
    """
    global_max = float(evaluator_meta["global_max"])
    top_10_pct_val = float(evaluator_meta["top_10_pct_val"])
    top_5_pct_val = float(evaluator_meta["top_5_pct_val"])
    ref_lookup: dict[str, float] = evaluator_meta["ref_lookup"]

    # Initial warm-up observations
    observed_records: list[dict[str, Any]] = []
    seen_pids: set[str] = set()

    for step_init, idx in enumerate(init_indices):
        cand_row = candidate_pool.iloc[idx]
        pid = str(cand_row["policy_id"])
        # Deterministic paired simulator seed
        sim_seed = int((optimizer_seed * 10000 + step_init * 100 + idx) % (2**31 - 1))
        response = oracle.query(cand_row, seed=sim_seed)
        observed_records.append(
            {
                "policy_id": pid,
                "C1": float(cand_row["C1"]),
                "C2": float(cand_row["C2"]),
                "C3": float(cand_row["C3"]),
                "C4": float(cand_row["C4"]),
                "simulated_lifetime": response.target,
                "reference_mean_lifetime": ref_lookup[pid],
            }
        )
        seen_pids.add(pid)

    history: list[dict[str, Any]] = []

    # Best expected reference value found among evaluated policies
    best_ref_val = max(r["reference_mean_lifetime"] for r in observed_records)
    best_sim_val = max(r["simulated_lifetime"] for r in observed_records)

    # Initial state (Step 0)
    history.append(
        {
            "step": 0,
            "strategy": strategy,
            "seed": optimizer_seed,
            "best_simulated_lifetime": best_sim_val,
            "best_reference_mean": best_ref_val,
            "simple_regret": max(0.0, global_max - best_ref_val),
            "hit_top_10_pct": int(best_ref_val >= top_10_pct_val),
            "hit_top_5_pct": int(best_ref_val >= top_5_pct_val),
            "queried_policy_id": None,
        }
    )

    # Closed-loop BO iterations
    for step in range(1, total_queries + 1):
        unseen_mask = ~candidate_pool["policy_id"].isin(seen_pids)
        unseen_pool = candidate_pool[unseen_mask].copy().reset_index(drop=True)

        if unseen_pool.empty:
            break

        if strategy == "random":
            rng_step = np.random.default_rng(optimizer_seed + step * 1000)
            selected_idx = int(rng_step.integers(0, len(unseen_pool)))
            selected_cand = unseen_pool.iloc[selected_idx]

        elif strategy in {"greedy", "gp_ucb"}:
            # Build training data from revealed observations
            X_train = np.array([[r[c] for c in feature_cols] for r in observed_records], dtype=float)
            y_train = np.array([r["simulated_lifetime"] for r in observed_records], dtype=float)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Fit GP surrogate
            kernel = ConstantKernel(1.0, (1e-2, 1e4)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
                noise_level=1.0, noise_level_bounds=(1e-5, 1e2)
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=optimizer_seed,
            )
            gp.fit(X_train_scaled, y_train)

            # Predict on unseen candidates
            X_unseen = unseen_pool[feature_cols].to_numpy(dtype=float)
            X_unseen_scaled = scaler.transform(X_unseen)

            if strategy == "greedy":
                pred_mean = gp.predict(X_unseen_scaled)
                selected_idx = int(np.argmax(pred_mean))
            else:  # "gp_ucb"
                pred_mean, pred_std = gp.predict(X_unseen_scaled, return_std=True)
                ucb_scores = pred_mean + beta * pred_std
                selected_idx = int(np.argmax(ucb_scores))

            selected_cand = unseen_pool.iloc[selected_idx]

        else:
            raise ValueError(f"Unknown optimization strategy: {strategy!r}")

        selected_pid = str(selected_cand["policy_id"])
        cand_orig_idx = int(candidate_pool[candidate_pool["policy_id"] == selected_pid].index[0])

        # Paired simulator seed
        sim_seed = int((optimizer_seed * 10000 + (len(init_indices) + step) * 100 + cand_orig_idx) % (2**31 - 1))
        response = oracle.query(selected_cand, seed=sim_seed)

        ref_val = ref_lookup[selected_pid]
        observed_records.append(
            {
                "policy_id": selected_pid,
                "C1": float(selected_cand["C1"]),
                "C2": float(selected_cand["C2"]),
                "C3": float(selected_cand["C3"]),
                "C4": float(selected_cand["C4"]),
                "simulated_lifetime": response.target,
                "reference_mean_lifetime": ref_val,
            }
        )
        seen_pids.add(selected_pid)

        best_ref_val = max(best_ref_val, ref_val)
        best_sim_val = max(best_sim_val, response.target)

        history.append(
            {
                "step": step,
                "strategy": strategy,
                "seed": optimizer_seed,
                "best_simulated_lifetime": best_sim_val,
                "best_reference_mean": best_ref_val,
                "simple_regret": max(0.0, global_max - best_ref_val),
                "hit_top_10_pct": int(best_ref_val >= top_10_pct_val),
                "hit_top_5_pct": int(best_ref_val >= top_5_pct_val),
                "queried_policy_id": selected_pid,
            }
        )

    return history


def run_attia_optimization_benchmark(
    adapter: AttiaAdapter,
    budgets: Sequence[int] = (10, 15, 20, 30),
    initial_policies: int = 5,
    n_seeds: int = 30,
    beta: float = 1.0,
    output_dir: Path | str | None = None,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Runs low-budget closed-loop Bayesian Optimization benchmark across multiple budgets and paired seeds."""
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "outputs" / "attia"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Candidate Pool & Simulator Oracle
    candidate_pool = adapter.load_candidate_pool(force_recompute=force_recompute)
    feature_cols = list(adapter.spec.feature_columns)
    oracle = AttiaSimulatorOracle(candidate_pool, mode="hi", variance=True)

    # 2. Reference Performance Landscape (Evaluator Only)
    ref_df, ref_meta = compute_or_load_reference_landscape(
        adapter,
        output_path=output_dir / "reference_landscape.csv",
        force_recompute=force_recompute,
    )
    ref_lookup = dict(zip(ref_df["policy_id"].astype(str), ref_df["reference_mean_lifetime"].astype(float)))
    evaluator_meta = {
        "global_max": ref_meta["global_max"],
        "top_10_pct_val": ref_meta["top_10_pct_val"],
        "top_5_pct_val": ref_meta["top_5_pct_val"],
        "ref_lookup": ref_lookup,
    }

    strategies = ["random", "greedy", "gp_ucb"]
    all_trajectories: list[dict[str, Any]] = []
    budget_sweep_results: list[dict[str, Any]] = []

    # Maximum budget queries
    max_budget = max(budgets)
    max_queries = max_budget - initial_policies

    # Pre-run 50-step full trajectories for each seed to allow exact budget slicing
    full_histories_by_seed: dict[int, dict[str, list[dict[str, Any]]]] = {}

    for seed_idx in range(n_seeds):
        seed_rng = np.random.default_rng(seed_idx * 7919 + 42)
        init_indices = seed_rng.choice(len(candidate_pool), size=initial_policies, replace=False).tolist()

        full_histories_by_seed[seed_idx] = {}
        for strat in strategies:
            hist = run_single_attia_optimization_trajectory(
                candidate_pool=candidate_pool,
                oracle=oracle,
                feature_cols=feature_cols,
                strategy=strat,
                init_indices=init_indices,
                total_queries=max_queries,
                evaluator_meta=evaluator_meta,
                optimizer_seed=seed_idx,
                beta=beta,
            )
            full_histories_by_seed[seed_idx][strat] = hist

    # Compute metrics for each budget
    for budget in budgets:
        n_queries = budget - initial_policies
        strategy_metrics: dict[str, dict[str, Any]] = {}

        # Collect trajectory stats per strategy
        strat_regrets: dict[str, list[float]] = {s: [] for s in strategies}
        strat_best_means: dict[str, list[float]] = {s: [] for s in strategies}
        strat_aucs: dict[str, list[float]] = {s: [] for s in strategies}
        strat_top10_hits: dict[str, list[int]] = {s: [] for s in strategies}
        strat_top5_hits: dict[str, list[int]] = {s: [] for s in strategies}
        strat_q_top10: dict[str, list[float]] = {s: [] for s in strategies}
        strat_q_top5: dict[str, list[float]] = {s: [] for s in strategies}

        for seed_idx in range(n_seeds):
            for strat in strategies:
                hist = full_histories_by_seed[seed_idx][strat]
                sliced_hist = [h for h in hist if h["step"] <= n_queries]

                # Final step metrics
                final_row = sliced_hist[-1]
                final_regret = float(final_row["simple_regret"])
                final_best_ref = float(final_row["best_reference_mean"])

                strat_regrets[strat].append(final_regret)
                strat_best_means[strat].append(final_best_ref)

                # Regret AUC (cumulative average regret over steps)
                regret_series = [h["simple_regret"] for h in sliced_hist]
                auc_val = float(np.mean(regret_series))
                strat_aucs[strat].append(auc_val)

                # Hit rates
                hit_10 = int(any(h["hit_top_10_pct"] for h in sliced_hist))
                hit_5 = int(any(h["hit_top_5_pct"] for h in sliced_hist))
                strat_top10_hits[strat].append(hit_10)
                strat_top5_hits[strat].append(hit_5)

                # Queries to reach threshold
                step_10 = next((h["step"] for h in sliced_hist if h["hit_top_10_pct"]), n_queries + 1)
                step_5 = next((h["step"] for h in sliced_hist if h["hit_top_5_pct"]), n_queries + 1)
                strat_q_top10[strat].append(float(step_10))
                strat_q_top5[strat].append(float(step_5))

                if budget == max_budget:
                    for h in sliced_hist:
                        all_trajectories.append({**h, "budget": budget})

        for strat in strategies:
            regs = np.array(strat_regrets[strat], dtype=float)
            bests = np.array(strat_best_means[strat], dtype=float)
            aucs = np.array(strat_aucs[strat], dtype=float)
            top10 = np.array(strat_top10_hits[strat], dtype=float)
            top5 = np.array(strat_top5_hits[strat], dtype=float)
            q10 = np.array(strat_q_top10[strat], dtype=float)
            q5 = np.array(strat_q_top5[strat], dtype=float)

            reg_ci_low, reg_ci_high = np.percentile(regs, [2.5, 97.5])

            strategy_metrics[strat] = {
                "mean_best_seen": float(np.mean(bests)),
                "std_best_seen": float(np.std(bests, ddof=1)) if len(bests) > 1 else 0.0,
                "median_best_seen": float(np.median(bests)),
                "mean_simple_regret": float(np.mean(regs)),
                "std_simple_regret": float(np.std(regs, ddof=1)) if len(regs) > 1 else 0.0,
                "median_simple_regret": float(np.median(regs)),
                "simple_regret_95_ci": [float(reg_ci_low), float(reg_ci_high)],
                "mean_regret_auc": float(np.mean(aucs)),
                "median_regret_auc": float(np.median(aucs)),
                "top_10_pct_hit_rate": float(np.mean(top10)),
                "top_5_pct_hit_rate": float(np.mean(top5)),
                "mean_queries_to_top_10_pct": float(np.mean(q10)),
                "median_queries_to_top_10_pct": float(np.median(q10)),
                "mean_queries_to_top_5_pct": float(np.mean(q5)),
                "median_queries_to_top_5_pct": float(np.median(q5)),
            }

        # Paired Wilcoxon signed-rank tests: GP-UCB vs Greedy
        ucb_regs = np.array(strat_regrets["gp_ucb"], dtype=float)
        greedy_regs = np.array(strat_regrets["greedy"], dtype=float)
        diff_regrets = ucb_regs - greedy_regs

        ucb_aucs = np.array(strat_aucs["gp_ucb"], dtype=float)
        greedy_aucs = np.array(strat_aucs["greedy"], dtype=float)
        diff_aucs = ucb_aucs - greedy_aucs

        ucb_q5 = np.array(strat_q_top5["gp_ucb"], dtype=float)
        greedy_q5 = np.array(strat_q_top5["greedy"], dtype=float)
        diff_q5 = ucb_q5 - greedy_q5

        def _safe_wilcoxon(d: np.ndarray) -> float:
            if np.all(d == 0):
                return 1.0
            try:
                stat, p_val = wilcoxon(d, alternative="two-sided")
                return float(p_val)
            except Exception:
                return 1.0

        p_val_reg = _safe_wilcoxon(diff_regrets)
        p_val_auc = _safe_wilcoxon(diff_aucs)
        p_val_q5 = _safe_wilcoxon(diff_q5)

        paired_comparison = {
            "n_paired_seeds": n_seeds,
            "regret_diff_ucb_minus_greedy": {
                "mean": float(np.mean(diff_regrets)),
                "median": float(np.median(diff_regrets)),
                "std": float(np.std(diff_regrets, ddof=1)) if len(diff_regrets) > 1 else 0.0,
                "p_value": p_val_reg,
            },
            "auc_diff_ucb_minus_greedy": {
                "mean": float(np.mean(diff_aucs)),
                "median": float(np.median(diff_aucs)),
                "std": float(np.std(diff_aucs, ddof=1)) if len(diff_aucs) > 1 else 0.0,
                "p_value": p_val_auc,
            },
            "queries_to_top5_diff_ucb_minus_greedy": {
                "mean": float(np.mean(diff_q5)),
                "median": float(np.median(diff_q5)),
                "std": float(np.std(diff_q5, ddof=1)) if len(diff_q5) > 1 else 0.0,
                "p_value": p_val_q5,
            },
        }

        budget_sweep_results.append(
            {
                "budget": budget,
                "initial_policies": initial_policies,
                "queries": n_queries,
                "strategies": strategy_metrics,
                "paired_comparison_ucb_vs_greedy": paired_comparison,
            }
        )

    # Save trajectories CSV
    trajectories_df = pd.DataFrame(all_trajectories)
    trajectories_df.to_csv(output_dir / "optimization_history.csv", index=False)

    # Save budget sweep CSV
    sweep_rows: list[dict[str, Any]] = []
    for entry in budget_sweep_results:
        b_val = entry["budget"]
        q_val = entry["queries"]
        for strat, sm in entry["strategies"].items():
            sweep_rows.append(
                {
                    "budget": b_val,
                    "queries": q_val,
                    "strategy": strat,
                    "mean_best_seen": sm["mean_best_seen"],
                    "std_best_seen": sm["std_best_seen"],
                    "median_best_seen": sm["median_best_seen"],
                    "mean_simple_regret": sm["mean_simple_regret"],
                    "std_simple_regret": sm["std_simple_regret"],
                    "median_simple_regret": sm["median_simple_regret"],
                    "mean_regret_auc": sm["mean_regret_auc"],
                    "top_10_pct_hit_rate": sm["top_10_pct_hit_rate"],
                    "top_5_pct_hit_rate": sm["top_5_pct_hit_rate"],
                    "mean_queries_to_top_5_pct": sm["mean_queries_to_top_5_pct"],
                }
            )
    pd.DataFrame(sweep_rows).to_csv(output_dir / "budget_sweep.csv", index=False)

    with open(output_dir / "budget_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(budget_sweep_results, f, indent=2)

    # Main benchmark summary
    final_budget_entry = budget_sweep_results[-1]
    benchmark_summary = {
        "benchmark": "Attia et al. 2020 Fast-Charging Optimization Benchmark",
        "benchmark_nature": "simulator != experimental dataset",
        "official_repository": "https://github.com/chueh-ermon/battery-fast-charging-optimization",
        "source_commit_sha": "0068fd0136bcd65884f5cd94b2b967c1ba73a668",
        "total_valid_policies": len(candidate_pool),
        "design_features": feature_cols,
        "evaluator_thresholds": {
            "global_max": evaluator_meta["global_max"],
            "top_10_pct_threshold": evaluator_meta["top_10_pct_val"],
            "top_5_pct_threshold": evaluator_meta["top_5_pct_val"],
        },
        "optimization_parameters": {
            "initial_policies": initial_policies,
            "total_budget": max_budget,
            "total_queries": max_queries,
            "n_seeds": n_seeds,
            "beta": beta,
        },
        "strategy_comparison": final_budget_entry["strategies"],
        "paired_comparison_ucb_vs_greedy": final_budget_entry["paired_comparison_ucb_vs_greedy"],
        "budget_sweep": budget_sweep_results,
    }

    with open(output_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    return benchmark_summary


def run_attia_benchmark(adapter: AttiaAdapter | None = None) -> dict[str, Any]:
    """Top-level CLI runner for Attia benchmark."""
    if adapter is None:
        adapter = AttiaAdapter()
    return run_attia_optimization_benchmark(adapter)
