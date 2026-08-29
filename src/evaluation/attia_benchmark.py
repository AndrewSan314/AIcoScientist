from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

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
    generate_attia_simulator_seed,
)

logger = logging.getLogger(__name__)


def compute_bootstrap_mean_ci(
    data: np.ndarray,
    n_bootstraps: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes a deterministic non-parametric bootstrap confidence interval for the sample mean.

    Args:
        data: 1D array of sample values (e.g. across 30 benchmark seeds).
        n_bootstraps: Number of bootstrap resamples (default: 2000).
        ci: Confidence level (default: 0.95).
        seed: Deterministic seed for reproducible bootstrap RNG.

    Returns:
        (lower_ci, upper_ci) for the sample mean.
    """
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    if n <= 1 or np.all(arr == arr[0]):
        val = float(arr[0]) if n > 0 else 0.0
        return val, val

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))
    boot_means = np.mean(arr[boot_indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return lower, upper


def run_single_attia_optimization_trajectory(
    candidate_pool: pd.DataFrame,
    oracle: AttiaSimulatorOracle,
    feature_cols: list[str],
    strategy: str,  # "random", "greedy", "gp_ucb"
    init_indices: list[int],
    total_queries: int,
    optimizer_seed: int,
    beta: float = 1.0,
) -> list[dict[str, Any]]:
    """Runs a single closed-loop optimization trajectory on the Attia candidate space.

    HARD OPTIMIZER / EVALUATOR SEPARATION:
    - candidate_pool contains ONLY policy_id, C1, C2, C3, C4.
    - Zero evaluator reference data (no global max, no true reference lifetime, no thresholds).
    - Fair stochastic seeding: simulator seed is deterministically derived ONLY from (optimizer_seed, policy_id).
    - Records ONLY information available during physical/simulated experimental execution.
    """
    # Initial warm-up observations
    observed_records: list[dict[str, Any]] = []
    seen_pids: set[str] = set()

    for idx in init_indices:
        cand_row = candidate_pool.iloc[idx]
        pid = str(cand_row["policy_id"])
        # Fair stochastic seed: depends ONLY on benchmark_seed + policy_id
        sim_seed = generate_attia_simulator_seed(benchmark_seed=optimizer_seed, policy_id=pid)
        response = oracle.query(cand_row, seed=sim_seed)
        observed_records.append(
            {
                "policy_id": pid,
                "C1": float(cand_row["C1"]),
                "C2": float(cand_row["C2"]),
                "C3": float(cand_row["C3"]),
                "C4": float(cand_row["C4"]),
                "simulator_seed": sim_seed,
                "simulated_lifetime": float(response.target),
            }
        )
        seen_pids.add(pid)

    history: list[dict[str, Any]] = []
    current_best_sim = max(r["simulated_lifetime"] for r in observed_records)

    # Initial state (Step 0 summary after warmups)
    history.append(
        {
            "benchmark_seed": optimizer_seed,
            "strategy": strategy,
            "step": 0,
            "policy_id": None,
            "C1": None,
            "C2": None,
            "C3": None,
            "C4": None,
            "simulator_seed": None,
            "simulated_lifetime": None,
            "best_observed_lifetime": current_best_sim,
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
            # Build training data strictly from observed experimental outcomes
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

        # Fair stochastic seed: depends ONLY on benchmark_seed + policy_id
        sim_seed = generate_attia_simulator_seed(benchmark_seed=optimizer_seed, policy_id=selected_pid)
        response = oracle.query(selected_cand, seed=sim_seed)

        observed_records.append(
            {
                "policy_id": selected_pid,
                "C1": float(selected_cand["C1"]),
                "C2": float(selected_cand["C2"]),
                "C3": float(selected_cand["C3"]),
                "C4": float(selected_cand["C4"]),
                "simulator_seed": sim_seed,
                "simulated_lifetime": float(response.target),
            }
        )
        seen_pids.add(selected_pid)
        current_best_sim = max(current_best_sim, float(response.target))

        history.append(
            {
                "benchmark_seed": optimizer_seed,
                "strategy": strategy,
                "step": step,
                "policy_id": selected_pid,
                "C1": float(selected_cand["C1"]),
                "C2": float(selected_cand["C2"]),
                "C3": float(selected_cand["C3"]),
                "C4": float(selected_cand["C4"]),
                "simulator_seed": sim_seed,
                "simulated_lifetime": float(response.target),
                "best_observed_lifetime": current_best_sim,
            }
        )

    return history


def evaluate_trajectory_metrics(
    raw_history: list[dict[str, Any]],
    init_pids: list[str],
    ref_lookup: dict[str, float],
    global_max: float,
    top_10_pct_val: float,
    top_5_pct_val: float,
) -> list[dict[str, Any]]:
    """Evaluator stage: Joins raw optimizer trajectory with the latent reference landscape.

    Computes ground-truth regret, hit rates, and regret AUC strictly in post-processing.
    """
    evaluated_history: list[dict[str, Any]] = []

    # Warmup policies evaluation
    warmup_ref_vals = [ref_lookup[pid] for pid in init_pids]
    best_ref_so_far = max(warmup_ref_vals)

    for entry in raw_history:
        row = dict(entry)
        pid = row["policy_id"]

        if row["step"] == 0 or pid is None:
            ref_val = None
        else:
            ref_val = ref_lookup[str(pid)]
            best_ref_so_far = max(best_ref_so_far, ref_val)

        row["reference_true_lifetime"] = ref_val
        row["best_reference_true"] = best_ref_so_far
        row["simple_regret"] = max(0.0, global_max - best_ref_so_far)
        row["hit_top_10_pct"] = int(best_ref_so_far >= top_10_pct_val)
        row["hit_top_5_pct"] = int(best_ref_so_far >= top_5_pct_val)

        evaluated_history.append(row)

    return evaluated_history


def run_attia_optimization_benchmark(
    adapter: AttiaAdapter,
    budgets: Sequence[int] = (10, 15, 20, 30),
    initial_policies: int = 5,
    n_seeds: int = 30,
    beta: float = 1.0,
    output_dir: Path | str | None = None,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Runs closed-loop Bayesian Optimization benchmark across multiple budgets and paired seeds.

    Scientific Guarantees:
    1. Fair stochastic seeding: seed = sha256(benchmark_seed + policy_id) (independent of strategy and step).
    2. Strict optimizer/evaluator separation: optimizer loop has zero reference table visibility.
    3. Latent true objective: regret computed against deterministic reference_true_lifetime (variance=False).
    4. Rigorous uncertainty: computes both 95% empirical outcome interval and deterministic bootstrap 95% CI of mean.
    5. Cache provenance: reference landscape verified via manifest.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "outputs" / "attia"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Candidate Pool & Strict Simulator Oracle
    candidate_pool = adapter.load_candidate_pool(force_recompute=force_recompute)
    feature_cols = list(adapter.spec.feature_columns)
    oracle = AttiaSimulatorOracle(candidate_pool, mode="hi", variance=True)

    # 2. Reference Performance Landscape (Evaluator Only)
    ref_df, ref_meta = compute_or_load_reference_landscape(
        adapter,
        output_path=output_dir / "reference_landscape.csv",
        force_recompute=force_recompute,
    )
    # Use latent true deterministic lifetime for benchmark regret
    ref_lookup = dict(zip(ref_df["policy_id"].astype(str), ref_df["reference_true_lifetime"].astype(float)))
    global_max = float(ref_meta["global_max"])
    top_10_pct_val = float(ref_meta["top_10_pct_val"])
    top_5_pct_val = float(ref_meta["top_5_pct_val"])

    strategies = ["random", "greedy", "gp_ucb"]
    all_evaluated_trajectories: list[dict[str, Any]] = []
    budget_sweep_results: list[dict[str, Any]] = []

    # Maximum budget queries
    max_budget = max(budgets)
    max_queries = max_budget - initial_policies

    # Pre-run full trajectories for each seed to allow exact budget slicing
    full_evaluated_by_seed: dict[int, dict[str, list[dict[str, Any]]]] = {}

    for seed_idx in range(n_seeds):
        seed_rng = np.random.default_rng(seed_idx * 7919 + 42)
        init_indices = seed_rng.choice(len(candidate_pool), size=initial_policies, replace=False).tolist()
        init_pids = [str(candidate_pool.iloc[i]["policy_id"]) for i in init_indices]

        full_evaluated_by_seed[seed_idx] = {}
        for strat in strategies:
            # 1. Run pure optimizer trajectory (ZERO evaluator data passed)
            raw_hist = run_single_attia_optimization_trajectory(
                candidate_pool=candidate_pool,
                oracle=oracle,
                feature_cols=feature_cols,
                strategy=strat,
                init_indices=init_indices,
                total_queries=max_queries,
                optimizer_seed=seed_idx,
                beta=beta,
            )
            # 2. Evaluator stage joins with reference landscape
            eval_hist = evaluate_trajectory_metrics(
                raw_history=raw_hist,
                init_pids=init_pids,
                ref_lookup=ref_lookup,
                global_max=global_max,
                top_10_pct_val=top_10_pct_val,
                top_5_pct_val=top_5_pct_val,
            )
            full_evaluated_by_seed[seed_idx][strat] = eval_hist

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
                hist = full_evaluated_by_seed[seed_idx][strat]
                sliced_hist = [h for h in hist if h["step"] <= n_queries]

                # Final step metrics
                final_row = sliced_hist[-1]
                final_regret = float(final_row["simple_regret"])
                final_best_ref = float(final_row["best_reference_true"])

                strat_regrets[strat].append(final_regret)
                strat_best_means[strat].append(final_best_ref)

                # Regret AUC (mean simple regret across steps 0..n_queries)
                regret_series = [h["simple_regret"] for h in sliced_hist]
                auc_val = float(np.mean(regret_series))
                strat_aucs[strat].append(auc_val)

                # Hit rates
                hit_10 = int(any(h["hit_top_10_pct"] for h in sliced_hist))
                hit_5 = int(any(h["hit_top_5_pct"] for h in sliced_hist))
                strat_top10_hits[strat].append(hit_10)
                strat_top5_hits[strat].append(hit_5)

                # Queries to reach threshold (1..n_queries, or n_queries + 1 if not reached)
                step_10 = next((h["step"] for h in sliced_hist if h["hit_top_10_pct"]), n_queries + 1)
                step_5 = next((h["step"] for h in sliced_hist if h["hit_top_5_pct"]), n_queries + 1)
                strat_q_top10[strat].append(float(step_10))
                strat_q_top5[strat].append(float(step_5))

                if budget == max_budget:
                    for h in sliced_hist:
                        all_evaluated_trajectories.append({**h, "budget": budget})

        for strat in strategies:
            regs = np.array(strat_regrets[strat], dtype=float)
            bests = np.array(strat_best_means[strat], dtype=float)
            aucs = np.array(strat_aucs[strat], dtype=float)
            top10 = np.array(strat_top10_hits[strat], dtype=float)
            top5 = np.array(strat_top5_hits[strat], dtype=float)
            q10 = np.array(strat_q_top10[strat], dtype=float)
            q5 = np.array(strat_q_top5[strat], dtype=float)

            # Empirical outcome interval across seeds
            reg_emp_low, reg_emp_high = np.percentile(regs, [2.5, 97.5])
            # Deterministic bootstrap 95% CI of the sample mean
            reg_mean_ci_low, reg_mean_ci_high = compute_bootstrap_mean_ci(regs, n_bootstraps=2000, seed=42 + budget)
            best_mean_ci_low, best_mean_ci_high = compute_bootstrap_mean_ci(bests, n_bootstraps=2000, seed=1042 + budget)
            auc_mean_ci_low, auc_mean_ci_high = compute_bootstrap_mean_ci(aucs, n_bootstraps=2000, seed=2042 + budget)

            strategy_metrics[strat] = {
                "mean_best_seen": float(np.mean(bests)),
                "std_best_seen": float(np.std(bests, ddof=1)) if len(bests) > 1 else 0.0,
                "median_best_seen": float(np.median(bests)),
                "mean_best_seen_95_ci": [float(best_mean_ci_low), float(best_mean_ci_high)],
                "mean_simple_regret": float(np.mean(regs)),
                "std_simple_regret": float(np.std(regs, ddof=1)) if len(regs) > 1 else 0.0,
                "median_simple_regret": float(np.median(regs)),
                "mean_simple_regret_95_ci": [float(reg_mean_ci_low), float(reg_mean_ci_high)],
                "simple_regret_95_empirical_interval": [float(reg_emp_low), float(reg_emp_high)],
                "mean_regret_auc": float(np.mean(aucs)),
                "median_regret_auc": float(np.median(aucs)),
                "mean_regret_auc_95_ci": [float(auc_mean_ci_low), float(auc_mean_ci_high)],
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

    # Save complete optimization history CSV (fully reconstructable queries + evaluator columns)
    trajectories_df = pd.DataFrame(all_evaluated_trajectories)
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
                    "mean_best_seen_95_ci_low": sm["mean_best_seen_95_ci"][0],
                    "mean_best_seen_95_ci_high": sm["mean_best_seen_95_ci"][1],
                    "mean_simple_regret": sm["mean_simple_regret"],
                    "std_simple_regret": sm["std_simple_regret"],
                    "median_simple_regret": sm["median_simple_regret"],
                    "mean_simple_regret_95_ci_low": sm["mean_simple_regret_95_ci"][0],
                    "mean_simple_regret_95_ci_high": sm["mean_simple_regret_95_ci"][1],
                    "simple_regret_95_empirical_interval_low": sm["simple_regret_95_empirical_interval"][0],
                    "simple_regret_95_empirical_interval_high": sm["simple_regret_95_empirical_interval"][1],
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
        "reference_objective": "reference_true_lifetime",
        "reference_objective_description": (
            "reference_true_lifetime is the deterministic latent PDE thermal-degradation objective (variance=False). "
            "simulated_lifetime is the noisy experimental observation draw with cell-to-cell Gaussian variation (sigma=164)."
        ),
        "fair_stochastic_seeding_rule": "sha256(benchmark_seed + policy_id) % (2^31 - 1)",
        "official_repository": "https://github.com/chueh-ermon/battery-fast-charging-optimization",
        "source_commit_sha": "0068fd0136bcd65884f5cd94b2b967c1ba73a668",
        "total_valid_policies": len(candidate_pool),
        "design_features": feature_cols,
        "evaluator_thresholds": {
            "global_max_true": global_max,
            "top_10_pct_threshold": top_10_pct_val,
            "top_5_pct_threshold": top_5_pct_val,
        },
        "optimization_parameters": {
            "initial_policies": initial_policies,
            "total_budget": max_budget,
            "total_queries": max_queries,
            "n_seeds": n_seeds,
            "beta": beta,
            "bootstrap_replicates": 2000,
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
