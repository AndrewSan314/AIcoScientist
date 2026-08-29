from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from src.datasets.attia import AttiaAdapter, compute_expected_c4
from src.evaluation.attia_benchmark import compute_bootstrap_mean_ci
from src.evaluation.attia_oracle import (
    generate_attia_simulator_seed,
    simulate_attia_policy,
)
from src.optimization.search_space import SearchSpace

logger = logging.getLogger(__name__)

DISCRETE_GRID_OPTIMUM_TRUE: float = 1079.0  # Latent true lifetime of best 224-grid policy (ATTIA_P113)


def run_single_attia_continuous_trajectory(
    search_space: SearchSpace,
    discrete_pool: pd.DataFrame,
    init_indices: list[int],
    total_queries: int,
    strategy: str,  # "random", "greedy", "gp_ucb"
    optimizer_seed: int,
    beta: float = 1.0,
    n_candidates_per_step: int = 5000,
    refine_continuous: bool = True,
) -> list[dict[str, Any]]:
    """Runs a single continuous closed-loop Bayesian Optimization trajectory.

    Workflow:
    - Starts with 5 initial discrete warmup protocols (identical across strategies for fairness).
    - At each step:
      1. Generates 5000 feasible continuous candidate proposals.
      2. Predicts with GP surrogate.
      3. Selects optimal proposal via acquisition function + continuous local refinement.
      4. Queries simulator oracle with fair stochastic seed.
    """
    feature_cols = ["C1", "C2", "C3", "C4"]
    free_cols = ["C1", "C2", "C3"]

    # Initial warm-up observations (from canonical discrete pool for reproducible fairness)
    observed_records: list[dict[str, Any]] = []

    for idx in init_indices:
        cand_row = discrete_pool.iloc[idx]
        pid = str(cand_row["policy_id"])
        c1, c2, c3, c4 = float(cand_row["C1"]), float(cand_row["C2"]), float(cand_row["C3"]), float(cand_row["C4"])

        sim_seed = generate_attia_simulator_seed(benchmark_seed=optimizer_seed, policy_id=pid)
        sim_life = simulate_attia_policy(c1, c2, c3, mode="hi", variance=True, seed=sim_seed)
        true_life = simulate_attia_policy(c1, c2, c3, mode="hi", variance=False, seed=0)

        observed_records.append(
            {
                "policy_id": pid,
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "C4": c4,
                "simulator_seed": sim_seed,
                "simulated_lifetime": float(sim_life),
                "reference_true_lifetime": float(true_life),
                "is_novel": False,
                "min_dist_to_discrete": 0.0,
            }
        )

    history: list[dict[str, Any]] = []
    current_best_sim = max(r["simulated_lifetime"] for r in observed_records)
    current_best_true = max(r["reference_true_lifetime"] for r in observed_records)

    # Initial state (Step 0 summary)
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
            "reference_true_lifetime": None,
            "best_reference_true": current_best_true,
            "simple_regret": max(0.0, DISCRETE_GRID_OPTIMUM_TRUE - current_best_true),
            "improvement_over_grid": max(0.0, current_best_true - DISCRETE_GRID_OPTIMUM_TRUE),
            "is_novel": False,
            "min_dist_to_discrete": 0.0,
        }
    )

    # Closed-loop Continuous BO iterations
    for step in range(1, total_queries + 1):
        # 1. Sample large feasible continuous candidate batch
        cand_batch = search_space.sample_feasible(
            n=n_candidates_per_step,
            seed=optimizer_seed * 1000 + step * 100 + 7,
        )

        # 2. Fit GP surrogate on observed data
        X_train = np.array([[r[c] for c in feature_cols] for r in observed_records], dtype=float)
        y_train = np.array([r["simulated_lifetime"] for r in observed_records], dtype=float)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

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

        # 3. Predict across feasible candidate batch
        X_cand = cand_batch[feature_cols].to_numpy(dtype=float)
        X_cand_scaled = scaler.transform(X_cand)

        if strategy == "random":
            rng_step = np.random.default_rng(optimizer_seed + step * 1000)
            selected_idx = int(rng_step.integers(0, len(cand_batch)))
            best_cand_dict = cand_batch.iloc[selected_idx].to_dict()

        elif strategy in {"greedy", "gp_ucb"}:
            if strategy == "greedy":
                scores = gp.predict(X_cand_scaled)
            else:  # "gp_ucb"
                pred_mean, pred_std = gp.predict(X_cand_scaled, return_std=True)
                scores = pred_mean + beta * pred_std

            top_idx = int(np.argmax(scores))
            init_c1 = float(cand_batch.iloc[top_idx]["C1"])
            init_c2 = float(cand_batch.iloc[top_idx]["C2"])
            init_c3 = float(cand_batch.iloc[top_idx]["C3"])

            best_cand_dict = cand_batch.iloc[top_idx].to_dict()

            # Optional local continuous refinement via Scipy optimize
            if refine_continuous:
                def _obj(x: np.ndarray) -> float:
                    c1_val, c2_val, c3_val = float(x[0]), float(x[1]), float(x[2])
                    c4_val = float(compute_expected_c4(c1_val, c2_val, c3_val))
                    cand_test = {"C1": c1_val, "C2": c2_val, "C3": c3_val, "C4": c4_val}
                    if not search_space.is_feasible(cand_test):
                        return 1e6  # Constraint penalty

                    x_feat = np.array([[c1_val, c2_val, c3_val, c4_val]])
                    x_sc = scaler.transform(x_feat)
                    if strategy == "greedy":
                        return -float(gp.predict(x_sc)[0])
                    else:
                        m, s = gp.predict(x_sc, return_std=True)
                        return -float(m[0] + beta * s[0])

                bounds = [(3.6, 8.0), (3.6, 7.0), (3.6, 5.6)]
                opt_res = minimize(
                    _obj,
                    x0=np.array([init_c1, init_c2, init_c3]),
                    bounds=bounds,
                    method="L-BFGS-B",
                    options={"maxiter": 25, "ftol": 1e-4},
                )

                if opt_res.success:
                    ref_c1, ref_c2, ref_c3 = float(opt_res.x[0]), float(opt_res.x[1]), float(opt_res.x[2])
                    ref_c4 = float(compute_expected_c4(ref_c1, ref_c2, ref_c3))
                    ref_cand = {"C1": round(ref_c1, 3), "C2": round(ref_c2, 3), "C3": round(ref_c3, 3), "C4": round(ref_c4, 3)}
                    if search_space.is_feasible(ref_cand):
                        best_cand_dict = ref_cand

        else:
            raise ValueError(f"Unknown optimization strategy: {strategy!r}")

        # 4. Check novelty against discrete 224-policy grid
        c1 = float(best_cand_dict["C1"])
        c2 = float(best_cand_dict["C2"])
        c3 = float(best_cand_dict["C3"])
        c4 = float(compute_expected_c4(c1, c2, c3))

        single_cand_df = pd.DataFrame([{"C1": c1, "C2": c2, "C3": c3, "C4": c4}])
        novelty_df = search_space.check_novelty(
            single_cand_df,
            reference_points=discrete_pool,
            feature_cols=feature_cols,
            tol=1e-3,
        )
        is_novel = bool(novelty_df["is_novel"].iloc[0])
        min_dist = float(novelty_df["min_distance"].iloc[0])

        pid = f"ATTIA_CONT_S{optimizer_seed:02d}_ST{step:02d}"

        # 5. Evaluate simulator with fair stochastic seed
        sim_seed = generate_attia_simulator_seed(benchmark_seed=optimizer_seed, policy_id=pid)
        sim_life = simulate_attia_policy(c1, c2, c3, mode="hi", variance=True, seed=sim_seed)
        true_life = simulate_attia_policy(c1, c2, c3, mode="hi", variance=False, seed=0)

        observed_records.append(
            {
                "policy_id": pid,
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "C4": c4,
                "simulator_seed": sim_seed,
                "simulated_lifetime": float(sim_life),
                "reference_true_lifetime": float(true_life),
                "is_novel": is_novel,
                "min_dist_to_discrete": min_dist,
            }
        )

        current_best_sim = max(current_best_sim, float(sim_life))
        current_best_true = max(current_best_true, float(true_life))

        history.append(
            {
                "benchmark_seed": optimizer_seed,
                "strategy": strategy,
                "step": step,
                "policy_id": pid,
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "C4": c4,
                "simulator_seed": sim_seed,
                "simulated_lifetime": float(sim_life),
                "best_observed_lifetime": current_best_sim,
                "reference_true_lifetime": float(true_life),
                "best_reference_true": current_best_true,
                "simple_regret": max(0.0, DISCRETE_GRID_OPTIMUM_TRUE - current_best_true),
                "improvement_over_grid": max(0.0, current_best_true - DISCRETE_GRID_OPTIMUM_TRUE),
                "is_novel": is_novel,
                "min_dist_to_discrete": min_dist,
            }
        )

    return history


def run_attia_continuous_benchmark(
    adapter: AttiaAdapter | None = None,
    budgets: Sequence[int] = (10, 15, 20, 30),
    initial_policies: int = 5,
    n_seeds: int = 30,
    beta: float = 1.0,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Runs the continuous Bayesian Optimization benchmark across multiple budgets and paired seeds."""
    if adapter is None:
        adapter = AttiaAdapter()

    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "outputs" / "attia_continuous"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discrete_pool = adapter.load_candidate_pool()
    search_space: SearchSpace = adapter.continuous_search_space()

    # Save search space summary
    search_space_meta = {
        **search_space.to_dict(),
        "discrete_reference_grid_size": len(discrete_pool),
        "discrete_grid_optimum_true": DISCRETE_GRID_OPTIMUM_TRUE,
        "notes": "Continuous optimization allows off-grid charging rates C1, C2, C3 with automatic C4 derivation.",
    }
    with open(output_dir / "search_space_summary.json", "w", encoding="utf-8") as f:
        json.dump(search_space_meta, f, indent=2)

    strategies = ["random", "greedy", "gp_ucb"]
    max_budget = max(budgets)
    max_queries = max_budget - initial_policies

    full_trajectories_by_seed: dict[int, dict[str, list[dict[str, Any]]]] = {}
    all_proposed_protocols: list[dict[str, Any]] = []
    all_history_records: list[dict[str, Any]] = []

    logger.info(
        "Running Continuous Bayesian Optimization benchmark across %d seeds and budgets %s...",
        n_seeds,
        budgets,
    )

    for seed_idx in range(n_seeds):
        seed_rng = np.random.default_rng(seed_idx * 7919 + 42)
        init_indices = seed_rng.choice(len(discrete_pool), size=initial_policies, replace=False).tolist()

        full_trajectories_by_seed[seed_idx] = {}
        for strat in strategies:
            hist = run_single_attia_continuous_trajectory(
                search_space=search_space,
                discrete_pool=discrete_pool,
                init_indices=init_indices,
                total_queries=max_queries,
                strategy=strat,
                optimizer_seed=seed_idx,
                beta=beta,
            )
            full_trajectories_by_seed[seed_idx][strat] = hist

            for row in hist:
                if row["step"] > 0:
                    all_proposed_protocols.append(
                        {
                            "benchmark_seed": seed_idx,
                            "strategy": strat,
                            "step": row["step"],
                            "policy_id": row["policy_id"],
                            "C1": row["C1"],
                            "C2": row["C2"],
                            "C3": row["C3"],
                            "C4": row["C4"],
                            "simulated_lifetime": row["simulated_lifetime"],
                            "reference_true_lifetime": row["reference_true_lifetime"],
                            "is_novel": row["is_novel"],
                            "min_dist_to_discrete": row["min_dist_to_discrete"],
                        }
                    )

    # Compute metrics for each budget
    budget_sweep_results: list[dict[str, Any]] = []
    overall_best_continuous_protocol: dict[str, Any] = {
        "reference_true_lifetime": -1.0,
        "policy_id": None,
        "C1": None,
        "C2": None,
        "C3": None,
        "C4": None,
        "strategy": None,
        "seed": None,
        "step": None,
        "is_novel": False,
        "min_dist_to_discrete": None,
        "improvement_over_grid": 0.0,
    }

    for budget in budgets:
        n_queries = budget - initial_policies
        strategy_metrics: dict[str, dict[str, Any]] = {}

        strat_bests: dict[str, list[float]] = {s: [] for s in strategies}
        strat_regrets: dict[str, list[float]] = {s: [] for s in strategies}
        strat_aucs: dict[str, list[float]] = {s: [] for s in strategies}
        strat_novel_rates: dict[str, list[float]] = {s: [] for s in strategies}
        strat_improvements: dict[str, list[float]] = {s: [] for s in strategies}

        for seed_idx in range(n_seeds):
            for strat in strategies:
                hist = full_trajectories_by_seed[seed_idx][strat]
                sliced = [h for h in hist if h["step"] <= n_queries]

                final_row = sliced[-1]
                best_val = float(final_row["best_reference_true"])
                regret_val = float(final_row["simple_regret"])
                imp_val = float(final_row["improvement_over_grid"])

                strat_bests[strat].append(best_val)
                strat_regrets[strat].append(regret_val)
                strat_improvements[strat].append(imp_val)

                regret_series = [h["simple_regret"] for h in sliced]
                strat_aucs[strat].append(float(np.mean(regret_series)))

                novel_count = sum(1 for h in sliced if h["step"] > 0 and h["is_novel"])
                query_count = sum(1 for h in sliced if h["step"] > 0)
                novel_rate = (novel_count / query_count) if query_count > 0 else 0.0
                strat_novel_rates[strat].append(novel_rate)

                # Track global best
                for h in sliced:
                    if h["step"] > 0 and float(h["reference_true_lifetime"]) > overall_best_continuous_protocol["reference_true_lifetime"]:
                        overall_best_continuous_protocol = {
                            "reference_true_lifetime": float(h["reference_true_lifetime"]),
                            "policy_id": h["policy_id"],
                            "C1": h["C1"],
                            "C2": h["C2"],
                            "C3": h["C3"],
                            "C4": h["C4"],
                            "strategy": strat,
                            "seed": seed_idx,
                            "step": h["step"],
                            "is_novel": bool(h["is_novel"]),
                            "min_dist_to_discrete": float(h["min_dist_to_discrete"]),
                            "improvement_over_grid": max(0.0, float(h["reference_true_lifetime"]) - DISCRETE_GRID_OPTIMUM_TRUE),
                        }

                if budget == max_budget:
                    for h in sliced:
                        all_history_records.append({**h, "budget": budget})

        for strat in strategies:
            bests = np.array(strat_bests[strat], dtype=float)
            regs = np.array(strat_regrets[strat], dtype=float)
            aucs = np.array(strat_aucs[strat], dtype=float)
            novs = np.array(strat_novel_rates[strat], dtype=float)
            imps = np.array(strat_improvements[strat], dtype=float)

            reg_emp_low, reg_emp_high = np.percentile(regs, [2.5, 97.5])
            reg_ci_low, reg_ci_high = compute_bootstrap_mean_ci(regs, n_bootstraps=2000, seed=42 + budget)
            best_ci_low, best_ci_high = compute_bootstrap_mean_ci(bests, n_bootstraps=2000, seed=1042 + budget)

            strategy_metrics[strat] = {
                "mean_best_seen": float(np.mean(bests)),
                "std_best_seen": float(np.std(bests, ddof=1)) if len(bests) > 1 else 0.0,
                "median_best_seen": float(np.median(bests)),
                "mean_best_seen_95_ci": [float(best_ci_low), float(best_ci_high)],
                "mean_simple_regret": float(np.mean(regs)),
                "std_simple_regret": float(np.std(regs, ddof=1)) if len(regs) > 1 else 0.0,
                "median_simple_regret": float(np.median(regs)),
                "mean_simple_regret_95_ci": [float(reg_ci_low), float(reg_ci_high)],
                "simple_regret_95_empirical_interval": [float(reg_emp_low), float(reg_emp_high)],
                "mean_regret_auc": float(np.mean(aucs)),
                "mean_novel_proposal_rate": float(np.mean(novs)),
                "mean_improvement_over_grid": float(np.mean(imps)),
                "max_improvement_over_grid": float(np.max(imps)),
            }

        # Paired Wilcoxon test (UCB vs Greedy)
        diff_regrets = np.array(strat_regrets["gp_ucb"]) - np.array(strat_regrets["greedy"])
        diff_aucs = np.array(strat_aucs["gp_ucb"]) - np.array(strat_aucs["greedy"])

        def _safe_wilcoxon(d: np.ndarray) -> float:
            if np.all(d == 0):
                return 1.0
            try:
                stat, p_val = wilcoxon(d, alternative="two-sided")
                return float(p_val)
            except Exception:
                return 1.0

        budget_sweep_results.append(
            {
                "budget": budget,
                "initial_policies": initial_policies,
                "queries": n_queries,
                "strategies": strategy_metrics,
                "paired_comparison_ucb_vs_greedy": {
                    "n_paired_seeds": n_seeds,
                    "regret_diff_ucb_minus_greedy": {
                        "mean": float(np.mean(diff_regrets)),
                        "median": float(np.median(diff_regrets)),
                        "std": float(np.std(diff_regrets, ddof=1)) if len(diff_regrets) > 1 else 0.0,
                        "p_value": _safe_wilcoxon(diff_regrets),
                    },
                    "auc_diff_ucb_minus_greedy": {
                        "mean": float(np.mean(diff_aucs)),
                        "median": float(np.median(diff_aucs)),
                        "std": float(np.std(diff_aucs, ddof=1)) if len(diff_aucs) > 1 else 0.0,
                        "p_value": _safe_wilcoxon(diff_aucs),
                    },
                },
            }
        )

    # Save output artifacts
    pd.DataFrame(all_history_records).to_csv(output_dir / "optimization_history.csv", index=False)
    pd.DataFrame(all_proposed_protocols).to_csv(output_dir / "proposed_protocols.csv", index=False)

    final_budget_entry = budget_sweep_results[-1]
    benchmark_summary = {
        "benchmark": "Attia et al. 2020 Continuous Fast-Charging Bayesian Optimization Benchmark",
        "benchmark_nature": "simulator != experimental dataset",
        "discrete_grid_optimum": {
            "policy_id": "ATTIA_P113",
            "C1": 6.0,
            "C2": 4.8,
            "C3": 4.0,
            "C4": 4.8,
            "reference_true_lifetime": DISCRETE_GRID_OPTIMUM_TRUE,
        },
        "best_continuous_protocol_discovered": overall_best_continuous_protocol,
        "continuous_improvement_found": bool(overall_best_continuous_protocol["improvement_over_grid"] > 0),
        "continuous_improvement_cycles": overall_best_continuous_protocol["improvement_over_grid"],
        "optimization_parameters": {
            "initial_policies": initial_policies,
            "total_budget": max_budget,
            "total_queries": max_queries,
            "n_seeds": n_seeds,
            "beta": beta,
            "candidates_evaluated_per_step": 5000,
            "bootstrap_replicates": 2000,
        },
        "strategy_comparison": final_budget_entry["strategies"],
        "paired_comparison_ucb_vs_greedy": final_budget_entry["paired_comparison_ucb_vs_greedy"],
        "budget_sweep": budget_sweep_results,
    }

    with open(output_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    return benchmark_summary
