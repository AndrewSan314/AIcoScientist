from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from src.datasets.auirh import (
    AUIRH_CANDIDATE_ID_COLUMN,
    AUIRH_FEATURE_COLUMNS,
    AUIRH_LIBRARIES,
    AuIrRhAdapter,
    AuIrRhExperimentOracle,
)
from src.legacy.native_optimizer.acquisition import (
    expected_improvement_acquisition,
    greedy_acquisition,
    predict_latent_gp,
    safe_cholesky,
    ucb_acquisition,
)

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


def fit_auirh_gp(
    observed_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seed: int,
) -> tuple[GaussianProcessRegressor, StandardScaler]:
    """Fits Gaussian Process surrogate with Matern 5/2 kernel on observed Au-Ir-Rh samples."""
    X_obs = observed_df[feature_cols].to_numpy(dtype=float)
    y_obs = observed_df[target_col].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_obs)

    kernel = ConstantKernel(1.0, (1e-2, 1e4)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=seed,
    )
    gp.fit(X_scaled, y_obs)
    return gp, scaler


def run_single_reproduction_trajectory(
    candidate_pool: pd.DataFrame,
    oracle: AuIrRhExperimentOracle,
    target_name: str,
    strategy: str,
    init_sample_index: int,
    total_budget: int = 50,
    seed: int = 42,
    beta: float = 2.0,
) -> list[dict[str, Any]]:
    """Executes a single closed-loop optimization trajectory on the Au-Ir-Rh candidate pool.

    Strategies supported:
    - 'random': uniform random finite-pool sampling
    - 'greedy': pure exploitation on latent GP mean
    - 'gp_ucb': uncertainty-aware upper confidence bound (beta=2.0)
    - 'thompson_sampling': joint GP posterior covariance draws via Cholesky decomposition
    - 'expected_improvement' / 'ei': analytic expected improvement (xi=0.01)
    """
    oracle.reset()
    global_optimum = oracle.global_best_value
    feature_cols = list(AUIRH_FEATURE_COLUMNS)

    # 1. Warm-up: 1 initial measured material
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

    rng = np.random.default_rng(seed)
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
        }
    )

    for step in range(2, total_budget + 1):
        unseen_pool = candidate_pool[~candidate_pool[AUIRH_CANDIDATE_ID_COLUMN].isin(seen_cids)].copy().reset_index(drop=True)
        if unseen_pool.empty:
            break

        step_seed = seed * 1000 + step * 10 + 7
        selected_row: pd.Series
        acq_score_val = 0.0

        if strategy == "random":
            chosen_idx = rng.integers(0, len(unseen_pool))
            selected_row = unseen_pool.iloc[chosen_idx]
            acq_score_val = float(rng.uniform(0, 1))

        else:
            obs_df = pd.DataFrame(observed_records)
            gp, scaler = fit_auirh_gp(obs_df, feature_cols, target_name, seed=step_seed)

            X_unseen = unseen_pool[feature_cols].to_numpy(dtype=float)
            X_unseen_scaled = scaler.transform(X_unseen)

            pred_mean, pred_std = predict_latent_gp(gp, X_unseen_scaled, return_std=True)
            pred_mean = np.asarray(pred_mean, dtype=float).flatten()
            pred_std = np.asarray(pred_std, dtype=float).flatten()

            if strategy == "greedy":
                scores = greedy_acquisition(pred_mean, objective="maximize")
            elif strategy in {"gp_ucb", "gp_ucb_1", "gp_ucb_2"}:
                b = 1.0 if strategy == "gp_ucb_1" else beta
                scores = ucb_acquisition(pred_mean, pred_std, beta=b, objective="maximize")
            elif strategy == "thompson_sampling":
                mu_latent, cov_latent = predict_latent_gp(gp, X_unseen_scaled, return_cov=True)
                mu_latent = np.asarray(mu_latent, dtype=float).flatten()
                cov_latent = np.asarray(cov_latent, dtype=float)
                L = safe_cholesky(cov_latent, base_jitter=1e-8)
                step_rng = np.random.default_rng(step_seed)
                z = step_rng.standard_normal(size=len(mu_latent))
                scores = mu_latent + (L @ z)
            elif strategy in {"expected_improvement", "ei"}:
                scores = expected_improvement_acquisition(
                    mean=pred_mean,
                    std=pred_std,
                    best_observed=current_best,
                    xi=0.0,
                    objective="maximize",
                )
            else:
                raise ValueError(f"Unknown strategy: '{strategy}'")

            best_score_idx = int(np.argmax(scores))
            selected_row = unseen_pool.iloc[best_score_idx]
            acq_score_val = float(scores[best_score_idx])

        cid = str(selected_row[AUIRH_CANDIDATE_ID_COLUMN])
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

        current_best = max(current_best, obs_val)
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
                "acquisition_score": acq_score_val,
            }
        )

    return trajectory


def _reproduction_worker(args: tuple[int, int, int, str, str, str | None, int]) -> tuple[int, list[dict[str, Any]]]:
    s_idx, run_seed, init_idx, strat, target_name, library, total_budget = args
    worker_adapter = AuIrRhAdapter(target=target_name, library=library)
    worker_pool = worker_adapter.get_candidate_pool()
    worker_oracle = worker_adapter.create_oracle()
    worker_traj = run_single_reproduction_trajectory(
        candidate_pool=worker_pool,
        oracle=worker_oracle,
        target_name=target_name,
        strategy=strat,
        init_sample_index=init_idx,
        total_budget=total_budget,
        seed=run_seed,
    )
    mode_label = library if library else "pooled"
    for r in worker_traj:
        r["benchmark_mode"] = mode_label
    return s_idx, worker_traj


def run_auirh_reproduction_benchmark(
    target_name: str = "k0",
    library: str | None = None,
    strategies: Sequence[str] = ("random", "greedy", "gp_ucb", "thompson_sampling", "expected_improvement"),
    n_seeds: int = 30,
    total_budget: int = 50,
    base_seed: int = 42,
    output_dir: Path | str = "outputs/auirh/reproduction",
) -> dict[str, Any]:
    """Executes multi-seed reproduction benchmark across strategies."""
    adapter = AuIrRhAdapter(target=target_name, library=library)
    candidate_pool = adapter.get_candidate_pool()
    pool_size = len(candidate_pool)
    oracle = adapter.create_oracle()
    global_optimum = oracle.global_best_value

    mode_label = library if library else "pooled"
    logger.info("Starting Au-Ir-Rh reproduction benchmark: target=%s, mode=%s, seeds=%d, pool_size=%d", target_name, mode_label, n_seeds, pool_size)

    rng_init = np.random.default_rng(base_seed)
    init_indices = [int(rng_init.integers(0, pool_size)) for _ in range(n_seeds)]

    all_step_records: list[dict[str, Any]] = []
    strategy_results: dict[str, Any] = {}

    from concurrent.futures import ProcessPoolExecutor
    max_w = min(8, n_seeds)

    for strat in strategies:
        logger.info("Running reproduction strategy '%s' across %d seeds...", strat, n_seeds)
        step_bests: list[list[float]] = [[] for _ in range(n_seeds)]
        step_regrets: list[list[float]] = [[] for _ in range(n_seeds)]
        traj_results: list[list[dict[str, Any]]] = [[] for _ in range(n_seeds)]

        args_list = [
            (i, base_seed + i, init_indices[i], strat, target_name, library, total_budget)
            for i in range(n_seeds)
        ]
        with ProcessPoolExecutor(max_workers=max_w) as executor:
            for s_idx, traj in executor.map(_reproduction_worker, args_list):
                traj_results[s_idx] = traj

        final_bests: list[float] = []
        final_regrets: list[float] = []
        aucs: list[float] = []
        queries_to_opt: list[int] = []
        queries_to_10pct: list[int] = []
        queries_to_5pct: list[int] = []
        queries_to_1pct: list[int] = []
        queries_to_01pct: list[int] = []

        for s_idx in range(n_seeds):
            traj = traj_results[s_idx]
            all_step_records.extend(traj)

            bests = [step["best_observed"] for step in traj]
            regrets = [step["regret"] for step in traj]
            step_bests[s_idx] = bests
            step_regrets[s_idx] = regrets

            final_best = bests[-1]
            final_regret = regrets[-1]
            final_bests.append(final_best)
            final_regrets.append(final_regret)
            aucs.append(float(np.mean(bests)))

            # Step thresholds
            q_opt = total_budget + 1
            q_10 = total_budget + 1
            q_5 = total_budget + 1
            q_1 = total_budget + 1
            q_01 = total_budget + 1

            for s_idx, row in enumerate(traj):
                rel_r = row["relative_regret"]
                if rel_r <= 0.10 and q_10 > total_budget:
                    q_10 = s_idx + 1
                if rel_r <= 0.05 and q_5 > total_budget:
                    q_5 = s_idx + 1
                if rel_r <= 0.01 and q_1 > total_budget:
                    q_1 = s_idx + 1
                if rel_r <= 0.001 and q_01 > total_budget:
                    q_01 = s_idx + 1
                if row["regret"] <= 1e-9 and q_opt > total_budget:
                    q_opt = s_idx + 1

            queries_to_opt.append(q_opt)
            queries_to_10pct.append(q_10)
            queries_to_5pct.append(q_5)
            queries_to_1pct.append(q_1)
            queries_to_01pct.append(q_01)

        bests_arr = np.array(step_bests)
        regrets_arr = np.array(step_regrets)

        mean_final_regret = float(np.mean(final_regrets))
        ci_lower, ci_upper = compute_bootstrap_ci_95(final_regrets, seed=base_seed)
        hit_rate = float(np.mean([1.0 if q <= total_budget else 0.0 for q in queries_to_opt]))
        hit_10_rate = float(np.mean([1.0 if q <= total_budget else 0.0 for q in queries_to_10pct]))
        hit_5_rate = float(np.mean([1.0 if q <= total_budget else 0.0 for q in queries_to_5pct]))
        hit_1_rate = float(np.mean([1.0 if q <= total_budget else 0.0 for q in queries_to_1pct]))
        hit_01_rate = float(np.mean([1.0 if q <= total_budget else 0.0 for q in queries_to_01pct]))

        med_q_opt = float(np.median(queries_to_opt))
        med_q_10 = float(np.median(queries_to_10pct))
        med_q_5 = float(np.median(queries_to_5pct))
        med_q_1 = float(np.median(queries_to_1pct))
        med_q_01 = float(np.median(queries_to_01pct))

        strategy_results[strat] = {
            "mean_best_so_far": [float(x) for x in np.mean(bests_arr, axis=0)],
            "std_best_so_far": [float(x) for x in np.std(bests_arr, axis=0)],
            "mean_regret": [float(x) for x in np.mean(regrets_arr, axis=0)],
            "std_regret": [float(x) for x in np.std(regrets_arr, axis=0)],
            "final_regret_mean": mean_final_regret,
            "final_regret_std": float(np.std(final_regrets)),
            "final_regret_ci95": [ci_lower, ci_upper],
            "best_so_far_auc_mean": float(np.mean(aucs)),
            "best_so_far_auc_std": float(np.std(aucs)),
            "optimum_hit_rate": hit_rate,
            "median_queries_to_optimum": med_q_opt,
            "fraction_pool_to_optimum": float(med_q_opt / pool_size),
            "success_rate_10pct": hit_10_rate,
            "median_queries_10pct": med_q_10,
            "fraction_pool_10pct": float(med_q_10 / pool_size),
            "success_rate_5pct": hit_5_rate,
            "median_queries_5pct": med_q_5,
            "fraction_pool_5pct": float(med_q_5 / pool_size),
            "success_rate_1pct": hit_1_rate,
            "median_queries_1pct": med_q_1,
            "fraction_pool_1pct": float(med_q_1 / pool_size),
            "success_rate_01pct": hit_01_rate,
            "median_queries_01pct": med_q_01,
            "fraction_pool_01pct": float(med_q_01 / pool_size),
        }

    out_base = Path(output_dir) / mode_label / target_name
    out_base.mkdir(parents=True, exist_ok=True)

    df_steps = pd.DataFrame(all_step_records)
    csv_path = out_base / "per_step.csv"
    df_steps.to_csv(csv_path, index=False)

    summary = {
        "dataset": "Au-Ir-Rh Autonomous SECCM Benchmark",
        "benchmark_type": "reproduction_baseline",
        "mode": mode_label,
        "target": target_name,
        "pool_size": pool_size,
        "global_optimum": global_optimum,
        "n_seeds": n_seeds,
        "total_budget": total_budget,
        "base_seed": base_seed,
        "strategies": strategy_results,
    }
    json_path = out_base / "summary.json"
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Saved reproduction summary to %s and steps to %s", json_path, csv_path)
    return summary
