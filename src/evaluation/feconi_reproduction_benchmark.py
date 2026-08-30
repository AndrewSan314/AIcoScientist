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

from src.datasets.feconi import (
    FECONI_CANDIDATE_ID_COLUMN,
    FECONI_FEATURE_COLUMNS,
    FeCoNiAdapter,
    FeCoNiExperimentOracle,
)
from src.optimization.acquisition import (
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


def fit_feconi_gp(
    observed_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seed: int,
) -> tuple[GaussianProcessRegressor, StandardScaler]:
    """Fits Gaussian Process surrogate with Matern 5/2 kernel on observed material samples."""
    X_obs = observed_df[feature_cols].to_numpy(dtype=float)
    y_obs = observed_df[target_col].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_obs)

    kernel = ConstantKernel(1.0, (1e-2, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5) + WhiteKernel(
        noise_level=1e-3, noise_level_bounds=(1e-5, 1e1)
    )
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
    oracle: FeCoNiExperimentOracle,
    target_name: str,
    strategy: str,
    init_sample_index: int,
    total_budget: int = 100,
    seed: int = 42,
    beta: float = 2.0,
) -> list[dict[str, Any]]:
    """Executes a single closed-loop optimization trajectory on the Fe-Co-Ni candidate pool.

    Protocol (Wang et al. 2022):
    1. Start from 1 randomly chosen measured material.
    2. Fit GP surrogate on observed samples.
    3. Predict on remaining unmeasured candidates in the 921 pool.
    4. Compute acquisition function score.
    5. Select argmax candidate and query oracle.
    6. Repeat for budget.
    """
    oracle.reset()
    global_optimum = oracle.global_best_value
    feature_cols = list(FECONI_FEATURE_COLUMNS)

    # 1. Warm-up: 1 initial measured material
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

    for step in range(2, total_budget + 1):
        unseen_pool = candidate_pool[~candidate_pool[FECONI_CANDIDATE_ID_COLUMN].isin(seen_cids)].copy().reset_index(drop=True)
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
            gp, scaler = fit_feconi_gp(obs_df, feature_cols, target_name, seed=step_seed)

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
                    xi=0.01,
                    objective="maximize",
                )
            else:
                raise ValueError(f"Unknown strategy: '{strategy}'")

            best_score_idx = int(np.argmax(scores))
            selected_row = unseen_pool.iloc[best_score_idx]
            acq_score_val = float(scores[best_score_idx])

        cid = str(selected_row[FECONI_CANDIDATE_ID_COLUMN])
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
                "acquisition_score": acq_score_val,
            }
        )

    return trajectory


from joblib import Parallel, delayed


def _run_seed_worker(
    seed: int,
    candidate_pool: pd.DataFrame,
    oracle: FeCoNiExperimentOracle,
    target_name: str,
    strategies: Sequence[str],
    total_budget: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    init_idx = int(rng.integers(0, len(candidate_pool)))
    seed_rows: list[dict[str, Any]] = []
    for strat in strategies:
        traj = run_single_reproduction_trajectory(
            candidate_pool=candidate_pool,
            oracle=oracle,
            target_name=target_name,
            strategy=strat,
            init_sample_index=init_idx,
            total_budget=total_budget,
            seed=seed,
        )
        seed_rows.extend(traj)
    return seed_rows


def run_feconi_reproduction_benchmark(
    target_name: str = "Kerr",
    strategies: Sequence[str] = ("random", "greedy", "gp_ucb", "thompson_sampling", "expected_improvement"),
    seeds: Sequence[int] = range(100),
    total_budget: int = 100,
    output_dir: Path | str | None = None,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Runs the multi-seed paper reproduction benchmark for Fe-Co-Ni."""
    adapter = FeCoNiAdapter(target=target_name)
    pool = adapter.get_candidate_pool()
    oracle = adapter.create_oracle(allow_duplicate_queries=False)
    global_best = oracle.global_best_value

    worker_results = Parallel(n_jobs=n_jobs)(
        delayed(_run_seed_worker)(
            seed=seed,
            candidate_pool=pool,
            oracle=oracle,
            target_name=target_name,
            strategies=strategies,
            total_budget=total_budget,
        )
        for seed in seeds
    )

    all_rows: list[dict[str, Any]] = []
    for r in worker_results:
        all_rows.extend(r)

    results_df = pd.DataFrame(all_rows)

    summary_metrics: dict[str, Any] = {
        "target": target_name,
        "global_best": float(global_best),
        "total_seeds": len(seeds),
        "total_budget": total_budget,
        "strategies": {},
    }

    for strat in strategies:
        strat_df = results_df[results_df["strategy"] == strat]
        seed_aucs: list[float] = []
        steps_to_10: list[int] = []
        steps_to_5: list[int] = []
        steps_to_1: list[int] = []
        steps_to_01: list[int] = []
        steps_to_exact: list[int] = []
        final_regrets: list[float] = []
        final_best: list[float] = []

        for s in seeds:
            s_traj = strat_df[strat_df["run_seed"] == s].sort_values("iteration")
            if s_traj.empty:
                continue
            bests = s_traj["best_observed"].to_numpy()
            # Trapezoidal area under curve
            if total_budget > 1 and len(bests) > 1:
                trapz_area = float(np.sum((bests[1:] + bests[:-1]) * 0.5))
                auc = float(trapz_area / (total_budget - 1))
            else:
                auc = float(bests[0])
            seed_aucs.append(auc)
            final_regrets.append(float(s_traj["regret"].iloc[-1]))
            final_best.append(float(bests[-1]))

            devs = s_traj["percent_deviation"].to_numpy()
            hit_10 = np.where(devs <= 10.0)[0]
            steps_to_10.append(int(hit_10[0] + 1) if len(hit_10) > 0 else total_budget + 1)

            hit_5 = np.where(devs <= 5.0)[0]
            steps_to_5.append(int(hit_5[0] + 1) if len(hit_5) > 0 else total_budget + 1)

            hit_1 = np.where(devs <= 1.0)[0]
            steps_to_1.append(int(hit_1[0] + 1) if len(hit_1) > 0 else total_budget + 1)

            hit_01 = np.where(devs <= 0.1)[0]
            steps_to_01.append(int(hit_01[0] + 1) if len(hit_01) > 0 else total_budget + 1)

            hit_exact = np.where(bests >= global_best - 1e-6)[0]
            steps_to_exact.append(int(hit_exact[0] + 1) if len(hit_exact) > 0 else total_budget + 1)

        auc_mean = float(np.mean(seed_aucs))
        auc_ci = compute_bootstrap_ci_95(seed_aucs)
        regret_mean = float(np.mean(final_regrets))
        regret_ci = compute_bootstrap_ci_95(final_regrets)

        summary_metrics["strategies"][strat] = {
            "mean_auc": auc_mean,
            "auc_95_ci": list(auc_ci),
            "mean_final_regret": regret_mean,
            "final_regret_95_ci": list(regret_ci),
            "mean_final_best": float(np.mean(final_best)),
            "success_rate_10pct": float(np.mean([1 if s <= total_budget else 0 for s in steps_to_10])),
            "success_rate_5pct": float(np.mean([1 if s <= total_budget else 0 for s in steps_to_5])),
            "success_rate_1pct": float(np.mean([1 if s <= total_budget else 0 for s in steps_to_1])),
            "success_rate_0.1pct": float(np.mean([1 if s <= total_budget else 0 for s in steps_to_01])),
            "exact_optimum_hit_rate": float(np.mean([1 if s <= total_budget else 0 for s in steps_to_exact])),
            "mean_steps_to_10pct": float(np.mean(steps_to_10)),
            "median_steps_to_10pct": float(np.median(steps_to_10)),
            "mean_steps_to_5pct": float(np.mean(steps_to_5)),
            "median_steps_to_5pct": float(np.median(steps_to_5)),
            "mean_steps_to_1pct": float(np.mean(steps_to_1)),
            "median_steps_to_1pct": float(np.median(steps_to_1)),
            "median_steps_to_exact_optimum": float(np.median(steps_to_exact)),
        }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path / "per_step.csv", index=False)
        with open(out_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)

    return results_df, summary_metrics
