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
    compute_true_mc_nei,
    expected_improvement_acquisition,
    greedy_acquisition,
    predict_latent_gp,
    ucb_acquisition,
)
from src.optimization.search_space import ContinuousVariable, SearchSpace
from src.optimization.trust_region import TrustRegionState, TuRBOTrustRegion

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


def fit_gp_surrogate(
    observed_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seed: int,
) -> tuple[GaussianProcessRegressor, StandardScaler]:
    """Fits Gaussian Process surrogate model matching ClosedLoopOptimizer._fit_surrogate semantics."""
    X_obs = observed_df[feature_cols].to_numpy(dtype=float)
    y_obs = observed_df[target_col].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_obs)

    kernel = ConstantKernel(1.0, (1e-2, 1e4)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
        noise_level=1.0, noise_level_bounds=(1e-5, 1e2)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=seed,
    )
    gp.fit(X_scaled, y_obs)
    return gp, scaler


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
    strategy: str,  # "random", "greedy", "gp_ucb", "expected_improvement", "noisy_expected_improvement", "turbo_nei"
    init_sample_index: int,
    total_budget: int = 50,
    seed: int = 42,
    beta: float = 2.0,
) -> list[dict[str, Any]]:
    """Runs a single closed-loop optimization trajectory comparing AIcoScientist strategies."""
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

    # Setup trust region if using turbo (using default frozen ClosedLoopOptimizer parameters)
    turbo: TuRBOTrustRegion | None = None
    search_space = get_feconi_search_space()
    if "turbo" in strategy:
        turbo = TuRBOTrustRegion(search_space=search_space)
        turbo.initialize(
            center_candidate={"Co": float(init_cand["Co"]), "Fe": float(init_cand["Fe"])},
            initial_best_value=float(init_res[target_name]),
        )

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
        is_escape = False

        if strategy == "random":
            chosen_idx = rng.integers(0, len(unseen_pool))
            selected_row = unseen_pool.iloc[chosen_idx]
            acq_score_val = float(rng.uniform(0, 1))

        else:
            obs_df = pd.DataFrame(observed_records)
            gp, scaler = fit_gp_surrogate(obs_df, feature_cols, target_name, seed=step_seed)

            # Determine candidate subset via frozen TuRBO trust region
            if turbo is not None:
                is_escape = turbo.should_global_escape(step)
                if not is_escape:
                    box = turbo.get_bounding_box()
                    co_min, co_max = box["Co"]
                    fe_min, fe_max = box["Fe"]
                    in_tr = (
                        (unseen_pool["Co"] >= co_min)
                        & (unseen_pool["Co"] <= co_max)
                        & (unseen_pool["Fe"] >= fe_min)
                        & (unseen_pool["Fe"] <= fe_max)
                    )
                    eval_pool = unseen_pool[in_tr].reset_index(drop=True)
                    if eval_pool.empty:
                        # Fallback to full pool if TR box is exhausted of unmeasured points
                        eval_pool = unseen_pool
                else:
                    eval_pool = unseen_pool
            else:
                eval_pool = unseen_pool

            X_eval = eval_pool[feature_cols].to_numpy(dtype=float)
            X_eval_scaled = scaler.transform(X_eval)

            if strategy == "greedy":
                pred_mean = predict_latent_gp(gp, X_eval_scaled, return_std=False)
                pred_mean = np.asarray(pred_mean, dtype=float).flatten()
                scores = greedy_acquisition(pred_mean, objective="maximize")
            elif strategy in {"gp_ucb", "gp_ucb_1", "gp_ucb_2"}:
                pred_mean, pred_std = predict_latent_gp(gp, X_eval_scaled, return_std=True)
                pred_mean = np.asarray(pred_mean, dtype=float).flatten()
                pred_std = np.asarray(pred_std, dtype=float).flatten()
                b = 1.0 if strategy == "gp_ucb_1" else beta
                scores = ucb_acquisition(pred_mean, pred_std, beta=b, objective="maximize")
            elif strategy in {"expected_improvement", "ei"}:
                pred_mean, pred_std = predict_latent_gp(gp, X_eval_scaled, return_std=True)
                pred_mean = np.asarray(pred_mean, dtype=float).flatten()
                pred_std = np.asarray(pred_std, dtype=float).flatten()
                scores = expected_improvement_acquisition(
                    mean=pred_mean,
                    std=pred_std,
                    best_observed=current_best,
                    xi=0.01,
                    objective="maximize",
                )
            elif strategy in {"noisy_expected_improvement", "nei", "turbo_nei", "turbo_ei"}:
                X_obs = obs_df[feature_cols].to_numpy(dtype=float)
                X_obs_scaled = scaler.transform(X_obs)

                scores = compute_true_mc_nei(
                    gp=gp,
                    X_observed_scaled=X_obs_scaled,
                    X_candidates_scaled=X_eval_scaled,
                    n_fantasies=256,
                    xi=0.01,
                    objective="maximize",
                    seed=step_seed,
                )
            else:
                raise ValueError(f"Unknown strategy: '{strategy}'")

            best_idx = int(np.argmax(scores))
            selected_row = eval_pool.iloc[best_idx]
            acq_score_val = float(scores[best_idx])

        # Query Oracle
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

        # Update TuRBO state using frozen ClosedLoopOptimizer lifecycle on D_{t+1}
        if turbo is not None and turbo.state is not None:
            # 1. Refit GP surrogate strictly on D_{t+1}
            obs_df_next = pd.DataFrame(observed_records)
            gp_next, scaler_next = fit_gp_surrogate(obs_df_next, feature_cols, target_name, seed=step_seed + 1)

            # 2. Compute latent joint posterior and covariance over all observed points in D_{t+1}
            X_obs_all = obs_df_next[feature_cols].to_numpy(dtype=float)
            X_obs_all_sc = scaler_next.transform(X_obs_all)
            p_obs_m, p_obs_cov = predict_latent_gp(gp_next, X_obs_all_sc, return_cov=True)
            p_obs_m = np.asarray(p_obs_m, dtype=float)
            p_obs_cov = np.asarray(p_obs_cov, dtype=float)

            # 3. Identify newly observed candidate (last row) and previous incumbent (excluding last row)
            cand_idx = len(p_obs_m) - 1
            p_cand_m = float(p_obs_m[cand_idx])
            p_cand_v = float(p_obs_cov[cand_idx, cand_idx])
            p_cand_s = float(np.sqrt(max(p_cand_v, 1e-12)))

            if len(p_obs_m) > 1:
                prev_obs_m = p_obs_m[:-1]
                inc_idx = int(np.argmax(prev_obs_m))
                p_inc_m = float(prev_obs_m[inc_idx])
                p_inc_v = float(p_obs_cov[inc_idx, inc_idx])
                p_cand_inc_cov = float(p_obs_cov[cand_idx, inc_idx])
            else:
                p_inc_m = p_cand_m
                p_inc_v = p_cand_v
                p_cand_inc_cov = p_cand_v
            p_inc_s = float(np.sqrt(max(p_inc_v, 1e-12)))

            # 4. Fallback center from unmeasured pool if TuRBO restart triggers
            fallback_center: dict[str, Any] | None = None
            fallback_cid: str | None = None
            can_restart = (
                (turbo.state.failure_counter + 1 >= turbo.state.failure_tolerance and (turbo.state.length / 2.0) < turbo.state.min_length)
                or (turbo.state.length < turbo.state.min_length)
            )
            if can_restart:
                unseen_rem = candidate_pool[~candidate_pool[FECONI_CANDIDATE_ID_COLUMN].isin(seen_cids)].copy().reset_index(drop=True)
                if not unseen_rem.empty:
                    X_rem = unseen_rem[feature_cols].to_numpy(dtype=float)
                    X_rem_sc = scaler_next.transform(X_rem)
                    scores_restart = compute_true_mc_nei(
                        gp=gp_next,
                        X_observed_scaled=X_obs_all_sc,
                        X_candidates_scaled=X_rem_sc,
                        n_fantasies=256,
                        xi=0.01,
                        objective="maximize",
                        seed=step_seed + 2,
                    )
                    best_restart_idx = int(np.argmax(scores_restart))
                    best_restart_row = unseen_rem.iloc[best_restart_idx]
                    fallback_center = {"Co": float(best_restart_row["Co"]), "Fe": float(best_restart_row["Fe"])}
                    fallback_cid = str(best_restart_row[FECONI_CANDIDATE_ID_COLUMN])

            # 5. Advance TuRBO state using full covariance-aware posterior evidence
            turbo.update(
                observed_candidate={"Co": float(selected_row["Co"]), "Fe": float(selected_row["Fe"])},
                observed_value=obs_val,
                posterior_candidate_mean=p_cand_m,
                posterior_incumbent_mean=p_inc_m,
                posterior_candidate_variance=p_cand_v,
                posterior_incumbent_variance=p_inc_v,
                posterior_candidate_incumbent_covariance=p_cand_inc_cov,
                posterior_candidate_std=p_cand_s,
                posterior_incumbent_std=p_inc_s,
                objective="maximize",
                fallback_center=fallback_center,
                fallback_candidate_id=fallback_cid,
                global_escape=is_escape,
            )

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


def _run_aicoscientist_seed_worker(
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
        traj = run_single_aicoscientist_trajectory(
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


def run_feconi_aicoscientist_benchmark(
    target_name: str = "Coer",
    strategies: Sequence[str] = ("random", "greedy", "gp_ucb", "expected_improvement", "noisy_expected_improvement", "turbo_nei"),
    seeds: Sequence[int] = range(30),
    total_budget: int = 50,
    output_dir: Path | str | None = None,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Executes the AIcoScientist optimizer benchmark suite across multiple seeds."""
    adapter = FeCoNiAdapter(target=target_name)
    pool = adapter.get_candidate_pool()
    oracle = adapter.create_oracle(allow_duplicate_queries=False)
    global_best = oracle.global_best_value

    worker_results = Parallel(n_jobs=n_jobs)(
        delayed(_run_aicoscientist_seed_worker)(
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
