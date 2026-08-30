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
    strategy: str,  # "random", "greedy", "gp_ucb", "expected_improvement", "noisy_expected_improvement", "turbo_nei"
    init_sample_index: int,
    total_budget: int = 50,
    seed: int = 42,
    beta: float = 2.0,
) -> list[dict[str, Any]]:
    """Runs a single closed-loop optimization trajectory comparing AIcoScientist strategies."""
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

    # Setup trust region if using turbo (using default frozen ClosedLoopOptimizer parameters)
    turbo: TuRBOTrustRegion | None = None
    search_space = get_auirh_search_space()
    if "turbo" in strategy:
        turbo = TuRBOTrustRegion(search_space=search_space)
        turbo.initialize(
            center_candidate={"Au": float(init_cand["Au"]), "Ir": float(init_cand["Ir"])},
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
            "turbo_length": float(turbo.state.length) if (turbo and turbo.state) else np.nan,
        }
    )

    for step in range(2, total_budget + 1):
        unseen_pool = candidate_pool[~candidate_pool[AUIRH_CANDIDATE_ID_COLUMN].isin(seen_cids)].copy().reset_index(drop=True)
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

        elif strategy in {"greedy", "gp_ucb", "expected_improvement", "ei"}:
            obs_df = pd.DataFrame(observed_records)
            gp, scaler = fit_gp_surrogate(obs_df, feature_cols, target_name, seed=step_seed)

            X_unseen = unseen_pool[feature_cols].to_numpy(dtype=float)
            X_unseen_scaled = scaler.transform(X_unseen)

            pred_mean, pred_std = predict_latent_gp(gp, X_unseen_scaled, return_std=True)
            pred_mean = np.asarray(pred_mean, dtype=float).flatten()
            pred_std = np.asarray(pred_std, dtype=float).flatten()

            if strategy == "greedy":
                scores = greedy_acquisition(pred_mean, objective="maximize")
            elif strategy == "gp_ucb":
                scores = ucb_acquisition(pred_mean, pred_std, beta=beta, objective="maximize")
            else:
                scores = expected_improvement_acquisition(
                    mean=pred_mean,
                    std=pred_std,
                    best_observed=current_best,
                    xi=0.01,
                    objective="maximize",
                )

            best_idx = int(np.argmax(scores))
            selected_row = unseen_pool.iloc[best_idx]
            acq_score_val = float(scores[best_idx])

        elif strategy in {"noisy_expected_improvement", "true_nei", "nei"}:
            obs_df = pd.DataFrame(observed_records)
            gp, scaler = fit_gp_surrogate(obs_df, feature_cols, target_name, seed=step_seed)

            X_obs = obs_df[feature_cols].to_numpy(dtype=float)
            X_obs_scaled = scaler.transform(X_obs)

            X_unseen = unseen_pool[feature_cols].to_numpy(dtype=float)
            X_unseen_scaled = scaler.transform(X_unseen)

            scores = compute_true_mc_nei(
                gp=gp,
                X_observed_scaled=X_obs_scaled,
                X_candidates_scaled=X_unseen_scaled,
                n_fantasies=256,
                xi=0.01,
                objective="maximize",
                seed=step_seed,
            )
            best_idx = int(np.argmax(scores))
            selected_row = unseen_pool.iloc[best_idx]
            acq_score_val = float(scores[best_idx])

        elif strategy in {"turbo_nei", "turbo"}:
            assert turbo is not None and turbo.state is not None
            obs_df = pd.DataFrame(observed_records)
            gp, scaler = fit_gp_surrogate(obs_df, feature_cols, target_name, seed=step_seed)

            X_obs = obs_df[feature_cols].to_numpy(dtype=float)
            X_obs_scaled = scaler.transform(X_obs)

            # Check if global escape step triggers
            is_escape = turbo.should_global_escape(step)
            if is_escape:
                active_pool = unseen_pool
            else:
                # Filter candidates within trust region bounding box
                box = turbo.get_bounding_box()
                au_min, au_max = box["Au"]
                ir_min, ir_max = box["Ir"]
                in_tr = (
                    (unseen_pool["Au"] >= au_min)
                    & (unseen_pool["Au"] <= au_max)
                    & (unseen_pool["Ir"] >= ir_min)
                    & (unseen_pool["Ir"] <= ir_max)
                )
                active_pool = unseen_pool[in_tr].copy().reset_index(drop=True)
                if active_pool.empty:
                    # Fallback to global pool if trust region contains no remaining candidates
                    active_pool = unseen_pool

            X_cand = active_pool[feature_cols].to_numpy(dtype=float)
            X_cand_scaled = scaler.transform(X_cand)

            scores = compute_true_mc_nei(
                gp=gp,
                X_observed_scaled=X_obs_scaled,
                X_candidates_scaled=X_cand_scaled,
                n_fantasies=256,
                xi=0.01,
                objective="maximize",
                seed=step_seed,
            )
            best_idx = int(np.argmax(scores))
            selected_row = active_pool.iloc[best_idx]
            acq_score_val = float(scores[best_idx])

        else:
            raise ValueError(f"Unknown strategy: '{strategy}'")

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
                unseen_rem = candidate_pool[~candidate_pool[AUIRH_CANDIDATE_ID_COLUMN].isin(seen_cids)].copy().reset_index(drop=True)
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
                    fallback_center = {"Au": float(best_restart_row["Au"]), "Ir": float(best_restart_row["Ir"])}
                    fallback_cid = str(best_restart_row[AUIRH_CANDIDATE_ID_COLUMN])

            # 5. Advance TuRBO state using full covariance-aware posterior evidence
            turbo.update(
                observed_candidate={"Au": float(selected_row["Au"]), "Ir": float(selected_row["Ir"])},
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
                "turbo_length": float(turbo.state.length) if (turbo and turbo.state) else np.nan,
            }
        )

    return trajectory


def _aicoscientist_worker(args: tuple[int, int, int, str, str, str | None, int]) -> tuple[int, list[dict[str, Any]]]:
    s_idx, run_seed, init_idx, strat, target_name, library, total_budget = args
    worker_adapter = AuIrRhAdapter(target=target_name, library=library)
    worker_pool = worker_adapter.get_candidate_pool()
    worker_oracle = worker_adapter.create_oracle()
    worker_traj = run_single_aicoscientist_trajectory(
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


def run_auirh_aicoscientist_benchmark(
    target_name: str = "k0",
    library: str | None = None,
    strategies: Sequence[str] = ("random", "greedy", "gp_ucb", "expected_improvement", "true_nei", "turbo_nei"),
    n_seeds: int = 30,
    total_budget: int = 50,
    base_seed: int = 42,
    output_dir: Path | str = "outputs/auirh/aicoscientist",
) -> dict[str, Any]:
    """Executes multi-seed AIcoScientist benchmark across strategies."""
    adapter = AuIrRhAdapter(target=target_name, library=library)
    candidate_pool = adapter.get_candidate_pool()
    pool_size = len(candidate_pool)
    oracle = adapter.create_oracle()
    global_optimum = oracle.global_best_value

    mode_label = library if library else "pooled"
    logger.info("Starting Au-Ir-Rh AIcoScientist benchmark: target=%s, mode=%s, seeds=%d, pool_size=%d", target_name, mode_label, n_seeds, pool_size)

    rng_init = np.random.default_rng(base_seed)
    init_indices = [int(rng_init.integers(0, pool_size)) for _ in range(n_seeds)]

    all_step_records: list[dict[str, Any]] = []
    strategy_results: dict[str, Any] = {}

    from concurrent.futures import ProcessPoolExecutor
    max_w = min(8, n_seeds)

    for strat in strategies:
        logger.info("Running AIcoScientist strategy '%s' across %d seeds...", strat, n_seeds)
        step_bests: list[list[float]] = [[] for _ in range(n_seeds)]
        step_regrets: list[list[float]] = [[] for _ in range(n_seeds)]
        traj_results: list[list[dict[str, Any]]] = [[] for _ in range(n_seeds)]

        args_list = [
            (i, base_seed + i, init_indices[i], strat, target_name, library, total_budget)
            for i in range(n_seeds)
        ]
        with ProcessPoolExecutor(max_workers=max_w) as executor:
            for s_idx, traj in executor.map(_aicoscientist_worker, args_list):
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
        "benchmark_type": "aicoscientist_evaluation",
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

    logger.info("Saved AIcoScientist summary to %s and steps to %s", json_path, csv_path)
    return summary


def run_cross_library_diagnostic(
    target_name: str = "k0",
    n_seeds: int = 10,
    prior_samples: int = 5,
    budget_in_dest: int = 30,
    base_seed: int = 42,
    output_dir: Path | str = "outputs/auirh/cross_library",
) -> dict[str, Any]:
    """Evaluates whether observations from a source library help or hurt optimization in a destination library.

    Strict firewall: Zero target values from the destination library are seen prior to optimization.
    Pairs evaluated:
    - Au-rich -> Ir-rich
    - Au-rich -> Rh-rich
    - Ir-rich -> Rh-rich
    - Rh-rich -> Ir-rich
    """
    pairs = [
        ("Au-rich", "Ir-rich"),
        ("Au-rich", "Rh-rich"),
        ("Ir-rich", "Rh-rich"),
        ("Rh-rich", "Ir-rich"),
    ]

    results: dict[str, Any] = {}
    feature_cols = list(AUIRH_FEATURE_COLUMNS)

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

        rng = np.random.default_rng(base_seed)

        for s_idx in range(n_seeds):
            seed = base_seed + s_idx
            seed_rng = np.random.default_rng(seed)

            # Pick 1 init sample in destination
            init_dst_idx = int(seed_rng.integers(0, len(dst_pool)))
            init_dst_row = dst_pool.iloc[init_dst_idx]
            dst_oracle.reset()
            init_res = dst_oracle.query(str(init_dst_row[AUIRH_CANDIDATE_ID_COLUMN]))

            # Run baseline on destination alone (cold start with GP-UCB)
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
                unseen = dst_pool[~dst_pool[AUIRH_CANDIDATE_ID_COLUMN].isin(seen_dst)].reset_index(drop=True)
                if unseen.empty:
                    break
                obs_df = pd.DataFrame(cold_records)
                gp, sc = fit_gp_surrogate(obs_df, feature_cols, target_name, seed=seed + step)
                X_unseen_sc = sc.transform(unseen[feature_cols].to_numpy())
                m, s = predict_latent_gp(gp, X_unseen_sc, return_std=True)
                scores = ucb_acquisition(m, s, beta=2.0, objective="maximize")
                b_idx = int(np.argmax(scores))
                row_sel = unseen.iloc[b_idx]
                cid = str(row_sel[AUIRH_CANDIDATE_ID_COLUMN])
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

            # Run with source prior observations (warm start with GP-UCB)
            src_oracle.reset()
            dst_oracle.reset()
            # Query prior_samples from source library
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

            # Add the 1 destination initial observation
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
                unseen = dst_pool[~dst_pool[AUIRH_CANDIDATE_ID_COLUMN].isin(seen_dst_warm)].reset_index(drop=True)
                if unseen.empty:
                    break
                obs_df = pd.DataFrame(warm_records)
                gp, sc = fit_gp_surrogate(obs_df, feature_cols, target_name, seed=seed + step)
                X_unseen_sc = sc.transform(unseen[feature_cols].to_numpy())
                m, s = predict_latent_gp(gp, X_unseen_sc, return_std=True)
                scores = ucb_acquisition(m, s, beta=2.0, objective="maximize")
                b_idx = int(np.argmax(scores))
                row_sel = unseen.iloc[b_idx]
                cid = str(row_sel[AUIRH_CANDIDATE_ID_COLUMN])
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

        results[pair_key] = {
            "source_library": src_lib,
            "destination_library": dst_lib,
            "prior_samples_from_source": prior_samples,
            "budget_in_destination": budget_in_dest,
            "mean_regret_cold_start": float(np.mean(regrets_without_prior)),
            "std_regret_cold_start": float(np.std(regrets_without_prior)),
            "mean_regret_warm_prior": float(np.mean(regrets_with_prior)),
            "std_regret_warm_prior": float(np.std(regrets_with_prior)),
            "regret_delta": float(np.mean(regrets_with_prior) - np.mean(regrets_without_prior)),
            "transfer_impact": "helpful" if np.mean(regrets_with_prior) < np.mean(regrets_without_prior) else "neutral/unhelpful",
        }

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_file = out_p / "summary.json"
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)

    logger.info("Saved cross-library diagnostic summary to %s", out_file)
    return results
