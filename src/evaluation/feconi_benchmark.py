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
    denoised_expected_improvement_acquisition,
    expected_improvement_acquisition,
    greedy_acquisition,
    predict_latent_gp,
    ucb_acquisition,
)
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
    """Fits Gaussian Process surrogate model with Matern 5/2 kernel."""
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


class DiscreteTuRBOState:
    """Trust Region manager adapted for discrete candidate pool optimization."""

    def __init__(
        self,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        length_min: float = 0.5**7,
        length_max: float = 1.6,
        length_init: float = 0.8,
        success_tolerance: int = 3,
        failure_tolerance: int = 5,
    ) -> None:
        self.bounds_min = np.asarray(bounds_min, dtype=float)
        self.bounds_max = np.asarray(bounds_max, dtype=float)
        self.length = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.length_init = length_init
        self.success_tolerance = success_tolerance
        self.failure_tolerance = failure_tolerance
        self.success_counter = 0
        self.failure_counter = 0
        self.center_coords: np.ndarray | None = None
        self.best_value = -float("inf")

    def initialize(self, center_coords: np.ndarray, value: float) -> None:
        self.center_coords = np.asarray(center_coords, dtype=float)
        self.best_value = float(value)
        self.length = self.length_init
        self.success_counter = 0
        self.failure_counter = 0

    def update(self, center_coords: np.ndarray, value: float) -> None:
        if value > self.best_value + 1e-4:
            self.best_value = float(value)
            self.center_coords = np.asarray(center_coords, dtype=float)
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter >= self.success_tolerance:
            self.length = min(self.length * 2.0, self.length_max)
            self.success_counter = 0

        if self.failure_counter >= self.failure_tolerance:
            self.length /= 2.0
            self.failure_counter = 0

        if self.length < self.length_min:
            # Restart trust region
            self.length = self.length_init
            self.success_counter = 0
            self.failure_counter = 0

    def filter_candidates(self, candidate_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Filters candidate pool to points falling within current trust region bounding box."""
        if self.center_coords is None or self.length >= self.length_max:
            return candidate_df

        coords = candidate_df[feature_cols].to_numpy(dtype=float)
        range_span = np.maximum(self.bounds_max - self.bounds_min, 1e-6)
        # Normalized distance per dimension
        norm_dist = np.abs(coords - self.center_coords) / range_span
        # Inside box if all dim distances <= length / 2
        in_box = np.all(norm_dist <= self.length / 2.0, axis=1)
        filtered = candidate_df[in_box]
        if len(filtered) == 0:
            # Fallback to full pool if box is empty
            return candidate_df
        return filtered


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

    # Setup trust region if using turbo
    turbo_state: DiscreteTuRBOState | None = None
    if "turbo" in strategy:
        bounds_min = candidate_pool[feature_cols].min().to_numpy()
        bounds_max = candidate_pool[feature_cols].max().to_numpy()
        turbo_state = DiscreteTuRBOState(bounds_min=bounds_min, bounds_max=bounds_max)
        init_center = np.array([float(init_cand["Co"]), float(init_cand["Fe"])])
        turbo_state.initialize(init_center, float(init_res[target_name]))

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
            gp, scaler = fit_gp_surrogate(obs_df, feature_cols, target_name, seed=step_seed)

            # Determine candidate subset (TuRBO restricts to TR box)
            if turbo_state is not None:
                eval_pool = turbo_state.filter_candidates(unseen_pool, feature_cols).reset_index(drop=True)
            else:
                eval_pool = unseen_pool

            X_eval = eval_pool[feature_cols].to_numpy(dtype=float)
            X_eval_scaled = scaler.transform(X_eval)

            pred_mean, pred_std = predict_latent_gp(gp, X_eval_scaled, return_std=True)
            pred_mean = np.asarray(pred_mean, dtype=float).flatten()
            pred_std = np.asarray(pred_std, dtype=float).flatten()

            if strategy == "greedy":
                scores = greedy_acquisition(pred_mean, objective="maximize")
            elif strategy in {"gp_ucb", "gp_ucb_1", "gp_ucb_2"}:
                b = 1.0 if strategy == "gp_ucb_1" else beta
                scores = ucb_acquisition(pred_mean, pred_std, beta=b, objective="maximize")
            elif strategy in {"expected_improvement", "ei"}:
                scores = expected_improvement_acquisition(
                    mean=pred_mean,
                    std=pred_std,
                    best_observed=current_best,
                    xi=0.01,
                    objective="maximize",
                )
            elif strategy in {"noisy_expected_improvement", "nei"}:
                # Predict on observed points for denoised incumbent calculation
                X_obs = obs_df[feature_cols].to_numpy(dtype=float)
                X_obs_scaled = scaler.transform(X_obs)
                obs_pred_means = predict_latent_gp(gp, X_obs_scaled, return_std=False)
                obs_pred_means = np.asarray(obs_pred_means, dtype=float).flatten()

                scores = denoised_expected_improvement_acquisition(
                    mean=pred_mean,
                    std=pred_std,
                    observed_posterior_means=obs_pred_means,
                    xi=0.01,
                    objective="maximize",
                )
            elif strategy in {"turbo_nei", "turbo_ei"}:
                X_obs = obs_df[feature_cols].to_numpy(dtype=float)
                X_obs_scaled = scaler.transform(X_obs)
                obs_pred_means = predict_latent_gp(gp, X_obs_scaled, return_std=False)
                obs_pred_means = np.asarray(obs_pred_means, dtype=float).flatten()

                scores = denoised_expected_improvement_acquisition(
                    mean=pred_mean,
                    std=pred_std,
                    observed_posterior_means=obs_pred_means,
                    xi=0.01,
                    objective="maximize",
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

        # Update TuRBO state
        if turbo_state is not None:
            sel_coords = np.array([float(selected_row["Co"]), float(selected_row["Fe"])])
            turbo_state.update(sel_coords, obs_val)

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
            "mean_steps_to_10pct": float(np.mean(steps_to_10)),
            "median_steps_to_10pct": float(np.median(steps_to_10)),
            "mean_steps_to_5pct": float(np.mean(steps_to_5)),
            "median_steps_to_5pct": float(np.median(steps_to_5)),
            "mean_steps_to_1pct": float(np.mean(steps_to_1)),
            "median_steps_to_1pct": float(np.median(steps_to_1)),
        }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path / "per_step.csv", index=False)
        with open(out_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)

    return results_df, summary_metrics
