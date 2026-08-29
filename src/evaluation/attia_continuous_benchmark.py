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

from src.datasets.attia import (
    AttiaAdapter,
    compute_expected_c4,
    generate_continuous_candidate_id,
)
from src.evaluation.attia_benchmark import compute_bootstrap_mean_ci
from src.evaluation.attia_oracle import (
    generate_attia_simulator_seed,
    simulate_attia_policy,
)
from src.optimization.acquisition import compute_acquisition
from src.optimization.adaptive_controller import AdaptiveBOController
from src.optimization.search_space import SearchSpace
from src.optimization.trust_region import TuRBOTrustRegion

logger = logging.getLogger(__name__)

ATTIA_SOURCE_COMMIT = "0068fd0136bcd65884f5cd94b2b967c1ba73a668"
SIMULATOR_VERSION = "1.0.0"


def derive_discrete_grid_optimum(discrete_pool: pd.DataFrame) -> dict[str, Any]:
    """Programmatically derives the latent optimum on the discrete 224-policy grid under variance=False."""
    best_record: dict[str, Any] = {
        "policy_id": None,
        "C1": None,
        "C2": None,
        "C3": None,
        "C4": None,
        "reference_true_lifetime": -1.0,
    }

    for _, row in discrete_pool.iterrows():
        pid = str(row["policy_id"])
        c1, c2, c3, c4 = float(row["C1"]), float(row["C2"]), float(row["C3"]), float(row["C4"])
        latent_val = float(simulate_attia_policy(c1, c2, c3, mode="hi", variance=False, seed=0))
        if latent_val > best_record["reference_true_lifetime"]:
            best_record = {
                "policy_id": pid,
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "C4": c4,
                "reference_true_lifetime": latent_val,
            }

    return best_record


def compute_or_load_continuous_reference(
    search_space: SearchSpace,
    output_dir: Path,
    discrete_pool: pd.DataFrame | None = None,
    n_sobol_samples: int = 2000,
    n_local_starts: int = 30,
    seed: int = 42,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Evaluator-only derivative-free global search to determine best-known continuous simulator reference.

    Workflow:
    1. Stage 1: Deterministic quasi-random / space-filling feasible scan of n_sobol_samples (variance=False).
    2. Stage 2: Select top n_local_starts candidate points.
    3. Stage 3: Perform derivative-free Nelder-Mead and coordinate pattern refinement around each top candidate.
    4. Save continuous_reference.json and continuous_reference_manifest.json.
    """
    ref_path = output_dir / "continuous_reference.json"
    manifest_path = output_dir / "continuous_reference_manifest.json"

    if not force_recompute and ref_path.is_file() and manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as fm:
                manifest_data = json.load(fm)
            if (
                manifest_data.get("n_sobol_samples") == n_sobol_samples
                and manifest_data.get("n_local_starts") == n_local_starts
                and manifest_data.get("evaluator_seed") == seed
                and manifest_data.get("search_method") == "derivative_free_sobol_plus_nelder_mead"
            ):
                with open(ref_path, "r", encoding="utf-8") as fr:
                    ref_data = json.load(fr)
                    if ref_data.get("best_known_latent_lifetime", 0.0) >= 1079.0:
                        return ref_data
        except Exception:
            pass

    logger.info("Computing derivative-free best-known continuous reference (n_samples=%d, starts=%d)...", n_sobol_samples, n_local_starts)

    # Stage 1: Dense deterministic feasible scan
    init_scan_df = search_space.sample_feasible(n=n_sobol_samples, seed=seed)
    scan_results: list[dict[str, Any]] = []

    for _, r in init_scan_df.iterrows():
        c1, c2, c3 = float(r["C1"]), float(r["C2"]), float(r["C3"])
        latent_life = float(simulate_attia_policy(c1, c2, c3, mode="hi", variance=False, seed=0))
        scan_results.append({"C1": c1, "C2": c2, "C3": c3, "latent_lifetime": latent_life})

    # Include discrete grid top points
    if discrete_pool is not None:
        for _, r in discrete_pool.iterrows():
            c1, c2, c3 = float(r["C1"]), float(r["C2"]), float(r["C3"])
            latent_life = float(simulate_attia_policy(c1, c2, c3, mode="hi", variance=False, seed=0))
            scan_results.append({"C1": c1, "C2": c2, "C3": c3, "latent_lifetime": latent_life})

    # Stage 2: Pick top candidates
    scan_results.sort(key=lambda x: x["latent_lifetime"], reverse=True)
    top_starts = scan_results[:n_local_starts]

    best_record: dict[str, Any] = {
        "best_known_C1": None,
        "best_known_C2": None,
        "best_known_C3": None,
        "best_known_C4": None,
        "best_known_latent_lifetime": -1.0,
        "candidate_id": None,
        "number_of_sobol_samples_evaluated": n_sobol_samples,
        "number_of_local_refinements": n_local_starts,
        "reference_search_method": "derivative_free_sobol_plus_nelder_mead",
        "reference_seed": seed,
        "description": "Best-known continuous simulator reference determined via derivative-free quasi-random scan and Nelder-Mead local refinement on latent PDE (variance=False). Not mathematically proven global optimum.",
    }

    # Stage 3: Derivative-Free Local Refinements (Nelder-Mead + Coordinate Pattern Search)
    def _obj_nelder_mead(x: np.ndarray) -> float:
        c1_v, c2_v, c3_v = float(x[0]), float(x[1]), float(x[2])
        c4_v = float(compute_expected_c4(c1_v, c2_v, c3_v))
        cand = {"C1": c1_v, "C2": c2_v, "C3": c3_v, "C4": c4_v}
        if not search_space.is_feasible(cand):
            return 1e6
        val = float(simulate_attia_policy(c1_v, c2_v, c3_v, mode="hi", variance=False, seed=0))
        return -val

    for item in top_starts:
        x0 = np.array([item["C1"], item["C2"], item["C3"]])
        res = minimize(
            _obj_nelder_mead,
            x0=x0,
            method="Nelder-Mead",
            options={"maxiter": 120, "xatol": 1e-4, "fatol": 1e-4},
        )
        candidates_to_eval = [x0]
        if res.success:
            candidates_to_eval.append(res.x)

        # Coordinate pattern search around candidate
        for delta in [0.05, 0.02, 0.01, 0.005]:
            for i in range(3):
                for sign in [-1.0, 1.0]:
                    x_step = np.copy(res.x if res.success else x0)
                    x_step[i] += sign * delta
                    candidates_to_eval.append(x_step)

        for pt in candidates_to_eval:
            c1_cand, c2_cand, c3_cand = float(pt[0]), float(pt[1]), float(pt[2])
            c4_cand = float(compute_expected_c4(c1_cand, c2_cand, c3_cand))
            cand_dict = {"C1": round(c1_cand, 4), "C2": round(c2_cand, 4), "C3": round(c3_cand, 4), "C4": round(c4_cand, 4)}
            if search_space.is_feasible(cand_dict):
                latent_v = float(simulate_attia_policy(cand_dict["C1"], cand_dict["C2"], cand_dict["C3"], mode="hi", variance=False, seed=0))
                if latent_v > best_record["best_known_latent_lifetime"]:
                    cand_id = generate_continuous_candidate_id(cand_dict["C1"], cand_dict["C2"], cand_dict["C3"], cand_dict["C4"])
                    best_record["best_known_C1"] = cand_dict["C1"]
                    best_record["best_known_C2"] = cand_dict["C2"]
                    best_record["best_known_C3"] = cand_dict["C3"]
                    best_record["best_known_C4"] = cand_dict["C4"]
                    best_record["best_known_latent_lifetime"] = latent_v
                    best_record["candidate_id"] = cand_id

    # Save reference JSON and Manifest
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(best_record, f, indent=2)

    manifest_meta = {
        "simulator_version": SIMULATOR_VERSION,
        "attia_source_commit": ATTIA_SOURCE_COMMIT,
        "search_space_name": search_space.name,
        "search_method": "derivative_free_sobol_plus_nelder_mead",
        "n_sobol_samples": n_sobol_samples,
        "n_local_starts": n_local_starts,
        "evaluator_seed": seed,
        "variance_mode": False,
        "best_known_candidate_id": best_record["candidate_id"],
        "best_known_latent_lifetime": best_record["best_known_latent_lifetime"],
        "coordinates": {
            "C1": best_record["best_known_C1"],
            "C2": best_record["best_known_C2"],
            "C3": best_record["best_known_C3"],
            "C4": best_record["best_known_C4"],
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_meta, f, indent=2)

    return best_record


def run_single_attia_continuous_trajectory(
    search_space: SearchSpace,
    discrete_pool: pd.DataFrame,
    init_indices: list[int],
    total_queries: int,
    strategy: str,  # "random", "greedy", "gp_ucb", "expected_improvement", "adaptive"
    optimizer_seed: int,
    beta: float = 1.0,
    n_candidates_per_step: int = 5000,
    refine_continuous: bool = True,
    duplicate_tol: float = 1e-3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Runs a single continuous closed-loop Bayesian Optimization trajectory.

    Optimizer Firewall:
    - Zero access to latent reference values (variance=False) or regret metrics.
    - GP surrogate fits exclusively on observed simulated_lifetime.
    - Returns (raw_trajectory_history, decision_trace_records).
    """
    feature_cols = ["C1", "C2", "C3", "C4"]

    controller = AdaptiveBOController() if strategy == "adaptive" else None

    # Initial observations from discrete warmup pool
    observed_records: list[dict[str, Any]] = []

    for w_idx, idx in enumerate(init_indices):
        cand_row = discrete_pool.iloc[idx]
        c1, c2, c3, c4 = float(cand_row["C1"]), float(cand_row["C2"]), float(cand_row["C3"]), float(cand_row["C4"])
        cand_id = str(cand_row["policy_id"])
        query_id = f"Q_S{optimizer_seed:02d}_{strategy}_ST00_INIT{w_idx:02d}"

        sim_seed = generate_attia_simulator_seed(benchmark_seed=optimizer_seed, policy_id=cand_id)
        sim_life = float(simulate_attia_policy(c1, c2, c3, mode="hi", variance=True, seed=sim_seed))

        observed_records.append(
            {
                "query_id": query_id,
                "candidate_id": cand_id,
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "C4": c4,
                "simulator_seed": sim_seed,
                "simulated_lifetime": sim_life,
                "is_off_grid": False,
                "min_distance_to_grid": 0.0,
                "is_new_vs_observed": True,
                "min_distance_to_observed": 0.0,
            }
        )

    history: list[dict[str, Any]] = []
    decision_trace: list[dict[str, Any]] = []
    current_best_sim = max(r["simulated_lifetime"] for r in observed_records)

    turbo: TuRBOTrustRegion | None = None
    if strategy == "turbo_nei":
        best_warmup_idx = int(np.argmax([r["simulated_lifetime"] for r in observed_records]))
        best_warmup_row = observed_records[best_warmup_idx]
        turbo = TuRBOTrustRegion(search_space=search_space, init_radius=0.8)
        turbo.initialize(best_warmup_row, best_warmup_row["simulated_lifetime"])

    # Initial state (Step 0 summary)
    history.append(
        {
            "benchmark_seed": optimizer_seed,
            "strategy": strategy,
            "step": 0,
            "query_id": None,
            "candidate_id": None,
            "C1": None,
            "C2": None,
            "C3": None,
            "C4": None,
            "simulator_seed": None,
            "simulated_lifetime": None,
            "best_observed_lifetime": current_best_sim,
            "is_off_grid": False,
            "min_distance_to_grid": 0.0,
            "is_new_vs_observed": False,
            "min_distance_to_observed": 0.0,
            "duplicate_rejections_at_step": 0,
            "acquisition_method": None,
            "acquisition_score": None,
            "exploration_score": None,
            "exploitation_score": None,
            "controller_reason": None,
            "should_stop": False,
            "stop_reason": None,
            "trust_region_center": json.dumps(turbo.state.center) if turbo and turbo.state else None,
            "trust_region_radius": float(turbo.state.radius) if turbo and turbo.state else None,
            "success_counter": int(turbo.state.success_counter) if turbo and turbo.state else 0,
            "failure_counter": int(turbo.state.failure_counter) if turbo and turbo.state else 0,
            "expanded": False,
            "contracted": False,
            "restarted": False,
        }
    )

    # Closed-loop Continuous BO iterations
    for step in range(1, total_queries + 1):
        # 1. Sample candidate batch (within trust region if turbo_nei, else globally)
        if strategy == "turbo_nei" and turbo is not None:
            cand_batch = turbo.sample_candidates(
                n=n_candidates_per_step,
                seed=optimizer_seed * 1000 + step * 100 + 7,
            )
        else:
            cand_batch = search_space.sample_feasible(
                n=n_candidates_per_step,
                seed=optimizer_seed * 1000 + step * 100 + 7,
            )

        # 2. Fit GP surrogate on observed simulated observations
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

        # Observed posterior means for denoised incumbent in NEI
        obs_posterior_means = gp.predict(X_train_scaled)

        # 3. Predict acquisition across feasible candidates
        X_cand = cand_batch[feature_cols].to_numpy(dtype=float)
        X_cand_scaled = scaler.transform(X_cand)
        pred_mean, pred_std = gp.predict(X_cand_scaled, return_std=True)

        observed_df = pd.DataFrame(observed_records)
        novelty_vs_observed = search_space.check_novelty(
            cand_batch,
            reference_points=observed_df,
            feature_cols=feature_cols,
            tol=duplicate_tol,
        )

        step_duplicate_rejections = 0
        current_method = strategy
        current_beta = beta
        current_xi = 0.01
        expl_score: float | None = None
        explt_score: float | None = None
        ctrl_reason: str | None = None
        should_stop_val = False
        stop_reason_val: str | None = None

        if strategy == "random":
            non_dup_indices = np.where(novelty_vs_observed["min_distance"].to_numpy() >= duplicate_tol)[0]
            rng_step = np.random.default_rng(optimizer_seed + step * 1000)
            if len(non_dup_indices) > 0:
                selected_idx = int(rng_step.choice(non_dup_indices))
            else:
                selected_idx = int(rng_step.integers(0, len(cand_batch)))
            best_cand_dict = cand_batch.iloc[selected_idx].to_dict()
            scores = np.zeros(len(cand_batch))
            acq_score_chosen = 0.0

        elif strategy in {"greedy", "gp_ucb", "expected_improvement", "nei", "turbo_nei", "adaptive"}:
            if strategy == "adaptive" and controller is not None:
                decision = controller.decide(
                    step=step,
                    total_queries=total_queries,
                    observed_targets=y_train,
                    pred_mean=pred_mean,
                    pred_std=pred_std,
                )
                current_method = decision.chosen_method
                current_beta = decision.beta
                current_xi = decision.xi
                expl_score = decision.exploration_score
                explt_score = decision.exploitation_score
                ctrl_reason = decision.controller_reason
                should_stop_val = decision.should_stop
                stop_reason_val = decision.stop_reason
            elif strategy in {"nei", "turbo_nei"}:
                current_method = "nei"
            else:
                current_method = strategy

            scores = compute_acquisition(
                method=current_method,
                mean=pred_mean,
                std=pred_std,
                best_observed=current_best_sim,
                beta=current_beta,
                xi=current_xi,
                observed_posterior_means=obs_posterior_means,
            )

            sorted_indices = np.argsort(scores)[::-1]
            selected_cand = None
            selected_score = 0.0

            for cand_i in sorted_indices:
                if novelty_vs_observed["min_distance"].iloc[cand_i] >= duplicate_tol:
                    selected_cand = cand_batch.iloc[cand_i].to_dict()
                    selected_score = float(scores[cand_i])
                    break
                else:
                    step_duplicate_rejections += 1

            if selected_cand is None:
                selected_cand = cand_batch.iloc[sorted_indices[0]].to_dict()
                selected_score = float(scores[sorted_indices[0]])

            best_cand_dict = selected_cand
            acq_score_chosen = selected_score

            # Optional local continuous refinement on the smooth surrogate acquisition function
            if refine_continuous and strategy not in {"turbo_nei"}:
                init_c1 = float(best_cand_dict["C1"])
                init_c2 = float(best_cand_dict["C2"])
                init_c3 = float(best_cand_dict["C3"])

                def _obj(x: np.ndarray) -> float:
                    c1_v, c2_v, c3_v = float(x[0]), float(x[1]), float(x[2])
                    c4_v = float(compute_expected_c4(c1_v, c2_v, c3_v))
                    cand_t = {"C1": c1_v, "C2": c2_v, "C3": c3_v, "C4": c4_v}
                    if not search_space.is_feasible(cand_t):
                        return 1e6

                    x_feat = np.array([[c1_v, c2_v, c3_v, c4_v]])
                    x_sc = scaler.transform(x_feat)
                    m, s = gp.predict(x_sc, return_std=True)
                    score_val = compute_acquisition(
                        method=current_method,
                        mean=m,
                        std=s,
                        best_observed=current_best_sim,
                        beta=current_beta,
                        xi=current_xi,
                        observed_posterior_means=obs_posterior_means,
                    )
                    return -float(score_val[0])

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
                    ref_cand = {"C1": round(ref_c1, 4), "C2": round(ref_c2, 4), "C3": round(ref_c3, 4), "C4": round(ref_c4, 4)}
                    if search_space.is_feasible(ref_cand):
                        ref_df = pd.DataFrame([ref_cand])
                        ref_nov = search_space.check_novelty(ref_df, reference_points=observed_df, feature_cols=feature_cols, tol=duplicate_tol)
                        if ref_nov["min_distance"].iloc[0] >= duplicate_tol:
                            best_cand_dict = ref_cand
                            acq_score_chosen = -float(opt_res.fun)
                        else:
                            step_duplicate_rejections += 1

        else:
            raise ValueError(f"Unknown optimization strategy: {strategy!r}")

        c1 = float(best_cand_dict["C1"])
        c2 = float(best_cand_dict["C2"])
        c3 = float(best_cand_dict["C3"])
        c4 = float(compute_expected_c4(c1, c2, c3))

        single_cand_df = pd.DataFrame([{"C1": c1, "C2": c2, "C3": c3, "C4": c4}])
        novelty_grid = search_space.check_novelty(
            single_cand_df,
            reference_points=discrete_pool,
            feature_cols=feature_cols,
            tol=duplicate_tol,
        )
        is_off_grid = bool(novelty_grid["is_novel"].iloc[0])
        dist_to_grid = float(novelty_grid["min_distance"].iloc[0])

        novelty_obs = search_space.check_novelty(
            single_cand_df,
            reference_points=pd.DataFrame(observed_records),
            feature_cols=feature_cols,
            tol=duplicate_tol,
        )
        is_new_obs = bool(novelty_obs["is_novel"].iloc[0])
        dist_to_obs = float(novelty_obs["min_distance"].iloc[0])

        cand_id = generate_continuous_candidate_id(c1, c2, c3, c4)
        query_id = f"Q_S{optimizer_seed:02d}_{strategy}_ST{step:02d}"

        # Query simulator oracle with fair stochastic seed
        sim_seed = generate_attia_simulator_seed(benchmark_seed=optimizer_seed, policy_id=cand_id)
        sim_life = float(simulate_attia_policy(c1, c2, c3, mode="hi", variance=True, seed=sim_seed))

        # Predict mean & std at selected point for logging
        sc_pt = scaler.transform([[c1, c2, c3, c4]])
        p_mean, p_std = gp.predict(sc_pt, return_std=True)

        observed_records.append(
            {
                "query_id": query_id,
                "candidate_id": cand_id,
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "C4": c4,
                "simulator_seed": sim_seed,
                "simulated_lifetime": sim_life,
                "is_off_grid": is_off_grid,
                "min_distance_to_grid": dist_to_grid,
                "is_new_vs_observed": is_new_obs,
                "min_distance_to_observed": dist_to_obs,
            }
        )

        current_best_sim = max(current_best_sim, sim_life)

        tr_update = {}
        if strategy == "turbo_nei" and turbo is not None:
            tr_update = turbo.update(
                observed_candidate={"C1": c1, "C2": c2, "C3": c3, "C4": c4},
                observed_value=sim_life,
                objective="maximize",
            )

        row_meta = {
            "benchmark_seed": optimizer_seed,
            "strategy": strategy,
            "step": step,
            "query_id": query_id,
            "candidate_id": cand_id,
            "C1": c1,
            "C2": c2,
            "C3": c3,
            "C4": c4,
            "simulator_seed": sim_seed,
            "simulated_lifetime": sim_life,
            "best_observed_lifetime": current_best_sim,
            "is_off_grid": is_off_grid,
            "min_distance_to_grid": dist_to_grid,
            "is_new_vs_observed": is_new_obs,
            "min_distance_to_observed": dist_to_obs,
            "duplicate_rejections_at_step": step_duplicate_rejections,
            "acquisition_method": current_method,
            "acquisition_score": acq_score_chosen,
            "exploration_score": expl_score,
            "exploitation_score": explt_score,
            "controller_reason": ctrl_reason,
            "should_stop": should_stop_val,
            "stop_reason": stop_reason_val,
            "trust_region_center": json.dumps(turbo.state.center) if turbo and turbo.state else None,
            "trust_region_radius": float(turbo.state.radius) if turbo and turbo.state else None,
            "success_counter": int(turbo.state.success_counter) if turbo and turbo.state else 0,
            "failure_counter": int(turbo.state.failure_counter) if turbo and turbo.state else 0,
            "expanded": bool(tr_update.get("expanded", False)),
            "contracted": bool(tr_update.get("contracted", False)),
            "restarted": bool(tr_update.get("restarted", False)),
        }
        history.append(row_meta)

        if strategy == "adaptive":
            decision_trace.append(
                {
                    "benchmark_seed": optimizer_seed,
                    "step": step,
                    "candidate_id": cand_id,
                    "C1": c1,
                    "C2": c2,
                    "C3": c3,
                    "C4": c4,
                    "predicted_mean": float(p_mean[0]),
                    "predicted_std": float(p_std[0]),
                    "acquisition_method": current_method,
                    "acquisition_score": acq_score_chosen,
                    "exploration_score": expl_score,
                    "exploitation_score": explt_score,
                    "controller_reason": ctrl_reason,
                    "simulated_lifetime": sim_life,
                    "best_observed_lifetime": current_best_sim,
                    "should_stop": should_stop_val,
                    "stop_reason": stop_reason_val,
                }
            )

    return history, decision_trace


def evaluate_continuous_trajectory(
    raw_trajectory: list[dict[str, Any]],
    init_indices: list[int],
    discrete_pool: pd.DataFrame,
    continuous_ref_lifetime: float,
    discrete_grid_optimum_lifetime: float,
) -> tuple[list[dict[str, Any]], bool]:
    """Evaluator-only function to join raw optimizer history with latent simulator evaluations.

    Returns (evaluated_history, reference_underestimated).
    """
    init_true_lifetimes = []
    for idx in init_indices:
        r = discrete_pool.iloc[idx]
        tl = float(simulate_attia_policy(float(r["C1"]), float(r["C2"]), float(r["C3"]), mode="hi", variance=False, seed=0))
        init_true_lifetimes.append(tl)

    current_best_true = max(init_true_lifetimes)
    evaluated_history: list[dict[str, Any]] = []
    reference_underestimated = False

    for row in raw_trajectory:
        evaluated_row = dict(row)

        if row["step"] == 0:
            evaluated_row["reference_true_lifetime"] = None
            evaluated_row["best_reference_true"] = current_best_true
            evaluated_row["continuous_simple_regret"] = max(0.0, continuous_ref_lifetime - current_best_true)
            evaluated_row["gap_to_discrete_grid_optimum"] = max(0.0, discrete_grid_optimum_lifetime - current_best_true)
            evaluated_row["improvement_over_discrete_grid"] = max(0.0, current_best_true - discrete_grid_optimum_lifetime)
            evaluated_row["beats_discrete_grid"] = bool(current_best_true > discrete_grid_optimum_lifetime)
        else:
            c1, c2, c3 = float(row["C1"]), float(row["C2"]), float(row["C3"])
            true_life = float(simulate_attia_policy(c1, c2, c3, mode="hi", variance=False, seed=0))
            if true_life > continuous_ref_lifetime + 1e-4:
                reference_underestimated = True

            current_best_true = max(current_best_true, true_life)

            evaluated_row["reference_true_lifetime"] = true_life
            evaluated_row["best_reference_true"] = current_best_true
            evaluated_row["continuous_simple_regret"] = max(0.0, continuous_ref_lifetime - current_best_true)
            evaluated_row["gap_to_discrete_grid_optimum"] = max(0.0, discrete_grid_optimum_lifetime - current_best_true)
            evaluated_row["improvement_over_discrete_grid"] = max(0.0, current_best_true - discrete_grid_optimum_lifetime)
            evaluated_row["beats_discrete_grid"] = bool(current_best_true > discrete_grid_optimum_lifetime)

        evaluated_history.append(evaluated_row)

    return evaluated_history, reference_underestimated


def run_attia_continuous_benchmark(
    adapter: AttiaAdapter | None = None,
    budgets: Sequence[int] = (10, 15, 20, 30),
    initial_policies: int = 5,
    n_seeds: int = 30,
    beta: float = 1.0,
    n_candidates_per_step: int = 5000,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Runs the hardened continuous Bayesian Optimization benchmark across multiple budgets and paired seeds."""
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

    # 1. Programmatically derive discrete grid optimum
    discrete_grid_optimum = derive_discrete_grid_optimum(discrete_pool)
    discrete_opt_true = float(discrete_grid_optimum["reference_true_lifetime"])

    # 2. Compute or load evaluator-only best-known continuous reference via derivative-free global search
    cont_ref_meta = compute_or_load_continuous_reference(
        search_space=search_space,
        output_dir=output_dir,
        discrete_pool=discrete_pool,
        n_sobol_samples=2000,
        n_local_starts=30,
        seed=42,
    )
    cont_ref_opt_true = float(cont_ref_meta["best_known_latent_lifetime"])

    # 3. Save search space provenance summary
    search_space_meta = {
        **search_space.to_dict(),
        "search_space_provenance": "continuous relaxation/interpolation of the parameter ranges used to construct the authors' discrete 224-policy space",
        "scientific_disclaimer": "The continuous search space is evaluated strictly under the author numerical PDE/Arrhenius degradation simulator. Off-grid policies have NOT been physically validated on battery hardware.",
        "discrete_reference_grid_size": len(discrete_pool),
        "derived_discrete_grid_optimum": discrete_grid_optimum,
        "best_known_continuous_reference": cont_ref_meta,
    }
    with open(output_dir / "search_space_summary.json", "w", encoding="utf-8") as f:
        json.dump(search_space_meta, f, indent=2)

    strategies = ["random", "greedy", "gp_ucb", "expected_improvement", "nei", "turbo_nei", "adaptive"]
    max_budget = max(budgets)
    max_queries = max_budget - initial_policies

    full_evaluated_trajectories: dict[int, dict[str, list[dict[str, Any]]]] = {}
    all_proposed_protocols: list[dict[str, Any]] = []
    all_history_records: list[dict[str, Any]] = []
    all_adaptive_decision_records: list[dict[str, Any]] = []
    all_turbo_state_records: list[dict[str, Any]] = []
    any_ref_underestimated = False

    logger.info(
        "Running hardened %d-strategy Continuous BO benchmark across %d seeds and budgets %s...",
        len(strategies),
        n_seeds,
        budgets,
    )

    for seed_idx in range(n_seeds):
        seed_rng = np.random.default_rng(seed_idx * 7919 + 42)
        init_indices = seed_rng.choice(len(discrete_pool), size=initial_policies, replace=False).tolist()

        full_evaluated_trajectories[seed_idx] = {}

        for strat in strategies:
            # Run pure optimizer (NO reference truth inside)
            raw_hist, decision_records = run_single_attia_continuous_trajectory(
                search_space=search_space,
                discrete_pool=discrete_pool,
                init_indices=init_indices,
                total_queries=max_queries,
                strategy=strat,
                optimizer_seed=seed_idx,
                beta=beta,
                n_candidates_per_step=n_candidates_per_step,
            )
            if strat == "adaptive":
                all_adaptive_decision_records.extend(decision_records)

            # Post-hoc evaluator joins latent true values
            eval_hist, ref_under = evaluate_continuous_trajectory(
                raw_trajectory=raw_hist,
                init_indices=init_indices,
                discrete_pool=discrete_pool,
                continuous_ref_lifetime=cont_ref_opt_true,
                discrete_grid_optimum_lifetime=discrete_opt_true,
            )
            if ref_under:
                any_ref_underestimated = True

            full_evaluated_trajectories[seed_idx][strat] = eval_hist

            for row in eval_hist:
                if row["step"] > 0:
                    all_proposed_protocols.append(
                        {
                            "benchmark_seed": seed_idx,
                            "strategy": strat,
                            "step": row["step"],
                            "query_id": row["query_id"],
                            "candidate_id": row["candidate_id"],
                            "C1": row["C1"],
                            "C2": row["C2"],
                            "C3": row["C3"],
                            "C4": row["C4"],
                            "simulator_seed": row["simulator_seed"],
                            "simulated_lifetime": row["simulated_lifetime"],
                            "reference_true_lifetime": row["reference_true_lifetime"],
                            "is_off_grid": row["is_off_grid"],
                            "min_distance_to_grid": row["min_distance_to_grid"],
                            "is_new_vs_observed": row["is_new_vs_observed"],
                            "min_distance_to_observed": row["min_distance_to_observed"],
                        }
                    )
                    if strat == "turbo_nei":
                        all_turbo_state_records.append(
                            {
                                "benchmark_seed": seed_idx,
                                "step": row["step"],
                                "trust_region_center": row.get("trust_region_center"),
                                "trust_region_radius": row.get("trust_region_radius"),
                                "success_counter": row.get("success_counter", 0),
                                "failure_counter": row.get("failure_counter", 0),
                                "expanded": bool(row.get("expanded", False)),
                                "contracted": bool(row.get("contracted", False)),
                                "restarted": bool(row.get("restarted", False)),
                                "candidate_id": row["candidate_id"],
                                "acquisition_score": row.get("acquisition_score"),
                            }
                        )

    # Compute metrics for each budget
    budget_sweep_results: list[dict[str, Any]] = []
    best_discovered_per_strategy: dict[str, dict[str, Any]] = {
        s: {
            "reference_true_lifetime": -1.0,
            "candidate_id": None,
            "C1": None,
            "C2": None,
            "C3": None,
            "C4": None,
            "seed": None,
            "step": None,
            "is_off_grid": False,
            "min_distance_to_grid": None,
            "improvement_over_discrete_grid": 0.0,
        }
        for s in strategies
    }

    for budget in budgets:
        n_queries = budget - initial_policies
        strategy_metrics: dict[str, dict[str, Any]] = {}

        strat_bests: dict[str, list[float]] = {s: [] for s in strategies}
        strat_cont_regrets: dict[str, list[float]] = {s: [] for s in strategies}
        strat_grid_gaps: dict[str, list[float]] = {s: [] for s in strategies}
        strat_grid_gap_aucs: dict[str, list[float]] = {s: [] for s in strategies}
        strat_off_grid_rates: dict[str, list[float]] = {s: [] for s in strategies}
        strat_beats_grid: dict[str, list[bool]] = {s: [] for s in strategies}
        strat_queries_to_beat: dict[str, list[int]] = {s: [] for s in strategies}
        strat_improvements: dict[str, list[float]] = {s: [] for s in strategies}
        strat_dup_rejections: dict[str, list[int]] = {s: [] for s in strategies}
        strat_stop_signals: dict[str, list[int]] = {s: [] for s in strategies}
        strat_method_switches: dict[str, list[int]] = {s: [] for s in strategies}

        for seed_idx in range(n_seeds):
            for strat in strategies:
                hist = full_evaluated_trajectories[seed_idx][strat]
                sliced = [h for h in hist if h["step"] <= n_queries]

                final_row = sliced[-1]
                best_val = float(final_row["best_reference_true"])
                cont_regret = float(final_row["continuous_simple_regret"])
                grid_gap = float(final_row["gap_to_discrete_grid_optimum"])
                imp_val = float(final_row["improvement_over_discrete_grid"])
                beats_grid = bool(final_row["beats_discrete_grid"])

                strat_bests[strat].append(best_val)
                strat_cont_regrets[strat].append(cont_regret)
                strat_grid_gaps[strat].append(grid_gap)
                strat_beats_grid[strat].append(beats_grid)
                strat_improvements[strat].append(imp_val)

                # Steps to first beat discrete grid
                first_beat_step = next((h["step"] for h in sliced if h["beats_discrete_grid"]), n_queries + 1)
                strat_queries_to_beat[strat].append(first_beat_step)

                # Stopping signal step
                stop_step = next((h["step"] for h in sliced if h.get("should_stop", False)), n_queries + 1)
                strat_stop_signals[strat].append(stop_step)

                # Method switches count for adaptive
                methods_seq = [h["acquisition_method"] for h in sliced if h["step"] > 0 and h["acquisition_method"]]
                switches = sum(1 for i in range(len(methods_seq) - 1) if methods_seq[i] != methods_seq[i + 1])
                strat_method_switches[strat].append(switches)

                grid_gap_series = [h["gap_to_discrete_grid_optimum"] for h in sliced]
                strat_grid_gap_aucs[strat].append(float(np.mean(grid_gap_series)))

                off_grid_count = sum(1 for h in sliced if h["step"] > 0 and h["is_off_grid"])
                query_count = sum(1 for h in sliced if h["step"] > 0)
                off_grid_rate = (off_grid_count / query_count) if query_count > 0 else 0.0
                strat_off_grid_rates[strat].append(off_grid_rate)

                tot_dup_rej = sum(h["duplicate_rejections_at_step"] for h in sliced)
                strat_dup_rejections[strat].append(tot_dup_rej)

                # Track best per strategy
                for h in sliced:
                    if h["step"] > 0 and float(h["reference_true_lifetime"]) > best_discovered_per_strategy[strat]["reference_true_lifetime"]:
                        best_discovered_per_strategy[strat] = {
                            "reference_true_lifetime": float(h["reference_true_lifetime"]),
                            "candidate_id": h["candidate_id"],
                            "C1": h["C1"],
                            "C2": h["C2"],
                            "C3": h["C3"],
                            "C4": h["C4"],
                            "seed": seed_idx,
                            "step": h["step"],
                            "is_off_grid": bool(h["is_off_grid"]),
                            "min_distance_to_grid": float(h["min_distance_to_grid"]),
                            "improvement_over_discrete_grid": max(0.0, float(h["reference_true_lifetime"]) - discrete_opt_true),
                        }

                if budget == max_budget:
                    for h in sliced:
                        all_history_records.append({**h, "budget": budget})

        for strat in strategies:
            bests = np.array(strat_bests[strat], dtype=float)
            cont_regs = np.array(strat_cont_regrets[strat], dtype=float)
            grid_gaps = np.array(strat_grid_gaps[strat], dtype=float)
            grid_aucs = np.array(strat_grid_gap_aucs[strat], dtype=float)
            off_grids = np.array(strat_off_grid_rates[strat], dtype=float)
            beats = np.array(strat_beats_grid[strat], dtype=bool)
            q_beats = np.array(strat_queries_to_beat[strat], dtype=int)
            imps = np.array(strat_improvements[strat], dtype=float)
            dup_rejs = np.array(strat_dup_rejections[strat], dtype=int)
            stops = np.array(strat_stop_signals[strat], dtype=int)
            switches = np.array(strat_method_switches[strat], dtype=int)

            best_ci_low, best_ci_high = compute_bootstrap_mean_ci(bests, n_bootstraps=2000, seed=1042 + budget)
            reg_ci_low, reg_ci_high = compute_bootstrap_mean_ci(cont_regs, n_bootstraps=2000, seed=2042 + budget)
            gap_ci_low, gap_ci_high = compute_bootstrap_mean_ci(grid_gaps, n_bootstraps=2000, seed=3042 + budget)

            cond_imps = imps[beats]
            mean_cond_imp = float(np.mean(cond_imps)) if len(cond_imps) > 0 else 0.0

            successful_beats_queries = q_beats[beats]
            mean_queries_to_beat = float(np.mean(successful_beats_queries)) if len(successful_beats_queries) > 0 else None

            strat_dict: dict[str, Any] = {
                "mean_best_seen": float(np.mean(bests)),
                "std_best_seen": float(np.std(bests, ddof=1)) if len(bests) > 1 else 0.0,
                "median_best_seen": float(np.median(bests)),
                "mean_best_seen_95_ci": [float(best_ci_low), float(best_ci_high)],
                "mean_continuous_simple_regret": float(np.mean(cont_regs)),
                "std_continuous_simple_regret": float(np.std(cont_regs, ddof=1)) if len(cont_regs) > 1 else 0.0,
                "median_continuous_simple_regret": float(np.median(cont_regs)),
                "mean_continuous_simple_regret_95_ci": [float(reg_ci_low), float(reg_ci_high)],
                "mean_gap_to_discrete_grid": float(np.mean(grid_gaps)),
                "mean_gap_to_discrete_grid_95_ci": [float(gap_ci_low), float(gap_ci_high)],
                "mean_grid_gap_auc": float(np.mean(grid_aucs)),
                "pct_seeds_beating_discrete_grid": float(np.mean(beats) * 100.0),
                "mean_queries_to_beat_discrete_grid": mean_queries_to_beat,
                "mean_improvement_conditional_on_beating_grid": mean_cond_imp,
                "max_improvement_over_grid": float(np.max(imps)),
                "mean_off_grid_proposal_rate": float(np.mean(off_grids)),
                "total_duplicate_rejections": int(np.sum(dup_rejs)),
                "mean_stopping_signal_step": float(np.mean(stops)),
                "mean_method_switches": float(np.mean(switches)),
            }

            if strat == "turbo_nei":
                seed_exp = [sum(1 for h in full_evaluated_trajectories[s_idx][strat] if 0 < h["step"] <= n_queries and h.get("expanded", False)) for s_idx in range(n_seeds)]
                seed_con = [sum(1 for h in full_evaluated_trajectories[s_idx][strat] if 0 < h["step"] <= n_queries and h.get("contracted", False)) for s_idx in range(n_seeds)]
                seed_res = [sum(1 for h in full_evaluated_trajectories[s_idx][strat] if 0 < h["step"] <= n_queries and h.get("restarted", False)) for s_idx in range(n_seeds)]
                seed_rad = [next((float(h.get("trust_region_radius")) for h in reversed(full_evaluated_trajectories[s_idx][strat]) if h["step"] <= n_queries and h.get("trust_region_radius") is not None), 0.8) for s_idx in range(n_seeds)]
                strat_dict["mean_trust_region_expansions"] = float(np.mean(seed_exp))
                strat_dict["mean_trust_region_contractions"] = float(np.mean(seed_con))
                strat_dict["mean_trust_region_restarts"] = float(np.mean(seed_res))
                strat_dict["final_trust_region_radius"] = float(np.mean(seed_rad))

            strategy_metrics[strat] = strat_dict

        def _safe_wilcoxon(d: np.ndarray) -> float:
            if np.all(d == 0):
                return 1.0
            try:
                stat, p_val = wilcoxon(d, alternative="two-sided")
                return float(p_val)
            except Exception:
                return 1.0

        paired_comparisons = {}
        pair_tuples = [
            ("turbo_nei_vs_random", "turbo_nei", "random"),
            ("turbo_nei_vs_greedy", "turbo_nei", "greedy"),
            ("turbo_nei_vs_gp_ucb", "turbo_nei", "gp_ucb"),
            ("turbo_nei_vs_ei", "turbo_nei", "expected_improvement"),
            ("turbo_nei_vs_nei", "turbo_nei", "nei"),
            ("turbo_nei_vs_adaptive", "turbo_nei", "adaptive"),
            ("nei_vs_ei", "nei", "expected_improvement"),
            ("adaptive_vs_random", "adaptive", "random"),
            ("adaptive_vs_greedy", "adaptive", "greedy"),
            ("adaptive_vs_gp_ucb", "adaptive", "gp_ucb"),
            ("adaptive_vs_ei", "adaptive", "expected_improvement"),
            ("ucb_vs_greedy", "gp_ucb", "greedy"),
        ]
        for pair_name, strat_a, strat_b in pair_tuples:
            d_best = np.array(strat_bests[strat_a]) - np.array(strat_bests[strat_b])
            d_reg = np.array(strat_cont_regrets[strat_a]) - np.array(strat_cont_regrets[strat_b])
            paired_comparisons[pair_name] = {
                "best_diff": {
                    "mean": float(np.mean(d_best)),
                    "median": float(np.median(d_best)),
                    "std": float(np.std(d_best, ddof=1)) if len(d_best) > 1 else 0.0,
                    "p_value": _safe_wilcoxon(d_best),
                },
                "regret_diff": {
                    "mean": float(np.mean(d_reg)),
                    "median": float(np.median(d_reg)),
                    "std": float(np.std(d_reg, ddof=1)) if len(d_reg) > 1 else 0.0,
                    "p_value": _safe_wilcoxon(d_reg),
                },
            }

        budget_sweep_results.append(
            {
                "budget": budget,
                "initial_policies": initial_policies,
                "queries": n_queries,
                "strategies": strategy_metrics,
                "paired_comparisons": paired_comparisons,
            }
        )

    # Save output artifacts
    pd.DataFrame(all_history_records).to_csv(output_dir / "optimization_history.csv", index=False)
    pd.DataFrame(all_proposed_protocols).to_csv(output_dir / "proposed_protocols.csv", index=False)
    pd.DataFrame(all_adaptive_decision_records).to_csv(output_dir / "adaptive_decision_trace.csv", index=False)
    pd.DataFrame(all_turbo_state_records).to_csv(output_dir / "turbo_state_history.csv", index=False)

    final_budget_entry = budget_sweep_results[-1]
    overall_best_continuous = max(best_discovered_per_strategy.values(), key=lambda x: x["reference_true_lifetime"])

    benchmark_summary = {
        "benchmark": "Attia et al. 2020 Hardened Continuous Fast-Charging Bayesian Optimization Benchmark",
        "benchmark_nature": "simulator != experimental dataset",
        "scientific_disclaimer": "The continuous search space is a numerical relaxation of the parameter bounds used in Attia et al. (Nature 2020). Performance is evaluated strictly under the author PDE thermal/Arrhenius degradation simulator. No physical experimental discovery is claimed.",
        "reference_underestimated": any_ref_underestimated,
        "derived_discrete_grid_optimum": discrete_grid_optimum,
        "best_known_continuous_reference": cont_ref_meta,
        "overall_best_continuous_discovered": overall_best_continuous,
        "best_discovered_per_strategy": best_discovered_per_strategy,
        "optimization_parameters": {
            "initial_policies": initial_policies,
            "total_budget": max_budget,
            "total_queries": max_queries,
            "n_seeds": n_seeds,
            "beta": beta,
            "candidates_evaluated_per_step": n_candidates_per_step,
            "bootstrap_replicates": 2000,
            "duplicate_tolerance": 1e-3,
        },
        "strategy_comparison": final_budget_entry["strategies"],
        "paired_comparisons": final_budget_entry["paired_comparisons"],
        "budget_sweep": budget_sweep_results,
    }

    with open(output_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    return benchmark_summary
