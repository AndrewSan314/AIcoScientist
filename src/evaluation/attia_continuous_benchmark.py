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
from src.optimization.acquisition import (
    compute_acquisition,
    compute_true_mc_nei,
    predict_latent_gp,
)
from src.optimization.adaptive_controller import AdaptiveBOController
from src.optimization.search_space import SearchSpace
from src.optimization.trust_region import TuRBOTrustRegion

logger = logging.getLogger(__name__)

ATTIA_SOURCE_COMMIT = "0068fd0136bcd65884f5cd94b2b967c1ba73a668"
SIMULATOR_VERSION = "1.0.0"


def fit_attia_continuous_gp(
    observed_records: list[dict[str, Any]],
    feature_cols: Sequence[str] = ("C1", "C2", "C3"),
    random_state: int = 42,
) -> tuple[GaussianProcessRegressor, StandardScaler, np.ndarray, np.ndarray]:
    """Fits GaussianProcessRegressor surrogate on observed continuous records.

    Returns:
        gp: Fitted GaussianProcessRegressor
        scaler: Fitted StandardScaler
        X_train_scaled: Scaled training features (N, d)
        y_train: Raw training targets (N,)
    """
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
        random_state=random_state,
    )
    gp.fit(X_train_scaled, y_train)
    return gp, scaler, X_train_scaled, y_train


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
        c1, c2, c3 = float(row["C1"]), float(row["C2"]), float(row["C3"])
        c4 = float(row["C4"])
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
    """Evaluator-only derivative-free global search to determine best-known continuous simulator reference."""
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

    # Save artifacts
    with open(ref_path, "w", encoding="utf-8") as fr:
        json.dump(best_record, fr, indent=2)

    manifest_content = {
        "n_sobol_samples": n_sobol_samples,
        "n_local_starts": n_local_starts,
        "evaluator_seed": seed,
        "search_method": "derivative_free_sobol_plus_nelder_mead",
        "best_known_latent_lifetime": best_record["best_known_latent_lifetime"],
        "candidate_id": best_record["candidate_id"],
    }
    with open(manifest_path, "w", encoding="utf-8") as fm:
        json.dump(manifest_content, fm, indent=2)

    return best_record


def run_single_attia_continuous_trajectory(
    search_space: SearchSpace,
    discrete_pool: pd.DataFrame,
    init_indices: list[int],
    total_queries: int,
    strategy: str = "gp_ucb",
    optimizer_seed: int = 42,
    beta: float = 1.0,
    duplicate_tol: float = 1e-3,
    n_candidates_per_step: int = 5000,
    refine_continuous: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Runs a single continuous BO trajectory with zero latent truth leakage.

    Surrogate GP is fit strictly on free continuous variables ["C1", "C2", "C3"].
    """
    feature_cols = ["C1", "C2", "C3"]  # Free design variables only

    controller: AdaptiveBOController | None = None
    if strategy == "adaptive":
        controller = AdaptiveBOController(
            base_beta=beta,
            base_xi=0.01,
            stagnation_threshold=3,
            ei_high_threshold=5.0,
        )

    # Warmup observations from selected discrete initial indices
    observed_records: list[dict[str, Any]] = []
    for idx in init_indices:
        r = discrete_pool.iloc[idx]
        c1, c2, c3 = float(r["C1"]), float(r["C2"]), float(r["C3"])
        c4 = float(compute_expected_c4(c1, c2, c3))
        p_id = str(r["policy_id"])
        sim_seed = generate_attia_simulator_seed(benchmark_seed=optimizer_seed, policy_id=p_id)
        sim_life = float(simulate_attia_policy(c1, c2, c3, mode="hi", variance=True, seed=sim_seed))

        observed_records.append(
            {
                "query_id": f"WARMUP_{p_id}",
                "candidate_id": p_id,
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
        turbo = TuRBOTrustRegion(search_space=search_space, init_length=0.8, global_escape_frequency=6)
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
            "trust_region_length": float(turbo.state.length) if turbo and turbo.state else None,
            "trust_region_radius": float(turbo.state.length) if turbo and turbo.state else None,
            "success_counter": int(turbo.state.success_counter) if turbo and turbo.state else 0,
            "failure_counter": int(turbo.state.failure_counter) if turbo and turbo.state else 0,
            "expanded": False,
            "contracted": False,
            "restarted": False,
            "global_escape": False,
            "proposal_predicted_mean": None,
            "proposal_predicted_std": None,
            "post_observation_candidate_mean": None,
            "post_observation_candidate_std": None,
            "post_observation_incumbent_mean": None,
            "post_observation_incumbent_std": None,
            "post_observation_candidate_incumbent_covariance": None,
            "posterior_candidate_mean": None,
            "posterior_candidate_std": None,
            "posterior_incumbent_mean": None,
            "posterior_incumbent_std": None,
            "success_probability": None,
            "restart_reason": None,
            "restart_candidate_id": None,
        }
    )

    # Closed-loop Continuous BO iterations
    for step in range(1, total_queries + 1):
        step_seed = optimizer_seed * 1000 + step * 100 + 7
        is_global_escape = False

        if strategy == "turbo_nei" and turbo is not None:
            if turbo.should_global_escape(step):
                is_global_escape = True

        # 1. Sample candidate batch (within trust region if turbo_nei and not global escape, else globally)
        if strategy == "turbo_nei" and turbo is not None and not is_global_escape:
            cand_batch = turbo.sample_candidates(
                n=n_candidates_per_step,
                seed=step_seed,
            )
        else:
            cand_batch = search_space.sample_feasible(
                n=n_candidates_per_step,
                seed=step_seed,
            )

        observed_df = pd.DataFrame(observed_records)
        novelty_vs_observed = search_space.check_novelty(
            cand_batch,
            reference_points=observed_df,
            feature_cols=feature_cols,
            tol=duplicate_tol,
        )

        # Resample candidate pool once if all candidates in initial pool are duplicates
        if not np.any(novelty_vs_observed["min_distance"].to_numpy() >= duplicate_tol):
            if strategy == "turbo_nei" and turbo is not None and not is_global_escape:
                cand_batch = turbo.sample_candidates(
                    n=n_candidates_per_step,
                    seed=step_seed + 1,
                )
            else:
                cand_batch = search_space.sample_feasible(
                    n=n_candidates_per_step,
                    seed=step_seed + 1,
                )
            novelty_vs_observed = search_space.check_novelty(
                cand_batch,
                reference_points=observed_df,
                feature_cols=feature_cols,
                tol=duplicate_tol,
            )
            if not np.any(novelty_vs_observed["min_distance"].to_numpy() >= duplicate_tol):
                raise RuntimeError(
                    f"No novel candidates found in search space within duplicate tolerance {duplicate_tol} after resampling at step {step}."
                )

        # 2. Fit GP surrogate on observed simulated observations using ONLY free variables C1, C2, C3
        gp, scaler, X_train_scaled, y_train = fit_attia_continuous_gp(
            observed_records, feature_cols=feature_cols, random_state=optimizer_seed + step
        )

        # Observed posterior latent means for denoised incumbent in NEI
        obs_posterior_means = predict_latent_gp(gp, X_train_scaled, return_std=False)

        # 3. Predict acquisition across feasible candidates using latent GP uncertainty
        X_cand = cand_batch[feature_cols].to_numpy(dtype=float)
        X_cand_scaled = scaler.transform(X_cand)
        pred_mean, pred_std = predict_latent_gp(gp, X_cand_scaled, return_std=True)

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
                gp=gp,
                X_observed_scaled=X_train_scaled,
                X_candidates_scaled=X_cand_scaled,
                seed=step_seed,
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

            # Local continuous refinement strictly on smooth acquisitions (NEI/TuRBO-NEI never refined)
            if refine_continuous and current_method in {"greedy", "gp_ucb", "expected_improvement", "denoised_expected_improvement"}:
                init_c1 = float(best_cand_dict["C1"])
                init_c2 = float(best_cand_dict["C2"])
                init_c3 = float(best_cand_dict["C3"])

                def _obj(x: np.ndarray) -> float:
                    c1_v, c2_v, c3_v = float(x[0]), float(x[1]), float(x[2])
                    c4_v = float(compute_expected_c4(c1_v, c2_v, c3_v))
                    cand_t = {"C1": c1_v, "C2": c2_v, "C3": c3_v, "C4": c4_v}
                    if not search_space.is_feasible(cand_t):
                        return 1e6

                    x_feat = np.array([[c1_v, c2_v, c3_v]])
                    x_sc = scaler.transform(x_feat)
                    m, s = predict_latent_gp(gp, x_sc, return_std=True)
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

        # Predict mean & std at selected point using 3D free variables and latent GP
        sc_pt = scaler.transform([[c1, c2, c3]])
        p_mean, p_std = predict_latent_gp(gp, sc_pt, return_std=True)

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

        tr_update: dict[str, Any] = {}
        if strategy == "turbo_nei" and turbo is not None:
            # 1. Refit GP strictly on D_{t+1} BEFORE evaluating posterior evidence and TuRBO updates
            gp_post, scaler_post, X_all_sc, y_all = fit_attia_continuous_gp(
                observed_records, feature_cols=feature_cols, random_state=step_seed + 1
            )

            # 2. Predict latent means, variances, and candidate-incumbent covariance using refitted GP
            mu_all, cov_all = predict_latent_gp(gp_post, X_all_sc, return_cov=True)
            mu_all = np.asarray(mu_all, dtype=float)
            cov_all = np.asarray(cov_all, dtype=float)

            cand_idx = len(mu_all) - 1
            p_cand_m = float(mu_all[cand_idx])
            p_cand_v = float(cov_all[cand_idx, cand_idx])
            p_cand_s = float(np.sqrt(max(p_cand_v, 1e-12)))

            if len(mu_all) > 1:
                prev_m = mu_all[:-1]
                inc_i = int(np.argmax(prev_m))
                p_inc_m = float(prev_m[inc_i])
                p_inc_v = float(cov_all[inc_i, inc_i])
                p_cand_inc_cov = float(cov_all[cand_idx, inc_i])
            else:
                p_inc_m = p_cand_m
                p_inc_v = p_cand_v
                p_cand_inc_cov = p_cand_v
            p_inc_s = float(np.sqrt(max(p_inc_v, 1e-12)))

            # 3. Global fallback candidate in case of restart scored via True NEI on refitted GP
            can_restart = (
                turbo.state is not None
                and (
                    (turbo.state.failure_counter + 1 >= turbo.state.failure_tolerance and (turbo.state.length / 2.0) < turbo.state.min_length)
                    or (turbo.state.length < turbo.state.min_length)
                )
            )
            fallback_center = None
            fallback_cid = None
            if can_restart:
                g_pool = search_space.sample_feasible(n=256, seed=optimizer_seed * 1000 + step * 79 + 1)
                nov_g = search_space.check_novelty(
                    g_pool, reference_points=pd.DataFrame(observed_records), feature_cols=feature_cols, tol=duplicate_tol
                )
                v_idx = np.where(nov_g["min_distance"].to_numpy() >= duplicate_tol)[0]
                if len(v_idx) == 0:
                    g_pool = search_space.sample_feasible(n=256, seed=optimizer_seed * 1000 + step * 79 + 101)
                    nov_g = search_space.check_novelty(
                        g_pool, reference_points=pd.DataFrame(observed_records), feature_cols=feature_cols, tol=duplicate_tol
                    )
                    v_idx = np.where(nov_g["min_distance"].to_numpy() >= duplicate_tol)[0]

                if len(v_idx) > 0:
                    valid_g_pool = g_pool.iloc[v_idx].reset_index(drop=True)
                    X_g = valid_g_pool[feature_cols].to_numpy(dtype=float)
                    X_g_sc = scaler_post.transform(X_g)
                    g_scores = compute_true_mc_nei(
                        gp_post,
                        X_observed_scaled=X_all_sc,
                        X_candidates_scaled=X_g_sc,
                        n_fantasies=64,
                        seed=step_seed + 2,
                    )
                    best_g_idx = int(np.argmax(g_scores))
                    b_row = valid_g_pool.iloc[best_g_idx].to_dict()
                else:
                    b_row = g_pool.iloc[0].to_dict()

                fallback_center = {k: float(b_row[k]) for k in feature_cols}
                fallback_cid = generate_continuous_candidate_id(
                    float(b_row["C1"]), float(b_row["C2"]), float(b_row["C3"]), float(b_row["C4"])
                )

            tr_update = turbo.update(
                observed_candidate={"C1": c1, "C2": c2, "C3": c3},
                observed_value=sim_life,
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
                global_escape=is_global_escape,
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
            "trust_region_length": float(turbo.state.length) if turbo and turbo.state else None,
            "trust_region_radius": float(turbo.state.length) if turbo and turbo.state else None,
            "success_counter": int(turbo.state.success_counter) if turbo and turbo.state else 0,
            "failure_counter": int(turbo.state.failure_counter) if turbo and turbo.state else 0,
            "expanded": bool(tr_update.get("expanded", False)),
            "contracted": bool(tr_update.get("contracted", False)),
            "restarted": bool(tr_update.get("restarted", False)),
            "global_escape": is_global_escape,
            # Explicit proposal-time surrogate prediction
            "proposal_predicted_mean": float(p_mean[0]),
            "proposal_predicted_std": float(p_std[0]),
            # Explicit post-observation refitted surrogate statistics
            "post_observation_candidate_mean": tr_update.get("posterior_candidate_mean", float(p_mean[0])),
            "post_observation_candidate_std": tr_update.get("posterior_candidate_std", float(p_std[0])),
            "post_observation_incumbent_mean": tr_update.get("posterior_incumbent_mean"),
            "post_observation_incumbent_std": tr_update.get("posterior_incumbent_std"),
            "post_observation_candidate_incumbent_covariance": tr_update.get("posterior_candidate_incumbent_covariance"),
            "success_probability": tr_update.get("success_probability"),
            # Backward compatibility aliases
            "posterior_candidate_mean": tr_update.get("posterior_candidate_mean", float(p_mean[0])),
            "posterior_candidate_std": tr_update.get("posterior_candidate_std", float(p_std[0])),
            "posterior_incumbent_mean": tr_update.get("posterior_incumbent_mean"),
            "posterior_incumbent_std": tr_update.get("posterior_incumbent_std"),
            "restart_reason": tr_update.get("restart_reason"),
            "restart_candidate_id": tr_update.get("restart_candidate_id"),
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
    If true_lifetime exceeds continuous_ref_lifetime, continuous_simple_regret becomes None / invalid.
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
            evaluated_row["continuous_simple_regret_valid"] = True
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

            if reference_underestimated:
                evaluated_row["continuous_simple_regret"] = None
                evaluated_row["continuous_simple_regret_valid"] = False
            else:
                evaluated_row["continuous_simple_regret"] = max(0.0, continuous_ref_lifetime - current_best_true)
                evaluated_row["continuous_simple_regret_valid"] = True

            evaluated_row["gap_to_discrete_grid_optimum"] = max(0.0, discrete_grid_optimum_lifetime - current_best_true)
            evaluated_row["improvement_over_discrete_grid"] = max(0.0, current_best_true - discrete_grid_optimum_lifetime)
            evaluated_row["beats_discrete_grid"] = bool(current_best_true > discrete_grid_optimum_lifetime)

        evaluated_history.append(evaluated_row)

    return evaluated_history, reference_underestimated


def _run_single_seed_strat(
    seed_idx: int,
    strat: str,
    search_space: SearchSpace,
    discrete_pool: pd.DataFrame,
    init_indices: list[int],
    max_queries: int,
    beta: float,
    n_candidates_per_step: int,
    cont_ref_opt_true: float,
    discrete_opt_true: float,
) -> tuple[int, str, list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Worker task executing a single strategy on a single seed."""
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
    eval_hist, ref_under = evaluate_continuous_trajectory(
        raw_trajectory=raw_hist,
        init_indices=init_indices,
        discrete_pool=discrete_pool,
        continuous_ref_lifetime=cont_ref_opt_true,
        discrete_grid_optimum_lifetime=discrete_opt_true,
    )
    return seed_idx, strat, eval_hist, decision_records, ref_under


def run_attia_continuous_benchmark(
    adapter: AttiaAdapter | None = None,
    budgets: Sequence[int] = (10, 15, 20, 30),
    initial_policies: int = 5,
    n_seeds: int = 30,
    beta: float = 1.0,
    n_candidates_per_step: int = 5000,
    output_dir: Path | str | None = None,
    n_jobs: int = -1,
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
        "Running hardened %d-strategy Continuous BO benchmark across %d seeds and budgets %s (n_jobs=%s)...",
        len(strategies),
        n_seeds,
        budgets,
        n_jobs,
    )

    tasks = []
    for seed_idx in range(n_seeds):
        seed_rng = np.random.default_rng(seed_idx * 7919 + 42)
        init_indices = seed_rng.choice(len(discrete_pool), size=initial_policies, replace=False).tolist()
        full_evaluated_trajectories[seed_idx] = {}
        for strat in strategies:
            tasks.append((seed_idx, strat, init_indices))

    if n_jobs == 1:
        results = [
            _run_single_seed_strat(
                seed_idx=s_idx,
                strat=strat,
                search_space=search_space,
                discrete_pool=discrete_pool,
                init_indices=i_indices,
                max_queries=max_queries,
                beta=beta,
                n_candidates_per_step=n_candidates_per_step,
                cont_ref_opt_true=cont_ref_opt_true,
                discrete_opt_true=discrete_opt_true,
            )
            for s_idx, strat, i_indices in tasks
        ]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_run_single_seed_strat)(
                seed_idx=s_idx,
                strat=strat,
                search_space=search_space,
                discrete_pool=discrete_pool,
                init_indices=i_indices,
                max_queries=max_queries,
                beta=beta,
                n_candidates_per_step=n_candidates_per_step,
                cont_ref_opt_true=cont_ref_opt_true,
                discrete_opt_true=discrete_opt_true,
            )
            for s_idx, strat, i_indices in tasks
        )

    # Sort results to ensure deterministic ordering regardless of worker completion order
    strat_order = {s: i for i, s in enumerate(strategies)}
    results.sort(key=lambda item: (item[0], strat_order.get(item[1], 0)))

    for seed_idx, strat, eval_hist, decision_records, ref_under in results:
        if strat == "adaptive":
            all_adaptive_decision_records.extend(decision_records)
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
                            "candidate_id": row["candidate_id"],
                            "trust_region_center_C1": float(json.loads(row["trust_region_center"])["C1"]) if row.get("trust_region_center") else None,
                            "trust_region_center_C2": float(json.loads(row["trust_region_center"])["C2"]) if row.get("trust_region_center") else None,
                            "trust_region_center_C3": float(json.loads(row["trust_region_center"])["C3"]) if row.get("trust_region_center") else None,
                            "trust_region_length": row.get("trust_region_length"),
                            "proposal_predicted_mean": row.get("proposal_predicted_mean"),
                            "proposal_predicted_std": row.get("proposal_predicted_std"),
                            "post_observation_candidate_mean": row.get("post_observation_candidate_mean"),
                            "post_observation_candidate_std": row.get("post_observation_candidate_std"),
                            "post_observation_incumbent_mean": row.get("post_observation_incumbent_mean"),
                            "post_observation_incumbent_std": row.get("post_observation_incumbent_std"),
                            "post_observation_candidate_incumbent_covariance": row.get("post_observation_candidate_incumbent_covariance"),
                            "posterior_candidate_mean": row.get("posterior_candidate_mean"),
                            "posterior_candidate_std": row.get("posterior_candidate_std"),
                            "posterior_incumbent_mean": row.get("posterior_incumbent_mean"),
                            "posterior_incumbent_std": row.get("posterior_incumbent_std"),
                            "success_probability": row.get("success_probability"),
                            "success_counter": row.get("success_counter", 0),
                            "failure_counter": row.get("failure_counter", 0),
                            "expanded": bool(row.get("expanded", False)),
                            "contracted": bool(row.get("contracted", False)),
                            "restarted": bool(row.get("restarted", False)),
                            "restart_reason": row.get("restart_reason"),
                            "restart_candidate_id": row.get("restart_candidate_id"),
                            "global_escape": bool(row.get("global_escape", False)),
                            "acquisition_score": row.get("acquisition_score"),
                            "simulated_lifetime": row.get("simulated_lifetime"),
                            "best_observed_lifetime": row.get("best_observed_lifetime"),
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
        strat_cont_regrets: dict[str, list[float | None]] = {s: [] for s in strategies}
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
                cont_regret = final_row["continuous_simple_regret"]
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
            valid_cont_regs = [r for r in strat_cont_regrets[strat] if r is not None]
            cont_regs = np.array(valid_cont_regs, dtype=float) if len(valid_cont_regs) == len(strat_cont_regrets[strat]) else None

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
            reg_ci_low, reg_ci_high = compute_bootstrap_mean_ci(cont_regs, n_bootstraps=2000, seed=2042 + budget) if cont_regs is not None else (None, None)
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
                "continuous_regret_valid": (cont_regs is not None),
                "mean_continuous_simple_regret": float(np.mean(cont_regs)) if cont_regs is not None else None,
                "std_continuous_simple_regret": float(np.std(cont_regs, ddof=1)) if (cont_regs is not None and len(cont_regs) > 1) else None,
                "median_continuous_simple_regret": float(np.median(cont_regs)) if cont_regs is not None else None,
                "mean_continuous_simple_regret_95_ci": [float(reg_ci_low), float(reg_ci_high)] if (reg_ci_low is not None and reg_ci_high is not None) else None,
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
                seed_esc = [sum(1 for h in full_evaluated_trajectories[s_idx][strat] if 0 < h["step"] <= n_queries and h.get("global_escape", False)) for s_idx in range(n_seeds)]
                seed_rad = [next((float(h.get("trust_region_length")) for h in reversed(full_evaluated_trajectories[s_idx][strat]) if h["step"] <= n_queries and h.get("trust_region_length") is not None), 0.8) for s_idx in range(n_seeds)]

                strat_dict["mean_trust_region_expansions"] = float(np.mean(seed_exp))
                strat_dict["mean_trust_region_contractions"] = float(np.mean(seed_con))
                strat_dict["mean_trust_region_restarts"] = float(np.mean(seed_res))
                strat_dict["mean_global_escapes"] = float(np.mean(seed_esc))
                strat_dict["mean_final_trust_region_length"] = float(np.mean(seed_rad))
                strat_dict["fraction_with_restarts"] = float(np.mean([1 if r > 0 else 0 for r in seed_res]))
                strat_dict["fraction_with_expansions"] = float(np.mean([1 if e > 0 else 0 for e in seed_exp]))

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
            paired_comparisons[pair_name] = {
                "best_diff": {
                    "mean": float(np.mean(d_best)),
                    "median": float(np.median(d_best)),
                    "std": float(np.std(d_best, ddof=1)) if len(d_best) > 1 else 0.0,
                    "p_value": _safe_wilcoxon(d_best),
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

    # Compute Sample Efficiency to Target Thresholds across full maximum budget
    thresholds_spec = {
        "threshold_a_discrete_opt_1079": 1079.0,
        "threshold_b_95pct_ref_1070_65": 1070.65,
        "threshold_c_97_5pct_ref_1098_83": 1098.83,
    }
    sample_efficiency_table: dict[str, dict[str, Any]] = {}

    for t_key, t_val in thresholds_spec.items():
        sample_efficiency_table[t_key] = {"target_lifetime": t_val, "strategies": {}}
        for strat in strategies:
            reach_steps = []
            for seed_idx in range(n_seeds):
                hist = full_evaluated_trajectories[seed_idx][strat]
                step_hit = next((h["step"] for h in hist if h["step"] > 0 and float(h["reference_true_lifetime"]) >= t_val), None)
                if step_hit is not None:
                    reach_steps.append(step_hit)

            n_hit = len(reach_steps)
            sample_efficiency_table[t_key]["strategies"][strat] = {
                "fraction_reaching": float(n_hit / n_seeds) if n_seeds > 0 else 0.0,
                "pct_reaching": float((n_hit / n_seeds) * 100.0) if n_seeds > 0 else 0.0,
                "mean_queries_successful": float(np.mean(reach_steps)) if n_hit > 0 else None,
                "median_queries_successful": float(np.median(reach_steps)) if n_hit > 0 else None,
                "std_queries_successful": float(np.std(reach_steps, ddof=1)) if n_hit > 1 else 0.0,
                "censored_runs_count": int(n_seeds - n_hit),
            }

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
        "continuous_regret_valid": not any_ref_underestimated,
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
        "sample_efficiency_to_threshold": sample_efficiency_table,
        "strategy_comparison": final_budget_entry["strategies"],
        "paired_comparisons": final_budget_entry["paired_comparisons"],
        "budget_sweep": budget_sweep_results,
    }

    with open(output_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    from src.science.provenance import build_benchmark_run_manifest

    run_manifest = build_benchmark_run_manifest(
        dataset_name="attia_continuous",
        comparison_baseline_commit="53a1c7241222105cdede343d5a155fdd5a97ee78",
        simulator_version=SIMULATOR_VERSION,
        attia_source_commit=ATTIA_SOURCE_COMMIT,
        n_seeds=n_seeds,
        budgets=budgets,
        strategies=strategies,
        initial_policies=initial_policies,
        candidate_pool_size=len(discrete_pool),
        duplicate_tolerance=1e-3,
        n_jobs=n_jobs,
        reference_underestimated=any_ref_underestimated,
    )
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as fm:
        json.dump(run_manifest, fm, indent=2)

    return benchmark_summary
