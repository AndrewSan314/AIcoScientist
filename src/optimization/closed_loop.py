from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from src.optimization.acquisition import compute_acquisition
from src.optimization.search_space import SearchSpace
from src.optimization.trust_region import TrustRegionState, TuRBOTrustRegion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentProposal:
    """Human-interpretable proposal for the next closed-loop experiment."""
    candidate_id: str
    design_variables: dict[str, Any]
    predicted_performance: float
    prediction_uncertainty: float
    acquisition_score: float
    acquisition_method: str
    trust_region_center: dict[str, Any] | None
    trust_region_radius: float | None
    recommendation_reason: str
    reason_code: str
    distance_to_nearest_observed: float
    step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            **self.design_variables,
            "predicted_performance": float(self.predicted_performance),
            "prediction_uncertainty": float(self.prediction_uncertainty),
            "acquisition_score": float(self.acquisition_score),
            "acquisition_method": self.acquisition_method,
            "trust_region_center": self.trust_region_center,
            "trust_region_radius": float(self.trust_region_radius) if self.trust_region_radius is not None else None,
            "recommendation_reason": self.recommendation_reason,
            "reason_code": self.reason_code,
            "distance_to_nearest_observed": float(self.distance_to_nearest_observed),
            "step": int(self.step),
        }


@dataclass(frozen=True)
class ExperimentResult:
    """Feedback returned by an experimental measurement or simulator oracle."""
    candidate_id: str
    design_variables: dict[str, Any]
    target_value: float
    observations: dict[str, Any] = field(default_factory=dict)
    is_success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentOracle(ABC):
    """Abstract interface for experimental data generation (hardware or simulator)."""

    @abstractmethod
    def evaluate(self, proposal: ExperimentProposal) -> ExperimentResult:
        """Executes an experiment (or queries numerical oracle) for the proposed candidate."""
        raise NotImplementedError


@dataclass
class OptimizerState:
    """Encapsulates the complete optimizer state across closed-loop iterations."""
    observed_records: list[dict[str, Any]]
    feature_cols: list[str]
    target_col: str
    objective: str
    step: int
    current_best: float
    trust_region: TuRBOTrustRegion | None = None
    gp_model: GaussianProcessRegressor | None = None
    scaler: StandardScaler | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_records": self.observed_records,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "objective": self.objective,
            "step": int(self.step),
            "current_best": float(self.current_best),
            "trust_region_state": self.trust_region.state.to_dict() if self.trust_region and self.trust_region.state else None,
            "history": self.history,
        }


class ClosedLoopOptimizer:
    """Universal domain-agnostic closed-loop materials optimizer.

    Maintains strict optimizer/evaluator separation. Supports TuRBO trust region,
    True Noisy Expected Improvement (NEI), GP-UCB, Greedy, and standard EI.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        feature_cols: list[str],
        target_col: str,
        strategy: str = "turbo_nei",
        objective: str = "maximize",
        beta: float = 1.0,
        xi: float = 0.01,
        duplicate_tol: float = 1e-3,
        n_candidates_per_step: int = 5000,
        random_state: int = 42,
    ) -> None:
        self.search_space = search_space
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.strategy = strategy.lower().strip()
        self.objective = objective
        self.beta = beta
        self.xi = xi
        self.duplicate_tol = duplicate_tol
        self.n_candidates_per_step = n_candidates_per_step
        self.random_state = random_state

    def initialize(self, initial_observations: pd.DataFrame | list[dict[str, Any]]) -> OptimizerState:
        """Initializes optimizer state from historical or warmup experimental observations."""
        if isinstance(initial_observations, pd.DataFrame):
            records = initial_observations.to_dict(orient="records")
        else:
            records = [dict(r) for r in initial_observations]

        if not records:
            raise ValueError("At least one initial observation is required to initialize the optimizer.")

        targets = [float(r[self.target_col]) for r in records]
        current_best = float(np.max(targets) if self.objective == "maximize" else np.min(targets))

        # Best initial candidate
        best_idx = int(np.argmax(targets) if self.objective == "maximize" else np.argmin(targets))
        best_record = records[best_idx]

        tr: TuRBOTrustRegion | None = None
        if "turbo" in self.strategy:
            tr = TuRBOTrustRegion(search_space=self.search_space)
            tr.initialize(best_record, current_best)

        state = OptimizerState(
            observed_records=records,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            objective=self.objective,
            step=0,
            current_best=current_best,
            trust_region=tr,
            gp_model=None,
            scaler=None,
            history=[],
        )
        self._fit_surrogate(state)
        return state

    def _fit_surrogate(self, state: OptimizerState) -> None:
        """Fits Gaussian Process probabilistic surrogate on optimizer-visible observations."""
        X_train = np.array([[r[c] for c in self.feature_cols] for r in state.observed_records], dtype=float)
        y_train = np.array([float(r[self.target_col]) for r in state.observed_records], dtype=float)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        kernel = ConstantKernel(1.0, (1e-2, 1e4)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
            noise_level=1.0, noise_level_bounds=(1e-5, 1e2)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=self.random_state + state.step,
        )
        gp.fit(X_scaled, y_train)

        state.scaler = scaler
        state.gp_model = gp

    def propose(self, state: OptimizerState) -> ExperimentProposal:
        """Proposes the next experiment using active strategy, surrogate model, and search space."""
        state.step += 1
        step_seed = self.random_state * 1000 + state.step * 100 + 7

        is_global_escape = False
        if state.trust_region is not None and "turbo" in self.strategy:
            if state.trust_region.should_global_escape(state.step):
                is_global_escape = True

        # 1. Generate candidate pool
        if state.trust_region is not None and "turbo" in self.strategy and not is_global_escape:
            cand_batch = state.trust_region.sample_candidates(
                n=self.n_candidates_per_step,
                seed=step_seed,
            )
            tr_center = state.trust_region.state.center if state.trust_region.state else None
            tr_radius = state.trust_region.state.length if state.trust_region.state else None
        else:
            cand_batch = self.search_space.sample_feasible(
                n=self.n_candidates_per_step,
                seed=step_seed,
            )
            tr_center = state.trust_region.state.center if (state.trust_region and state.trust_region.state) else None
            tr_radius = state.trust_region.state.length if (state.trust_region and state.trust_region.state) else None

        # 2. Fit/ensure surrogate
        if state.gp_model is None or state.scaler is None:
            self._fit_surrogate(state)

        # 3. Predict surrogate mean & epistemic uncertainty
        X_cand = cand_batch[self.feature_cols].to_numpy(dtype=float)
        X_cand_scaled = state.scaler.transform(X_cand)
        pred_mean, pred_std = state.gp_model.predict(X_cand_scaled, return_std=True)

        # Denoised posterior means and design at observed points
        X_obs = np.array([[r[c] for c in self.feature_cols] for r in state.observed_records], dtype=float)
        X_obs_scaled = state.scaler.transform(X_obs)
        obs_posterior_means = state.gp_model.predict(X_obs_scaled)

        # 4. Check duplicate / novelty vs observed
        observed_df = pd.DataFrame(state.observed_records)
        novelty_vs_observed = self.search_space.check_novelty(
            cand_batch,
            reference_points=observed_df,
            feature_cols=self.feature_cols,
            tol=self.duplicate_tol,
        )

        acq_method = "nei" if "nei" in self.strategy else self.strategy

        if self.strategy == "random":
            non_dup_idx = np.where(novelty_vs_observed["min_distance"].to_numpy() >= self.duplicate_tol)[0]
            rng_step = np.random.default_rng(step_seed)
            if len(non_dup_idx) > 0:
                selected_idx = int(rng_step.choice(non_dup_idx))
            else:
                selected_idx = int(rng_step.integers(0, len(cand_batch)))

            selected_cand = cand_batch.iloc[selected_idx].to_dict()
            acq_score = 0.0
            reason_code = "SPACE_FILLING_RANDOM"
            reason = "Recommended by exploratory uniform space-filling quasi-random sampling."
        else:
            scores = compute_acquisition(
                method=acq_method,
                mean=pred_mean,
                std=pred_std,
                best_observed=state.current_best,
                beta=self.beta,
                xi=self.xi,
                objective=self.objective,
                observed_posterior_means=obs_posterior_means,
                gp=state.gp_model,
                X_observed_scaled=X_obs_scaled,
                X_candidates_scaled=X_cand_scaled,
                seed=step_seed,
            )

            sorted_indices = np.argsort(scores)[::-1]
            selected_idx = None

            for idx in sorted_indices:
                if novelty_vs_observed["min_distance"].iloc[idx] >= self.duplicate_tol:
                    selected_idx = idx
                    break

            if selected_idx is None:
                selected_idx = sorted_indices[0]

            selected_cand = cand_batch.iloc[selected_idx].to_dict()
            acq_score = float(scores[selected_idx])

            m_val = float(pred_mean[selected_idx])
            s_val = float(pred_std[selected_idx])

            if is_global_escape:
                reason_code = "GLOBAL_ESCAPE"
                reason = (
                    f"Recommended via periodic global escape exploration outside local trust region "
                    f"(score={acq_score:.3f}, pred_mean={m_val:.2f}, std={s_val:.2f})."
                )
            elif "turbo" in self.strategy:
                reason_code = "TURBO_EXPLOITATION_NEI"
                reason = (
                    f"Recommended because candidate lies inside the active trust region "
                    f"(radius={tr_radius:.3f}) with high Noisy Expected Improvement (score={acq_score:.3f}, "
                    f"pred_mean={m_val:.2f}, std={s_val:.2f})."
                )
            elif "nei" in self.strategy:
                reason_code = "GLOBAL_EXPLORATION_NEI"
                reason = (
                    f"Recommended for maximal Joint-Posterior Noisy Expected Improvement (score={acq_score:.3f}, "
                    f"pred_mean={m_val:.2f}, std={s_val:.2f}) under noisy experimental observations."
                )
            elif self.strategy in {"gp_ucb", "ucb"}:
                reason_code = "UCB_HIGH_UNCERTAINTY"
                reason = (
                    f"Recommended for Upper Confidence Bound exploration-exploitation balance "
                    f"(score={acq_score:.3f}, pred_mean={m_val:.2f}, std={s_val:.2f}, beta={self.beta:.2f})."
                )
            elif self.strategy in {"expected_improvement", "ei"}:
                reason_code = "EXPECTED_IMPROVEMENT"
                reason = (
                    f"Recommended for Expected Improvement (score={acq_score:.3f}, pred_mean={m_val:.2f}, std={s_val:.2f})."
                )
            else:
                reason_code = "GREEDY_POSTERIOR_MEAN"
                reason = f"Recommended by {self.strategy} (score={acq_score:.3f}, pred_mean={m_val:.2f}, std={s_val:.2f})."

        cand_id = selected_cand.get("candidate_id") or selected_cand.get("policy_id")
        if not cand_id:
            coords = "_".join(f"{k}={float(selected_cand[k]):.4f}" for k in sorted(self.feature_cols) if k in selected_cand)
            cand_id = f"EXP_{hash(coords) % 1000000:06d}"

        design_vars = {k: selected_cand[k] for k in self.feature_cols if k in selected_cand}
        min_dist = float(novelty_vs_observed["min_distance"].iloc[selected_idx])

        proposal = ExperimentProposal(
            candidate_id=str(cand_id),
            design_variables=design_vars,
            predicted_performance=float(pred_mean[selected_idx]),
            prediction_uncertainty=float(pred_std[selected_idx]),
            acquisition_score=acq_score,
            acquisition_method=self.strategy,
            trust_region_center=tr_center,
            trust_region_radius=tr_radius,
            recommendation_reason=reason,
            reason_code=reason_code,
            distance_to_nearest_observed=min_dist,
            step=state.step,
        )
        return proposal

    def observe(
        self,
        state: OptimizerState,
        proposal: ExperimentProposal,
        result: ExperimentResult,
    ) -> OptimizerState:
        """Incorporates experimental result and updates surrogate model and trust region."""
        val = float(result.target_value)

        # Update current best
        if state.objective == "maximize":
            is_improvement = val > state.current_best + 1e-4
            state.current_best = max(state.current_best, val)
        else:
            is_improvement = val < state.current_best - 1e-4
            state.current_best = min(state.current_best, val)

        # Record observed row
        new_row = {
            "step": state.step,
            "candidate_id": proposal.candidate_id,
            **proposal.design_variables,
            **result.observations,
            self.target_col: val,
        }
        state.observed_records.append(new_row)

        # Update trust region if applicable
        tr_info = None
        if state.trust_region is not None:
            tr_info = state.trust_region.update(
                observed_candidate=proposal.design_variables,
                observed_value=val,
                objective=state.objective,
            )

        # Refit GP surrogate
        self._fit_surrogate(state)

        # Record step history
        step_record = {
            "step": state.step,
            "candidate_id": proposal.candidate_id,
            "target_value": val,
            "best_observed": state.current_best,
            "is_improvement": is_improvement,
            "reason_code": proposal.reason_code,
            "distance_to_nearest_observed": proposal.distance_to_nearest_observed,
            "trust_region_radius": tr_info.get("length", tr_info.get("radius")) if tr_info else None,
            "restarted": tr_info.get("restarted") if tr_info else False,
            "expanded": tr_info.get("expanded") if tr_info else False,
            "contracted": tr_info.get("contracted") if tr_info else False,
        }
        state.history.append(step_record)
        return state

    def save_state(self, state: OptimizerState, filepath: Path | str) -> None:
        """Serializes optimizer state to disk for persistent resumption."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = state.to_dict()
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_state(self, filepath: Path | str) -> OptimizerState:
        """Loads and reconstructs optimizer state from disk."""
        p = Path(filepath)
        data = json.loads(p.read_text(encoding="utf-8"))

        tr: TuRBOTrustRegion | None = None
        if data.get("trust_region_state") is not None and "turbo" in self.strategy:
            tr = TuRBOTrustRegion(search_space=self.search_space)
            tr.state = TrustRegionState.from_dict(data["trust_region_state"])

        state = OptimizerState(
            observed_records=data["observed_records"],
            feature_cols=data["feature_cols"],
            target_col=data["target_col"],
            objective=data["objective"],
            step=int(data["step"]),
            current_best=float(data["current_best"]),
            trust_region=tr,
            gp_model=None,
            scaler=None,
            history=data.get("history", []),
        )
        self._fit_surrogate(state)
        return state
