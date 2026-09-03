from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES
from src.science.actions import ActionType, normalize_action_type
from src.science.domain import HypothesisProvider, HypothesisTrainingContext
from src.science.hypothesis_models import (
    PredictiveDistribution,
    ScientificHypothesisModel,
)

logger = logging.getLogger(__name__)

# Broad, non-discriminating prior distribution before minimum experimental data (N < 3)
DEFAULT_BROAD_CAPACITY_MEAN: float = 0.50
DEFAULT_BROAD_CAPACITY_VAR: float = 0.25
VARIANCE_FLOOR: float = 0.005


class GlobalSmoothDescriptorHypothesis:
    """Hypothesis H1: Normalized discharge capacity varies smoothly across continuous 11D solvent descriptor space.

    Predictive model: Gaussian Process Regressor with stationary RBF kernel and WhiteKernel noise floor.
    """

    def __init__(self, hypothesis_id: str = "global_smooth_descriptor") -> None:
        self._id = hypothesis_id
        self._title = "Global Smooth Descriptor Hypothesis"
        self._statement = "Normalized discharge capacity C_norm_20 varies smoothly across continuous 11D solvent descriptor space."
        self._assumptions = [
            "Continuous ECFP PCA coordinates and molecular weight define a smooth metric space for capacity.",
            "Stationary RBF covariance captures global chemical similarity.",
            "Residual variance is homogeneous Gaussian observation noise.",
        ]
        self._gp: GaussianProcessRegressor | None = None
        self._scaler: StandardScaler | None = None
        self._is_fitted = False
        self._sample_count = 0

    @property
    def hypothesis_id(self) -> str:
        return self._id

    @property
    def title(self) -> str:
        return self._title

    @property
    def statement(self) -> str:
        return self._statement

    @property
    def assumptions(self) -> list[str]:
        return list(self._assumptions)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def supports_action(self, action_type: ActionType) -> bool:
        return normalize_action_type(action_type) == "CAPACITY_TEST"

    def fit_context(self, context: HypothesisTrainingContext) -> GlobalSmoothDescriptorHypothesis:
        cap_obs = context.observations_by_modality.get("CAPACITY_TEST", {})
        valid_pairs = [
            (context.candidate_features_by_id[cid], float(val))
            for cid, val in cap_obs.items()
            if cid in context.candidate_features_by_id and val is not None
        ]

        if len(valid_pairs) < 3:
            self._gp = None
            self._scaler = None
            self._is_fitted = False
            self._sample_count = len(valid_pairs)
            return self

        X = np.array([p[0] for p in valid_pairs], dtype=np.float64)
        y = np.array([p[1] for p in valid_pairs], dtype=np.float64)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kernel = ConstantKernel(1.0, (0.01, 10.0)) * RBF(1.0, (0.01, 1000.0)) + WhiteKernel(0.05, (1e-4, 1.0))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42)
        gp.fit(X_scaled, y)

        self._gp = gp
        self._scaler = scaler
        self._is_fitted = True
        self._sample_count = len(valid_pairs)
        return self

    def fit(
        self,
        composition_by_id: Mapping[str, np.ndarray],
        property_by_id: Mapping[str, float] | None = None,
        observed_property_ids: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if property_by_id is None:
            self._gp = None
            self._is_fitted = False
            return
        ctx = HypothesisTrainingContext(
            candidate_features_by_id=composition_by_id,
            observations_by_modality={"CAPACITY_TEST": property_by_id},
        )
        self.fit_context(ctx)

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray,
        **kwargs: Any,
    ) -> PredictiveDistribution:
        if not self._is_fitted or self._gp is None or self._scaler is None:
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=normalize_action_type(action_type),
                mean=np.array([DEFAULT_BROAD_CAPACITY_MEAN]),
                variance=np.array([DEFAULT_BROAD_CAPACITY_VAR]),
                metadata={"prior_kind": "broad_uninformative", "n_samples": self._sample_count},
            )

        X = np.atleast_2d(np.asarray(composition, dtype=np.float64))
        X_scaled = self._scaler.transform(X)
        pred_mean, pred_std = self._gp.predict(X_scaled, return_std=True)
        var = float(np.maximum(pred_std[0] ** 2, VARIANCE_FLOOR))

        return PredictiveDistribution(
            hypothesis_id=self._id,
            candidate_id=candidate_id,
            action_type=normalize_action_type(action_type),
            mean=np.array([float(pred_mean[0])]),
            variance=np.array([var]),
            metadata={"model": "GaussianProcessRegressor", "n_samples": self._sample_count},
        )

    def log_predictive_density(
        self,
        observation: np.ndarray | float,
        prediction: PredictiveDistribution,
    ) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(
        self,
        candidate_id: str | None = None,
        action_type: ActionType | None = None,
        composition: np.ndarray | None = None,
        **kwargs: Any,
    ) -> str:
        return "Measured 20th-cycle capacity deviates from global smooth descriptor interpolation across solvent feature space."


class RegularizedAdditiveDescriptorHypothesis:
    """Hypothesis H2: Capacity is primarily explained by a low-complexity regularized additive combination of descriptors.

    Predictive model: Bayesian Ridge regression with analytical posterior predictive mean and variance.
    """

    def __init__(self, hypothesis_id: str = "regularized_additive_descriptor") -> None:
        self._id = hypothesis_id
        self._title = "Regularized Additive Descriptor Hypothesis"
        self._statement = "Capacity is governed by low-complexity regularized additive descriptor linear combinations rather than localized non-linear regimes."
        self._assumptions = [
            "First-order linear contributions of solvent molecular weight and PCA components dominate capacity.",
            "Higher-order interactions are suppressed by L2 shrinkage regularization.",
            "Observation error is homoscedastic Gaussian noise.",
        ]
        self._model: BayesianRidge | None = None
        self._scaler: StandardScaler | None = None
        self._is_fitted = False
        self._sample_count = 0

    @property
    def hypothesis_id(self) -> str:
        return self._id

    @property
    def title(self) -> str:
        return self._title

    @property
    def statement(self) -> str:
        return self._statement

    @property
    def assumptions(self) -> list[str]:
        return list(self._assumptions)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def supports_action(self, action_type: ActionType) -> bool:
        return normalize_action_type(action_type) == "CAPACITY_TEST"

    def fit_context(self, context: HypothesisTrainingContext) -> RegularizedAdditiveDescriptorHypothesis:
        cap_obs = context.observations_by_modality.get("CAPACITY_TEST", {})
        valid_pairs = [
            (context.candidate_features_by_id[cid], float(val))
            for cid, val in cap_obs.items()
            if cid in context.candidate_features_by_id and val is not None
        ]

        if len(valid_pairs) < 3:
            self._model = None
            self._scaler = None
            self._is_fitted = False
            self._sample_count = len(valid_pairs)
            return self

        X = np.array([p[0] for p in valid_pairs], dtype=np.float64)
        y = np.array([p[1] for p in valid_pairs], dtype=np.float64)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = BayesianRidge()
        model.fit(X_scaled, y)

        self._model = model
        self._scaler = scaler
        self._is_fitted = True
        self._sample_count = len(valid_pairs)
        return self

    def fit(
        self,
        composition_by_id: Mapping[str, np.ndarray],
        property_by_id: Mapping[str, float] | None = None,
        observed_property_ids: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if property_by_id is None:
            self._model = None
            self._is_fitted = False
            return
        ctx = HypothesisTrainingContext(
            candidate_features_by_id=composition_by_id,
            observations_by_modality={"CAPACITY_TEST": property_by_id},
        )
        self.fit_context(ctx)

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray,
        **kwargs: Any,
    ) -> PredictiveDistribution:
        if not self._is_fitted or self._model is None or self._scaler is None:
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=normalize_action_type(action_type),
                mean=np.array([DEFAULT_BROAD_CAPACITY_MEAN]),
                variance=np.array([DEFAULT_BROAD_CAPACITY_VAR]),
                metadata={"prior_kind": "broad_uninformative", "n_samples": self._sample_count},
            )

        X = np.atleast_2d(np.asarray(composition, dtype=np.float64))
        X_scaled = self._scaler.transform(X)
        pred_mean, pred_std = self._model.predict(X_scaled, return_std=True)
        var_floor = float(kwargs.get("variance_floor", VARIANCE_FLOOR))
        var = float(np.maximum(pred_std[0] ** 2, var_floor))

        return PredictiveDistribution(
            hypothesis_id=self._id,
            candidate_id=candidate_id,
            action_type=normalize_action_type(action_type),
            mean=np.array([float(pred_mean[0])]),
            variance=np.array([var]),
            metadata={"model": "BayesianRidge", "n_samples": self._sample_count},
        )

    def log_predictive_density(
        self,
        observation: np.ndarray | float,
        prediction: PredictiveDistribution,
    ) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(
        self,
        candidate_id: str | None = None,
        action_type: ActionType | None = None,
        composition: np.ndarray | None = None,
        **kwargs: Any,
    ) -> str:
        return "Measured 20th-cycle capacity departs from regularized additive linear descriptor model due to nonlinear chemical interactions."


# Backward compatibility alias
SparseAdditiveDescriptorHypothesis = RegularizedAdditiveDescriptorHypothesis


class LocalChemicalRegimeHypothesis:
    """Hypothesis H3: Solvent performance is governed by local, non-linear chemical regimes.

    Predictive model: Random Forest ensemble predictive distribution (mean across trees,
    epistemic variance from tree disagreement + residual floor).
    """

    def __init__(self, hypothesis_id: str = "local_chemical_regime") -> None:
        self._id = hypothesis_id
        self._title = "Local Chemical Regime Hypothesis"
        self._statement = "Capacity is governed by discrete, localized chemical regimes where global smoothness breaks down."
        self._assumptions = [
            "Chemical compatibility partitions into discrete, non-linear functional regimes.",
            "Decision tree ensembles approximate piecewise-constant local regimes.",
            "Tree disagreement quantifies epistemic regime uncertainty.",
        ]
        self._rf: RandomForestRegressor | None = None
        self._is_fitted = False
        self._sample_count = 0
        self._residual_var = 0.05

    @property
    def hypothesis_id(self) -> str:
        return self._id

    @property
    def title(self) -> str:
        return self._title

    @property
    def statement(self) -> str:
        return self._statement

    @property
    def assumptions(self) -> list[str]:
        return list(self._assumptions)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def supports_action(self, action_type: ActionType) -> bool:
        return normalize_action_type(action_type) == "CAPACITY_TEST"

    def fit_context(self, context: HypothesisTrainingContext) -> LocalChemicalRegimeHypothesis:
        cap_obs = context.observations_by_modality.get("CAPACITY_TEST", {})
        valid_pairs = [
            (context.candidate_features_by_id[cid], float(val))
            for cid, val in cap_obs.items()
            if cid in context.candidate_features_by_id and val is not None
        ]

        if len(valid_pairs) < 3:
            self._rf = None
            self._is_fitted = False
            self._sample_count = len(valid_pairs)
            return self

        X = np.array([p[0] for p in valid_pairs], dtype=np.float64)
        y = np.array([p[1] for p in valid_pairs], dtype=np.float64)

        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X, y)

        preds = rf.predict(X)
        res_var = float(np.var(y - preds)) if len(y) > 1 else 0.05

        self._rf = rf
        self._residual_var = max(res_var, VARIANCE_FLOOR)
        self._is_fitted = True
        self._sample_count = len(valid_pairs)
        return self

    def fit(
        self,
        composition_by_id: Mapping[str, np.ndarray],
        property_by_id: Mapping[str, float] | None = None,
        observed_property_ids: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if property_by_id is None:
            self._rf = None
            self._is_fitted = False
            return
        ctx = HypothesisTrainingContext(
            candidate_features_by_id=composition_by_id,
            observations_by_modality={"CAPACITY_TEST": property_by_id},
        )
        self.fit_context(ctx)

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray,
        **kwargs: Any,
    ) -> PredictiveDistribution:
        if not self._is_fitted or self._rf is None:
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=normalize_action_type(action_type),
                mean=np.array([DEFAULT_BROAD_CAPACITY_MEAN]),
                variance=np.array([DEFAULT_BROAD_CAPACITY_VAR]),
                metadata={"prior_kind": "broad_uninformative", "n_samples": self._sample_count},
            )

        X = np.atleast_2d(np.asarray(composition, dtype=np.float64))
        tree_preds = np.array([t.predict(X)[0] for t in self._rf.estimators_], dtype=np.float64)
        pred_mean = float(np.mean(tree_preds))
        tree_var = float(np.var(tree_preds))
        var_floor = float(kwargs.get("variance_floor", VARIANCE_FLOOR))
        total_var = float(np.maximum(tree_var + self._residual_var, var_floor))

        return PredictiveDistribution(
            hypothesis_id=self._id,
            candidate_id=candidate_id,
            action_type=normalize_action_type(action_type),
            mean=np.array([pred_mean]),
            variance=np.array([total_var]),
            metadata={"model": "RandomForestRegressor", "n_samples": self._sample_count},
        )

    def log_predictive_density(
        self,
        observation: np.ndarray | float,
        prediction: PredictiveDistribution,
    ) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(
        self,
        candidate_id: str | None = None,
        action_type: ActionType | None = None,
        composition: np.ndarray | None = None,
        **kwargs: Any,
    ) -> str:
        return "Measured 20th-cycle capacity fails to match local decision-tree regime partitions across solvent clusters."


class ElectrolyteHypothesisProvider(HypothesisProvider):
    """Provides the three competing predictive structural hypotheses for electrolyte discovery."""

    def build_hypotheses(self) -> Mapping[str, ScientificHypothesisModel]:
        h1 = GlobalSmoothDescriptorHypothesis()
        h2 = RegularizedAdditiveDescriptorHypothesis()
        h3 = LocalChemicalRegimeHypothesis()
        return {
            h1.hypothesis_id: h1,
            h2.hypothesis_id: h2,
            h3.hypothesis_id: h3,
        }


def evaluate_hypothesis_calibration(
    df_historical: pd.DataFrame,
    feature_cols: Sequence[str] = ELECTROLYTE_SOLVENT_FEATURES,
    variance_floors: Sequence[float] = (0.0025, 0.005, 0.010),
) -> dict[str, Any]:
    """Evaluates predictive calibration and variance-floor sensitivity of H1, H2, and H3 on held-out historical data.

    Follows Phase 11 requirements:
    - MAE, RMSE
    - Mean and median log predictive density
    - 50% and 90% empirical interval coverage
    - Mean predicted uncertainty (std)
    - Standardized residual distribution
    - Variance floor sensitivity (0.5x, 1x, 2x)
    - Strict epistemic semantics: posterior preference among predictive models, NOT physical mechanism truth.
    """
    f_cols = list(feature_cols)
    batches = sorted(df_historical["batch"].unique())

    # 1. Leave-One-Batch-Out cross-validation for batches with adequate training data
    results_by_hyp = {
        "global_smooth_descriptor": {"y_true": [], "y_pred": [], "pred_std": [], "log_pdf": []},
        "regularized_additive_descriptor": {"y_true": [], "y_pred": [], "pred_std": [], "log_pdf": []},
        "local_chemical_regime": {"y_true": [], "y_pred": [], "pred_std": [], "log_pdf": []},
    }

    for test_b in batches:
        train_mask = (df_historical["batch"] != test_b)
        test_mask = (df_historical["batch"] == test_b)
        train_df = df_historical[train_mask]
        test_df = df_historical[test_mask]

        if len(train_df) < 5 or len(test_df) == 0:
            continue

        train_feats = {
            str(row["candidate_id"]): row[f_cols].to_numpy(dtype=np.float64)
            for _, row in train_df.iterrows()
        }
        train_obs = {
            str(row["candidate_id"]): float(row["C_norm_20"])
            for _, row in train_df.iterrows()
        }

        ctx = HypothesisTrainingContext(
            candidate_features_by_id=train_feats,
            observations_by_modality={"CAPACITY_TEST": train_obs},
        )

        provider = ElectrolyteHypothesisProvider()
        hyps = provider.build_hypotheses()
        for h in hyps.values():
            h.fit_context(ctx)

        for _, row in test_df.iterrows():
            cid = str(row["candidate_id"])
            y_true = float(row["C_norm_20"])
            comp = row[f_cols].to_numpy(dtype=np.float64)

            for hid, h in hyps.items():
                pred = h.predict_observation(
                    candidate_id=cid,
                    action_type="CAPACITY_TEST",
                    composition=comp,
                )
                p_mean = float(pred.mean[0])
                p_std = float(np.sqrt(pred.variance[0]))
                l_pdf = float(pred.log_pdf(y_true))

                results_by_hyp[hid]["y_true"].append(y_true)
                results_by_hyp[hid]["y_pred"].append(p_mean)
                results_by_hyp[hid]["pred_std"].append(p_std)
                results_by_hyp[hid]["log_pdf"].append(l_pdf)

    # Compute aggregate calibration metrics
    calibration_metrics = {}
    for hid, data in results_by_hyp.items():
        y_true_arr = np.array(data["y_true"], dtype=np.float64)
        y_pred_arr = np.array(data["y_pred"], dtype=np.float64)
        std_arr = np.array(data["pred_std"], dtype=np.float64)
        lpdf_arr = np.array(data["log_pdf"], dtype=np.float64)

        if len(y_true_arr) == 0:
            continue

        residuals = y_true_arr - y_pred_arr
        std_residuals = residuals / np.maximum(std_arr, 1e-6)

        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mean_lpd = float(np.mean(lpdf_arr))
        median_lpd = float(np.median(lpdf_arr))

        # Empirical predictive interval coverages: 50% (z=0.6745), 90% (z=1.6449)
        cov_50 = float(np.mean(np.abs(std_residuals) <= 0.67449))
        cov_90 = float(np.mean(np.abs(std_residuals) <= 1.64485))

        calibration_metrics[hid] = {
            "hypothesis_title": provider.build_hypotheses()[hid].title,
            "test_sample_count": len(y_true_arr),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mean_log_predictive_density": round(mean_lpd, 4),
            "median_log_predictive_density": round(median_lpd, 4),
            "coverage_50pct_interval": round(cov_50, 4),
            "coverage_90pct_interval": round(cov_90, 4),
            "mean_predicted_std": round(float(np.mean(std_arr)), 4),
            "standardized_residual_mean": round(float(np.mean(std_residuals)), 4),
            "standardized_residual_std": round(float(np.std(std_residuals)), 4),
        }

    # 2. Variance-floor sensitivity evaluation
    sensitivity_runs = []
    base_floor = VARIANCE_FLOOR
    for floor_val in variance_floors:
        ratio = floor_val / base_floor
        # Compute mean log predictive density for each hypothesis under this variance floor
        floor_metrics = {}
        for hid, data in results_by_hyp.items():
            y_true_arr = np.array(data["y_true"], dtype=np.float64)
            y_pred_arr = np.array(data["y_pred"], dtype=np.float64)
            std_raw = np.array(data["pred_std"], dtype=np.float64)
            # Recompute variance under new floor
            var_adj = np.maximum(std_raw ** 2, floor_val)
            std_adj = np.sqrt(var_adj)
            res = y_true_arr - y_pred_arr
            lpdf = -0.5 * np.log(2 * np.pi * var_adj) - 0.5 * (res ** 2) / var_adj
            floor_metrics[hid] = {
                "mean_lpd": round(float(np.mean(lpdf)), 4),
                "coverage_90": round(float(np.mean(np.abs(res / std_adj) <= 1.64485)), 4),
            }

        # Determine winner under this floor
        winner_hid = max(floor_metrics.keys(), key=lambda k: floor_metrics[k]["mean_lpd"])
        sensitivity_runs.append({
            "floor_scale": f"{ratio:.1f}x",
            "floor_variance": floor_val,
            "hypothesis_metrics": floor_metrics,
            "posterior_winner": winner_hid,
        })

    # Winner stability: check if winner is invariant across sensitivity floors
    winners = [s["posterior_winner"] for s in sensitivity_runs]
    is_winner_stable = bool(len(set(winners)) == 1)

    return {
        "calibration_scope": "Leave-One-Batch-Out Retrospective Validation over Pool-Compatible Historical Outcomes",
        "epistemic_statement": "Hypothesis posteriors quantify predictive preference among candidate structural models, NOT physical mechanism truth.",
        "hypotheses_calibration": calibration_metrics,
        "variance_floor_sensitivity": {
            "baseline_floor": base_floor,
            "evaluated_scales": [s["floor_scale"] for s in sensitivity_runs],
            "posterior_winner_stable": is_winner_stable,
            "sensitivity_runs": sensitivity_runs,
        },
    }
