from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

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


class SparseAdditiveDescriptorHypothesis:
    """Hypothesis H2: Capacity is primarily explained by a low-complexity regularized additive combination of descriptors.

    Predictive model: Bayesian Ridge regression with analytical posterior predictive mean and variance.
    """

    def __init__(self, hypothesis_id: str = "sparse_additive_descriptor") -> None:
        self._id = hypothesis_id
        self._title = "Sparse Additive Descriptor Hypothesis"
        self._statement = "Capacity is governed by low-complexity additive descriptor linear combinations rather than localized non-linear regimes."
        self._assumptions = [
            "First-order linear contributions of solvent molecular weight and PCA components dominate capacity.",
            "Higher-order interactions are suppressed by shrinkage regularization.",
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

    def fit_context(self, context: HypothesisTrainingContext) -> SparseAdditiveDescriptorHypothesis:
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
        var = float(np.maximum(pred_std[0] ** 2, VARIANCE_FLOOR))

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
        return "Measured 20th-cycle capacity departs from sparse additive linear descriptor model due to nonlinear chemical interactions."


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
        total_var = float(np.maximum(tree_var + self._residual_var, VARIANCE_FLOOR))

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
        h2 = SparseAdditiveDescriptorHypothesis()
        h3 = LocalChemicalRegimeHypothesis()
        return {
            h1.hypothesis_id: h1,
            h2.hypothesis_id: h2,
            h3.hypothesis_id: h3,
        }
