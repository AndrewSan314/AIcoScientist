from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

from src.science.actions import ActionType, normalize_action_type
from src.science.domain import HypothesisProvider, HypothesisTrainingContext
from src.science.hypothesis_models import (
    PredictiveDistribution,
    ScientificHypothesisModel,
)

logger = logging.getLogger(__name__)


class PrecursorThermodynamicsHypothesis:
    """Hypothesis A: Solid-state synthesis outcome is governed predominantly by precursor chemistry and thermodynamic driving force.

    Thermal processing parameters (furnace temperature and time) are treated as secondary or non-explanatory.
    """

    def __init__(self, hypothesis_id: str = "precursor_thermodynamics") -> None:
        self._id = hypothesis_id
        self._gp: GaussianProcessRegressor | None = None
        self._is_fitted = False
        self._training_sample_count = 0
        self._revealed_xrd_mean: np.ndarray | None = None
        self._revealed_xrd_var: np.ndarray | None = None
        self._revealed_ref_mean: np.ndarray | None = None
        self._revealed_ref_var: np.ndarray | None = None

    @property
    def hypothesis_id(self) -> str:
        return self._id

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def training_sample_count(self) -> int:
        return self._training_sample_count

    def supports_action(self, action_type: ActionType) -> bool:
        norm = normalize_action_type(action_type)
        return norm in ("OUTCOME_TEST", "PROPERTY", "XRD", "REFINEMENT")

    def fit_context(self, context: HypothesisTrainingContext) -> PrecursorThermodynamicsHypothesis:
        conv_obs = (
            context.observations_by_modality.get("OUTCOME_TEST")
            or context.observations_by_modality.get("PROPERTY")
            or {}
        )
        xrd_obs = context.observations_by_modality.get("XRD", {})
        ref_obs = context.observations_by_modality.get("REFINEMENT", {})

        # 1. Empirical characterization statistics from revealed observations
        if xrd_obs:
            xrd_mat = np.array([np.asarray(v, dtype=np.float64) for v in xrd_obs.values() if v is not None])
            if len(xrd_mat) > 0:
                self._revealed_xrd_mean = np.mean(xrd_mat, axis=0)
                self._revealed_xrd_var = np.maximum(np.var(xrd_mat, axis=0), 0.05)

        if ref_obs:
            ref_mat = np.array([np.asarray(v, dtype=np.float64) for v in ref_obs.values() if v is not None])
            if len(ref_mat) > 0:
                self._revealed_ref_mean = np.mean(ref_mat, axis=0)
                self._revealed_ref_var = np.maximum(np.var(ref_mat, axis=0), 0.02)

        # 2. Fit outcome GP excluding thermal conditions (features 1 and 2: temp and time)
        if len(conv_obs) < 2:
            self._is_fitted = False
            self._training_sample_count = len(conv_obs)
            return self

        X_rows = []
        y_rows = []
        for cid, val in conv_obs.items():
            if cid in context.candidate_features_by_id and isinstance(val, (int, float, np.number)):
                feat = np.asarray(context.candidate_features_by_id[cid], dtype=np.float64)
                # Feature subset: [energy] + [multi-hot precursors]
                thermo_feat = np.concatenate(([feat[0]], feat[3:]))
                X_rows.append(thermo_feat)
                y_rows.append(float(val))

        if len(y_rows) < 2:
            self._is_fitted = False
            self._training_sample_count = len(y_rows)
            return self

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.float64)

        kernel = ConstantKernel(1.0, (1e-3, 10.0)) * RBF(length_scale=2.0) + WhiteKernel(0.05, (1e-4, 1.0))
        self._gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self._gp.fit(X, y)
        self._is_fitted = True
        self._training_sample_count = len(y_rows)
        return self

    def fit(self, **kwargs: Any) -> PrecursorThermodynamicsHypothesis:
        context = kwargs.get("context")
        if context is None and "composition_by_id" in kwargs:
            from src.science.domain import HypothesisTrainingContext
            context = HypothesisTrainingContext(
                candidate_features_by_id=kwargs.get("composition_by_id", {}),
                observations_by_modality=kwargs.get("observations_by_modality", {}),
                modality_definitions={m.name: m for m in kwargs.get("modality_definitions", [])} if isinstance(kwargs.get("modality_definitions"), (list, tuple)) else kwargs.get("modality_definitions", {}),
                objective_definitions={o.name: o for o in kwargs.get("objective_definitions", [])} if isinstance(kwargs.get("objective_definitions"), (list, tuple)) else kwargs.get("objective_definitions", {}),
            )
        if context is not None:
            return self.fit_context(context)
        return self

    def predict(self, candidate_id: str, action_type: ActionType, candidate_composition: np.ndarray | None = None, **kwargs: Any) -> PredictiveDistribution:
        return self.predict_observation(candidate_id=candidate_id, action_type=action_type, composition=candidate_composition, **kwargs)

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray | None = None,
        **kwargs: Any,
    ) -> PredictiveDistribution:
        norm = normalize_action_type(action_type)
        comp = composition if composition is not None else np.zeros(49)

        if norm in ("OUTCOME_TEST", "PROPERTY"):
            if self._is_fitted and self._gp is not None:
                thermo_feat = np.concatenate(([comp[0]], comp[3:])).reshape(1, -1)
                m, s = self._gp.predict(thermo_feat, return_std=True)
                mean_val = float(np.clip(m[0], 0.0, 1.0))
                var_val = float(max(s[0] ** 2, 1e-4))
            else:
                # Prior: negative reaction energy promotes reaction
                energy = float(comp[0]) if len(comp) > 0 else 0.0
                mean_val = float(np.clip(0.5 - 2.0 * energy, 0.05, 0.95))
                var_val = 0.20

            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([mean_val], dtype=np.float64),
                variance=np.array([var_val], dtype=np.float64),
                metadata={"model": "precursor_thermodynamics_gp", "training_count": self._training_sample_count},
            )
        elif norm == "XRD":
            emb_dim = 8
            mean_vec = self._revealed_xrd_mean if self._revealed_xrd_mean is not None else np.zeros(emb_dim, dtype=np.float64)
            var_vec = self._revealed_xrd_var if self._revealed_xrd_var is not None else np.ones(emb_dim, dtype=np.float64) * 0.3
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean_vec,
                variance=var_vec,
                metadata={"model": "thermodynamics_empirical_xrd", "training_count": self._training_sample_count},
            )
        elif norm == "REFINEMENT":
            ref_dim = 4
            mean_vec = self._revealed_ref_mean if self._revealed_ref_mean is not None else np.array([0.1, 0.7, 0.2, 0.5], dtype=np.float64)
            var_vec = self._revealed_ref_var if self._revealed_ref_var is not None else np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float64)
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean_vec,
                variance=var_vec,
                metadata={"model": "thermodynamics_empirical_refinement", "training_count": self._training_sample_count},
            )

        raise ValueError(f"Unsupported action type for {self._id}: {action_type}")

    def log_predictive_density(self, observation: np.ndarray | float, prediction: PredictiveDistribution) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(self, candidate_id: str, action_type: ActionType, composition: np.ndarray) -> str:
        return "Synthesis outcome significantly deviates across thermal conditions despite favorable precursor thermodynamics."


class ProcessKineticsHypothesis:
    """Hypothesis B: Synthesis thermal processing conditions (heating temperature & time) govern solid-state reaction kinetics."""

    def __init__(self, hypothesis_id: str = "process_kinetics") -> None:
        self._id = hypothesis_id
        self._gp: GaussianProcessRegressor | None = None
        self._is_fitted = False
        self._training_sample_count = 0
        self._revealed_xrd_mean: np.ndarray | None = None
        self._revealed_xrd_var: np.ndarray | None = None

    @property
    def hypothesis_id(self) -> str:
        return self._id

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def training_sample_count(self) -> int:
        return self._training_sample_count

    def supports_action(self, action_type: ActionType) -> bool:
        norm = normalize_action_type(action_type)
        return norm in ("OUTCOME_TEST", "PROPERTY", "XRD", "REFINEMENT")

    def fit_context(self, context: HypothesisTrainingContext) -> ProcessKineticsHypothesis:
        conv_obs = (
            context.observations_by_modality.get("OUTCOME_TEST")
            or context.observations_by_modality.get("PROPERTY")
            or {}
        )
        xrd_obs = context.observations_by_modality.get("XRD", {})

        if xrd_obs:
            xrd_mat = np.array([np.asarray(v, dtype=np.float64) for v in xrd_obs.values() if v is not None])
            if len(xrd_mat) > 0:
                self._revealed_xrd_mean = np.mean(xrd_mat, axis=0)
                self._revealed_xrd_var = np.maximum(np.var(xrd_mat, axis=0), 0.05)

        if len(conv_obs) < 2:
            self._is_fitted = False
            self._training_sample_count = len(conv_obs)
            return self

        X_rows = []
        y_rows = []
        for cid, val in conv_obs.items():
            if cid in context.candidate_features_by_id and isinstance(val, (int, float, np.number)):
                feat = np.asarray(context.candidate_features_by_id[cid], dtype=np.float64)
                X_rows.append(feat)
                y_rows.append(float(val))

        if len(y_rows) < 2:
            self._is_fitted = False
            self._training_sample_count = len(y_rows)
            return self

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.float64)

        # Matern kernel with ARD emphasizing temperature and time (features 1 and 2)
        length_scales = np.ones(X.shape[1]) * 2.0
        length_scales[1] = 0.5  # temperature sensitivity
        length_scales[2] = 0.5  # time sensitivity

        kernel = ConstantKernel(1.0, (1e-3, 10.0)) * Matern(length_scale=length_scales, nu=2.5) + WhiteKernel(0.03, (1e-4, 1.0))
        self._gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self._gp.fit(X, y)
        self._is_fitted = True
        self._training_sample_count = len(y_rows)
        return self

    def fit(self, **kwargs: Any) -> ProcessKineticsHypothesis:
        context = kwargs.get("context")
        if context is None and "composition_by_id" in kwargs:
            from src.science.domain import HypothesisTrainingContext
            context = HypothesisTrainingContext(
                candidate_features_by_id=kwargs.get("composition_by_id", {}),
                observations_by_modality=kwargs.get("observations_by_modality", {}),
                modality_definitions={m.name: m for m in kwargs.get("modality_definitions", [])} if isinstance(kwargs.get("modality_definitions"), (list, tuple)) else kwargs.get("modality_definitions", {}),
                objective_definitions={o.name: o for o in kwargs.get("objective_definitions", [])} if isinstance(kwargs.get("objective_definitions"), (list, tuple)) else kwargs.get("objective_definitions", {}),
            )
        if context is not None:
            return self.fit_context(context)
        return self

    def predict(self, candidate_id: str, action_type: ActionType, candidate_composition: np.ndarray | None = None, **kwargs: Any) -> PredictiveDistribution:
        return self.predict_observation(candidate_id=candidate_id, action_type=action_type, composition=candidate_composition, **kwargs)

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray | None = None,
        **kwargs: Any,
    ) -> PredictiveDistribution:
        norm = normalize_action_type(action_type)
        comp = composition if composition is not None else np.zeros(49)

        if norm in ("OUTCOME_TEST", "PROPERTY"):
            if self._is_fitted and self._gp is not None:
                feat_vec = np.asarray(comp, dtype=np.float64).reshape(1, -1)
                m, s = self._gp.predict(feat_vec, return_std=True)
                mean_val = float(np.clip(m[0], 0.0, 1.0))
                var_val = float(max(s[0] ** 2, 1e-4))
            else:
                # Prior: higher temperature & longer duration promote solid-state kinetics
                temp_norm = float(comp[1]) if len(comp) > 1 else 0.5
                time_norm = float(comp[2]) if len(comp) > 2 else 0.5
                mean_val = float(np.clip(0.2 + 0.5 * temp_norm + 0.2 * time_norm, 0.05, 0.95))
                var_val = 0.18

            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([mean_val], dtype=np.float64),
                variance=np.array([var_val], dtype=np.float64),
                metadata={"model": "process_kinetics_matern_gp", "training_count": self._training_sample_count},
            )
        elif norm == "XRD":
            emb_dim = 8
            mean_vec = np.zeros(emb_dim, dtype=np.float64)
            if self._revealed_xrd_mean is not None:
                mean_vec = np.copy(self._revealed_xrd_mean)
            # Temperature correlates with crystalline peak sharpening / PC1
            temp_norm = float(comp[1]) if len(comp) > 1 else 0.5
            mean_vec[0] = (temp_norm - 0.5) * 0.4
            var_vec = self._revealed_xrd_var if self._revealed_xrd_var is not None else np.ones(emb_dim, dtype=np.float64) * 0.25
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean_vec,
                variance=var_vec,
                metadata={"model": "kinetics_thermal_xrd", "training_count": self._training_sample_count},
            )
        elif norm == "REFINEMENT":
            temp_norm = float(comp[1]) if len(comp) > 1 else 0.5
            target_est = float(np.clip(0.1 + 0.4 * temp_norm, 0.0, 1.0))
            prec_est = float(np.clip(0.8 - 0.4 * temp_norm, 0.0, 1.0))
            other_est = float(np.clip(1.0 - (target_est + prec_est), 0.0, 1.0))
            mean_vec = np.array([target_est, prec_est, other_est, 0.4], dtype=np.float64)
            var_vec = np.array([0.04, 0.04, 0.04, 0.04], dtype=np.float64)
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean_vec,
                variance=var_vec,
                metadata={"model": "kinetics_refinement_prior", "training_count": self._training_sample_count},
            )

        raise ValueError(f"Unsupported action type for {self._id}: {action_type}")

    def log_predictive_density(self, observation: np.ndarray | float, prediction: PredictiveDistribution) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(self, candidate_id: str, action_type: ActionType, composition: np.ndarray) -> str:
        return "High-temperature heating conditions fail to accelerate target reaction formation or exhibit kinetic stagnation."


class StructurePhaseInformedHypothesis:
    """Hypothesis C: Measured crystalline structure and Rietveld phase evolution provide decisive explanatory evidence for reaction outcome."""

    def __init__(self, hypothesis_id: str = "structure_phase_informed") -> None:
        self._id = hypothesis_id
        self._gp: GaussianProcessRegressor | None = None
        self._is_fitted = False
        self._training_sample_count = 0
        self._revealed_xrd_mean: np.ndarray | None = None
        self._revealed_xrd_var: np.ndarray | None = None
        self._revealed_ref_mean: np.ndarray | None = None
        self._revealed_ref_var: np.ndarray | None = None

    @property
    def hypothesis_id(self) -> str:
        return self._id

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def training_sample_count(self) -> int:
        return self._training_sample_count

    def supports_action(self, action_type: ActionType) -> bool:
        norm = normalize_action_type(action_type)
        return norm in ("OUTCOME_TEST", "PROPERTY", "XRD", "REFINEMENT")

    def _build_multimodal_feature(
        self,
        candidate_features: np.ndarray,
        xrd_val: Any | None,
        ref_val: Any | None,
    ) -> list[float]:
        """Builds explicit multimodal descriptor vector with boolean missingness indicators."""
        feat = list(np.asarray(candidate_features, dtype=np.float64))

        # Explicit XRD evidence block: [has_xrd: 1.0/0.0, pc0, pc1, pc2, pc3]
        if xrd_val is not None and isinstance(xrd_val, (list, tuple, np.ndarray)):
            xrd_arr = list(np.asarray(xrd_val, dtype=np.float64)[:4])
            if len(xrd_arr) < 4:
                xrd_arr.extend([0.0] * (4 - len(xrd_arr)))
            feat.append(1.0)
            feat.extend(xrd_arr)
        else:
            feat.append(0.0)
            feat.extend([0.0, 0.0, 0.0, 0.0])

        # Explicit Refinement evidence block: [has_ref: 1.0/0.0, target_frac, prec_frac, other_frac, rwp_scaled]
        if ref_val is not None and isinstance(ref_val, (list, tuple, np.ndarray)):
            ref_arr = list(np.asarray(ref_val, dtype=np.float64)[:4])
            if len(ref_arr) < 4:
                ref_arr.extend([0.0] * (4 - len(ref_arr)))
            feat.append(1.0)
            feat.extend(ref_arr)
        else:
            feat.append(0.0)
            feat.extend([0.0, 0.0, 0.0, 0.0])

        return feat

    def fit_context(self, context: HypothesisTrainingContext) -> StructurePhaseInformedHypothesis:
        conv_obs = (
            context.observations_by_modality.get("OUTCOME_TEST")
            or context.observations_by_modality.get("PROPERTY")
            or {}
        )
        xrd_obs = context.observations_by_modality.get("XRD", {})
        ref_obs = context.observations_by_modality.get("REFINEMENT", {})

        if xrd_obs:
            xrd_mat = np.array([np.asarray(v, dtype=np.float64) for v in xrd_obs.values() if v is not None])
            if len(xrd_mat) > 0:
                self._revealed_xrd_mean = np.mean(xrd_mat, axis=0)
                self._revealed_xrd_var = np.maximum(np.var(xrd_mat, axis=0), 0.05)

        if ref_obs:
            ref_mat = np.array([np.asarray(v, dtype=np.float64) for v in ref_obs.values() if v is not None])
            if len(ref_mat) > 0:
                self._revealed_ref_mean = np.mean(ref_mat, axis=0)
                self._revealed_ref_var = np.maximum(np.var(ref_mat, axis=0), 0.02)

        if len(conv_obs) < 2:
            self._is_fitted = False
            self._training_sample_count = len(conv_obs)
            return self

        X_rows = []
        y_rows = []
        for cid, val in conv_obs.items():
            if cid in context.candidate_features_by_id and isinstance(val, (int, float, np.number)):
                cand_feat = np.asarray(context.candidate_features_by_id[cid], dtype=np.float64)
                xrd_v = xrd_obs.get(cid)
                ref_v = ref_obs.get(cid)
                mm_feat = self._build_multimodal_feature(cand_feat, xrd_v, ref_v)
                X_rows.append(mm_feat)
                y_rows.append(float(val))

        if len(y_rows) < 2:
            self._is_fitted = False
            self._training_sample_count = len(y_rows)
            return self

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.float64)

        kernel = ConstantKernel(1.0, (1e-3, 10.0)) * RBF(length_scale=2.0) + WhiteKernel(0.02, (1e-4, 1.0))
        self._gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self._gp.fit(X, y)
        self._is_fitted = True
        self._training_sample_count = len(y_rows)
        return self

    def fit(self, **kwargs: Any) -> StructurePhaseInformedHypothesis:
        context = kwargs.get("context")
        if context is None and "composition_by_id" in kwargs:
            from src.science.domain import HypothesisTrainingContext
            context = HypothesisTrainingContext(
                candidate_features_by_id=kwargs.get("composition_by_id", {}),
                observations_by_modality=kwargs.get("observations_by_modality", {}),
                modality_definitions={m.name: m for m in kwargs.get("modality_definitions", [])} if isinstance(kwargs.get("modality_definitions"), (list, tuple)) else kwargs.get("modality_definitions", {}),
                objective_definitions={o.name: o for o in kwargs.get("objective_definitions", [])} if isinstance(kwargs.get("objective_definitions"), (list, tuple)) else kwargs.get("objective_definitions", {}),
            )
        if context is not None:
            return self.fit_context(context)
        return self

    def predict(self, candidate_id: str, action_type: ActionType, candidate_composition: np.ndarray | None = None, **kwargs: Any) -> PredictiveDistribution:
        return self.predict_observation(candidate_id=candidate_id, action_type=action_type, composition=candidate_composition, **kwargs)

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray | None = None,
        observed_xrd_embedding: np.ndarray | None = None,
        observed_modalities: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> PredictiveDistribution:
        norm = normalize_action_type(action_type)
        comp = composition if composition is not None else np.zeros(49)

        # Extract revealed structural evidence for this candidate
        xrd_val = None
        ref_val = None
        if observed_modalities is not None:
            xrd_val = observed_modalities.get("XRD", {}).get(candidate_id)
            ref_val = observed_modalities.get("REFINEMENT", {}).get(candidate_id)
        if xrd_val is None and observed_xrd_embedding is not None:
            xrd_val = observed_xrd_embedding

        if norm in ("OUTCOME_TEST", "PROPERTY"):
            if self._is_fitted and self._gp is not None:
                mm_feat = np.array([self._build_multimodal_feature(comp, xrd_val, ref_val)], dtype=np.float64)
                m, s = self._gp.predict(mm_feat, return_std=True)
                mean_val = float(np.clip(m[0], 0.0, 1.0))
                var_floor = 1e-4 if (xrd_val is not None or ref_val is not None) else 5e-3
                var_val = float(max(s[0] ** 2, var_floor))
            else:
                if ref_val is not None and isinstance(ref_val, (list, tuple, np.ndarray)):
                    # Target phase fraction directly anchors outcome expectation
                    target_frac = float(ref_val[0])
                    mean_val = float(np.clip(0.1 + 0.9 * target_frac, 0.05, 0.95))
                    var_val = 0.06
                else:
                    mean_val = 0.5
                    var_val = 0.22

            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([mean_val], dtype=np.float64),
                variance=np.array([var_val], dtype=np.float64),
                metadata={"model": "structure_phase_informed_gp", "training_count": self._training_sample_count},
            )
        elif norm == "XRD":
            emb_dim = 8
            mean_vec = self._revealed_xrd_mean if self._revealed_xrd_mean is not None else np.zeros(emb_dim, dtype=np.float64)
            var_vec = self._revealed_xrd_var if self._revealed_xrd_var is not None else np.ones(emb_dim, dtype=np.float64) * 0.25
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean_vec,
                variance=var_vec,
                metadata={"model": "structure_empirical_xrd", "training_count": self._training_sample_count},
            )
        elif norm == "REFINEMENT":
            ref_dim = 4
            mean_vec = self._revealed_ref_mean if self._revealed_ref_mean is not None else np.array([0.2, 0.5, 0.3, 0.5], dtype=np.float64)
            var_vec = self._revealed_ref_var if self._revealed_ref_var is not None else np.array([0.04, 0.04, 0.04, 0.04], dtype=np.float64)
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean_vec,
                variance=var_vec,
                metadata={"model": "structure_empirical_refinement", "training_count": self._training_sample_count},
            )

        raise ValueError(f"Unsupported action type for {self._id}: {action_type}")

    def log_predictive_density(self, observation: np.ndarray | float, prediction: PredictiveDistribution) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(self, candidate_id: str, action_type: ActionType, composition: np.ndarray) -> str:
        return "Measured diffraction profile or Rietveld phase evolution contradicts predicted crystalline reaction pathway."


class ALabHypothesisProvider(HypothesisProvider):
    """Factory provider creating the 3 canonical competing A-Lab solid-state hypotheses."""

    def build_hypotheses(self) -> dict[str, ScientificHypothesisModel]:
        return {
            "precursor_thermodynamics": PrecursorThermodynamicsHypothesis(),
            "process_kinetics": ProcessKineticsHypothesis(),
            "structure_phase_informed": StructurePhaseInformedHypothesis(),
        }

    def get_hypotheses(self) -> list[ScientificHypothesisModel]:
        return list(self.build_hypotheses().values())
