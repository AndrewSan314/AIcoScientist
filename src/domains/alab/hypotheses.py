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
    """Hypothesis A: Reaction conversion is determined primarily by precursor chemistry & thermodynamic driving force."""

    def __init__(self, hypothesis_id: str = "precursor_thermodynamics") -> None:
        self._id = hypothesis_id
        self._gp: GaussianProcessRegressor | None = None
        self._is_fitted = False
        self._training_sample_count = 0

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
        if len(conv_obs) < 2:
            self._is_fitted = False
            self._training_sample_count = len(conv_obs)
            return self

        X_rows = []
        y_rows = []
        for cid, val in conv_obs.items():
            if cid in context.candidate_features_by_id and isinstance(val, (int, float, np.number)):
                feat = np.asarray(context.candidate_features_by_id[cid], dtype=np.float64)
                # Use thermodynamic energy and precursor indices (feat[0], feat[3], feat[4])
                X_rows.append([feat[0], feat[3], feat[4]])
                y_rows.append(float(val))

        if len(y_rows) < 2:
            self._is_fitted = False
            self._training_sample_count = len(y_rows)
            return self

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.float64)

        kernel = ConstantKernel(1.0, (1e-3, 10.0)) * RBF(length_scale=[1.0, 5.0, 5.0]) + WhiteKernel(0.05, (1e-4, 1.0))
        self._gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self._gp.fit(X, y)
        self._is_fitted = True
        self._training_sample_count = len(y_rows)
        return self

    def fit(self, **kwargs: Any) -> PrecursorThermodynamicsHypothesis:
        context = kwargs.get("context")
        if context is not None:
            return self.fit_context(context)
        return self

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray | None = None,
        **kwargs: Any,
    ) -> PredictiveDistribution:
        norm = normalize_action_type(action_type)
        comp = composition if composition is not None else np.zeros(5)

        if norm in ("OUTCOME_TEST", "PROPERTY"):
            if self._is_fitted and self._gp is not None:
                feat_sub = np.array([[comp[0], comp[3], comp[4]]], dtype=np.float64)
                m, s = self._gp.predict(feat_sub, return_std=True)
                mean_val = float(np.clip(m[0], 0.0, 1.0))
                var_val = float(max(s[0] ** 2, 1e-4))
            else:
                mean_val = 0.5
                var_val = 0.25

            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([mean_val], dtype=np.float64),
                variance=np.array([var_val], dtype=np.float64),
                metadata={"model": "precursor_thermodynamics_gp"},
            )
        elif norm == "XRD":
            # Baseline structural expectation
            emb_dim = 8
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.zeros(emb_dim, dtype=np.float64),
                variance=np.ones(emb_dim, dtype=np.float64) * 0.5,
                metadata={"model": "unstructured_thermodynamic_prior"},
            )
        elif norm == "REFINEMENT":
            ref_dim = 4
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([0.5, 0.0, 0.0, 5.0], dtype=np.float64),
                variance=np.ones(ref_dim, dtype=np.float64) * 0.4,
                metadata={"model": "unstructured_refinement_prior"},
            )

        raise ValueError(f"Unsupported action type for {self._id}: {action_type}")

    def log_predictive_density(self, observation: np.ndarray | float, prediction: PredictiveDistribution) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(self, candidate_id: str, action_type: ActionType, composition: np.ndarray) -> str:
        return "Synthesis conversion significantly deviates across temperature conditions despite identical precursor thermodynamics."


class ProcessKineticsHypothesis:
    """Hypothesis B: Synthesis thermal program (heating temperature & time) governs solid-state kinetics."""

    def __init__(self, hypothesis_id: str = "process_kinetics") -> None:
        self._id = hypothesis_id
        self._gp: GaussianProcessRegressor | None = None
        self._is_fitted = False
        self._training_sample_count = 0

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

        kernel = ConstantKernel(1.0, (1e-3, 10.0)) * Matern(length_scale=[1.0, 0.5, 0.5, 5.0, 5.0], nu=2.5) + WhiteKernel(0.03, (1e-4, 1.0))
        self._gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self._gp.fit(X, y)
        self._is_fitted = True
        self._training_sample_count = len(y_rows)
        return self

    def fit(self, **kwargs: Any) -> ProcessKineticsHypothesis:
        context = kwargs.get("context")
        if context is not None:
            return self.fit_context(context)
        return self

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray | None = None,
        **kwargs: Any,
    ) -> PredictiveDistribution:
        norm = normalize_action_type(action_type)
        comp = composition if composition is not None else np.zeros(5)

        if norm in ("OUTCOME_TEST", "PROPERTY"):
            if self._is_fitted and self._gp is not None:
                feat_vec = np.asarray(comp, dtype=np.float64).reshape(1, -1)
                m, s = self._gp.predict(feat_vec, return_std=True)
                mean_val = float(np.clip(m[0], 0.0, 1.0))
                var_val = float(max(s[0] ** 2, 1e-4))
            else:
                # Prior: higher temperature yields higher kinetic conversion
                temp_norm = float(comp[1]) if len(comp) > 1 else 0.5
                mean_val = float(np.clip(0.3 + 0.5 * temp_norm, 0.0, 1.0))
                var_val = 0.20

            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([mean_val], dtype=np.float64),
                variance=np.array([var_val], dtype=np.float64),
                metadata={"model": "process_kinetics_matern_gp"},
            )
        elif norm == "XRD":
            emb_dim = 8
            # Thermal gradient influences peak sharpening / embedding drift
            temp_norm = float(comp[1]) if len(comp) > 1 else 0.0
            mean_vec = np.zeros(emb_dim, dtype=np.float64)
            mean_vec[0] = temp_norm * 0.5
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean_vec,
                variance=np.ones(emb_dim, dtype=np.float64) * 0.35,
                metadata={"model": "kinetics_thermal_xrd_prior"},
            )
        elif norm == "REFINEMENT":
            ref_dim = 4
            temp_norm = float(comp[1]) if len(comp) > 1 else 0.5
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([0.3 + 0.5 * temp_norm, 0.0, 0.0, 4.0], dtype=np.float64),
                variance=np.ones(ref_dim, dtype=np.float64) * 0.3,
                metadata={"model": "kinetics_refinement_prior"},
            )

        raise ValueError(f"Unsupported action type for {self._id}: {action_type}")

    def log_predictive_density(self, observation: np.ndarray | float, prediction: PredictiveDistribution) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(self, candidate_id: str, action_type: ActionType, composition: np.ndarray) -> str:
        return "High-temperature heating conditions fail to promote expected reaction conversion or exhibit kinetic stagnation."


class StructurePhaseInformedHypothesis:
    """Hypothesis C: Observed crystalline structure & Rietveld phase evolution provide decisive explanatory evidence."""

    def __init__(self, hypothesis_id: str = "structure_phase_informed") -> None:
        self._id = hypothesis_id
        self._gp: GaussianProcessRegressor | None = None
        self._is_fitted = False
        self._training_sample_count = 0

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

    def fit_context(self, context: HypothesisTrainingContext) -> StructurePhaseInformedHypothesis:
        conv_obs = (
            context.observations_by_modality.get("OUTCOME_TEST")
            or context.observations_by_modality.get("PROPERTY")
            or {}
        )
        xrd_obs = context.observations_by_modality.get("XRD", {})
        ref_obs = context.observations_by_modality.get("REFINEMENT", {})

        if len(conv_obs) < 2:
            self._is_fitted = False
            self._training_sample_count = len(conv_obs)
            return self

        X_rows = []
        y_rows = []
        for cid, val in conv_obs.items():
            if cid in context.candidate_features_by_id and isinstance(val, (int, float, np.number)):
                feat = list(np.asarray(context.candidate_features_by_id[cid], dtype=np.float64))
                # Add XRD embedding if observed, otherwise zeros
                if cid in xrd_obs and isinstance(xrd_obs[cid], (list, tuple, np.ndarray)):
                    feat.extend(list(np.asarray(xrd_obs[cid], dtype=np.float64)[:4]))
                else:
                    feat.extend([0.0, 0.0, 0.0, 0.0])

                # Add refinement target fraction if observed
                if cid in ref_obs and isinstance(ref_obs[cid], (list, tuple, np.ndarray)):
                    feat.append(float(ref_obs[cid][0]))
                else:
                    feat.append(0.0)

                X_rows.append(feat)
                y_rows.append(float(val))

        if len(y_rows) < 2:
            self._is_fitted = False
            self._training_sample_count = len(y_rows)
            return self

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.float64)

        kernel = ConstantKernel(1.0, (1e-3, 10.0)) * RBF(length_scale=np.ones(X.shape[1]) * 2.0) + WhiteKernel(0.02, (1e-4, 1.0))
        self._gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
        self._gp.fit(X, y)
        self._is_fitted = True
        self._training_sample_count = len(y_rows)
        return self

    def fit(self, **kwargs: Any) -> StructurePhaseInformedHypothesis:
        context = kwargs.get("context")
        if context is not None:
            return self.fit_context(context)
        return self

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
        comp = list(composition) if composition is not None else [0.0] * 5

        # Check for characterization evidence on this candidate
        xrd_val = None
        ref_val = None
        if observed_modalities is not None:
            xrd_val = observed_modalities.get("XRD", {}).get(candidate_id)
            ref_val = observed_modalities.get("REFINEMENT", {}).get(candidate_id)
        if xrd_val is None and observed_xrd_embedding is not None:
            xrd_val = observed_xrd_embedding

        if norm in ("OUTCOME_TEST", "PROPERTY"):
            if self._is_fitted and self._gp is not None:
                feat = list(comp)
                if xrd_val is not None and isinstance(xrd_val, (list, tuple, np.ndarray)):
                    feat.extend(list(np.asarray(xrd_val, dtype=np.float64)[:4]))
                else:
                    feat.extend([0.0, 0.0, 0.0, 0.0])

                if ref_val is not None and isinstance(ref_val, (list, tuple, np.ndarray)):
                    feat.append(float(ref_val[0]))
                else:
                    feat.append(0.0)

                feat_vec = np.asarray(feat, dtype=np.float64).reshape(1, -1)
                m, s = self._gp.predict(feat_vec, return_std=True)
                mean_val = float(np.clip(m[0], 0.0, 1.0))
                # Structural evidence tightens predictive uncertainty
                var_floor = 1e-4 if (xrd_val is not None or ref_val is not None) else 5e-3
                var_val = float(max(s[0] ** 2, var_floor))
            else:
                # If refinement is revealed and has high target fraction, high conversion
                if ref_val is not None and isinstance(ref_val, (list, tuple, np.ndarray)):
                    mean_val = float(ref_val[0])
                    var_val = 0.08
                else:
                    mean_val = 0.5
                    var_val = 0.22

            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([mean_val], dtype=np.float64),
                variance=np.array([var_val], dtype=np.float64),
                metadata={"model": "structure_phase_informed_gp", "has_xrd": xrd_val is not None},
            )
        elif norm == "XRD":
            emb_dim = 8
            # Specific distinctive structural expectation based on chemical precursors
            mean_vec = np.zeros(emb_dim, dtype=np.float64)
            p1_idx = float(comp[3]) if len(comp) > 3 else 0.0
            mean_vec[0] = (p1_idx % 5.0) * 0.2
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean_vec,
                variance=np.ones(emb_dim, dtype=np.float64) * 0.25,
                metadata={"model": "structural_phase_xrd_model"},
            )
        elif norm == "REFINEMENT":
            ref_dim = 4
            return PredictiveDistribution(
                hypothesis_id=self._id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([0.7, 0.1, 0.1, 3.2], dtype=np.float64),
                variance=np.ones(ref_dim, dtype=np.float64) * 0.2,
                metadata={"model": "structural_phase_refinement_model"},
            )

        raise ValueError(f"Unsupported action type for {self._id}: {action_type}")

    def log_predictive_density(self, observation: np.ndarray | float, prediction: PredictiveDistribution) -> float:
        return prediction.log_pdf(observation)

    def falsification_summary(self, candidate_id: str, action_type: ActionType, composition: np.ndarray) -> str:
        return "Crystalline diffraction features and Rietveld phase evolution fail to correlate with quantitative reaction conversion."


class ALabHypothesisProvider:
    """Hypothesis provider generating the three competing A-Lab scientific hypotheses."""

    def build_hypotheses(self) -> Mapping[str, ScientificHypothesisModel]:
        return {
            "precursor_thermodynamics": PrecursorThermodynamicsHypothesis(),
            "process_kinetics": ProcessKineticsHypothesis(),
            "structure_phase_informed": StructurePhaseInformedHypothesis(),
        }

    def get_hypotheses(self) -> Sequence[ScientificHypothesisModel]:
        return list(self.build_hypotheses().values())
