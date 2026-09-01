from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from src.science.actions import ActionType, normalize_action_type
from src.science.domain import HypothesisProvider, HypothesisTrainingContext
from src.science.hypothesis_models import (
    PredictiveDistribution,
    ScientificHypothesisModel,
)


class CompositionOnlyHypothesis:
    """Hypothesis predicting capacity strictly from nominal stoichiometry (Li_ratio, doping_conc)."""

    def __init__(self) -> None:
        self.gp_capacity = GaussianProcessRegressor(
            kernel=ConstantKernel(100.0) * RBF(length_scale=0.3) + WhiteKernel(noise_level=4.0),
            normalize_y=True,
            random_state=42,
        )
        self._fitted_capacity = False
        self._training_sample_count = 0

    @property
    def hypothesis_id(self) -> str:
        return "composition_only"

    @property
    def title(self) -> str:
        return "Composition-Controlled Capacity Hypothesis"

    @property
    def statement(self) -> str:
        return "Discharge capacity is dictated purely by Li/TM stoichiometry and dopant concentration."

    @property
    def assumptions(self) -> list[str]:
        return ["Sintering temperature does not induce material phase transitions", "Morphology is irrelevant to bulk capacity"]

    @property
    def is_fitted(self) -> bool:
        return self._fitted_capacity

    @property
    def fitted_capacity(self) -> bool:
        return self._fitted_capacity

    @property
    def training_sample_count(self) -> int:
        return self._training_sample_count

    def fit_context(self, context: HypothesisTrainingContext) -> None:
        """Fits hypothesis using generic domain training context."""
        cap_obs = context.observations_by_modality.get("CAPACITY_TEST", {})
        comp_map = context.candidate_features_by_id
        self.fit(composition_by_id=comp_map, property_by_id=cap_obs)

    def fit(
        self,
        composition_by_id: Mapping[str, np.ndarray],
        property_by_id: Mapping[str, float] | None = None,
        xrd_embedding_by_id: Mapping[str, np.ndarray] | None = None,
        observed_xrd_ids: set[str] | None = None,
        observed_property_ids: set[str] | None = None,
        observations_by_modality: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        props = dict(property_by_id or {})
        if observations_by_modality and "CAPACITY_TEST" in observations_by_modality:
            for cid, v in observations_by_modality["CAPACITY_TEST"].items():
                if isinstance(v, (int, float, np.number)):
                    props[cid] = float(v)

        obs_ids = [cid for cid in (observed_property_ids or props.keys()) if cid in props and cid in composition_by_id]
        if len(obs_ids) >= 2:
            X = np.array([composition_by_id[cid][:2] for cid in obs_ids], dtype=np.float64)  # Li_ratio, doping
            y = np.array([props[cid] for cid in obs_ids], dtype=np.float64)
            self.gp_capacity.fit(X, y)
            self._fitted_capacity = True
            self._training_sample_count = len(obs_ids)
        else:
            self._training_sample_count = len(obs_ids)

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray,
        observed_xrd_embedding: np.ndarray | None = None,
        observed_modalities: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> PredictiveDistribution:
        act_norm = normalize_action_type(action_type)
        if act_norm == "CAPACITY_TEST":
            comp2 = composition[:2].reshape(1, -1)
            if self._fitted_capacity:
                mean_val, std_val = self.gp_capacity.predict(comp2, return_std=True)
                mean = np.array([float(mean_val[0])])
                variance = np.array([float(max(std_val[0] ** 2, 1.0))])
            else:
                base = 140.0 + 30.0 * float(composition[0]) - 20.0 * float(composition[1])
                mean = np.array([base])
                variance = np.array([36.0])
            return PredictiveDistribution(
                hypothesis_id=self.hypothesis_id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean,
                variance=variance,
            )
        else:  # SEM morphology
            return PredictiveDistribution(
                hypothesis_id=self.hypothesis_id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([0.5, 0.5, 0.5, 0.5]),
                variance=np.array([0.25, 0.25, 0.25, 0.25]),
            )

    def log_predictive_density(
        self,
        observation: np.ndarray | float,
        prediction: PredictiveDistribution,
    ) -> float:
        return prediction.log_pdf(observation)

    def supports_action(self, action_type: ActionType) -> bool:
        return normalize_action_type(action_type) in ("CAPACITY_TEST", "SEM")

    def falsification_summary(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray,
    ) -> str:
        return f"Refuted if observed capacity on {candidate_id} deviates by >3 sigma from nominal composition curve."


class TemperatureMediatedHypothesis:
    """Hypothesis predicting nonlinear capacity activation mediated by sintering temperature."""

    def __init__(self) -> None:
        self.gp_capacity = GaussianProcessRegressor(
            kernel=ConstantKernel(100.0) * RBF(length_scale=[0.3, 0.3, 50.0]) + WhiteKernel(noise_level=4.0),
            normalize_y=True,
            random_state=42,
        )
        self.gp_sem = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=50.0) + WhiteKernel(noise_level=0.1),
            normalize_y=True,
            random_state=42,
        )
        self._fitted_capacity = False
        self._fitted_sem = False
        self._training_sample_count = 0

    @property
    def hypothesis_id(self) -> str:
        return "temperature_mediated"

    @property
    def title(self) -> str:
        return "Thermal Sintering Window Hypothesis"

    @property
    def statement(self) -> str:
        return "Capacity activation requires precise sintering temperature tuning (750-850 C)."

    @property
    def assumptions(self) -> list[str]:
        return ["Under-sintering produces unreacted precursors", "Over-sintering causes Li loss and phase degradation"]

    @property
    def is_fitted(self) -> bool:
        return self._fitted_capacity or self._fitted_sem

    @property
    def fitted_capacity(self) -> bool:
        return self._fitted_capacity

    @property
    def fitted_sem(self) -> bool:
        return self._fitted_sem

    @property
    def training_sample_count(self) -> int:
        return self._training_sample_count

    def fit_context(self, context: HypothesisTrainingContext) -> None:
        """Fits hypothesis using generic domain training context."""
        cap_obs = context.observations_by_modality.get("CAPACITY_TEST", {})
        sem_obs = context.observations_by_modality.get("SEM", {})
        comp_map = context.candidate_features_by_id
        self.fit(composition_by_id=comp_map, property_by_id=cap_obs, xrd_embedding_by_id=sem_obs)

    def fit(
        self,
        composition_by_id: Mapping[str, np.ndarray],
        property_by_id: Mapping[str, float] | None = None,
        xrd_embedding_by_id: Mapping[str, np.ndarray] | None = None,
        observed_xrd_ids: set[str] | None = None,
        observed_property_ids: set[str] | None = None,
        observations_by_modality: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        props = dict(property_by_id or {})
        sem_map = dict(xrd_embedding_by_id or {})

        if observations_by_modality:
            if "CAPACITY_TEST" in observations_by_modality:
                for cid, v in observations_by_modality["CAPACITY_TEST"].items():
                    if isinstance(v, (int, float, np.number)):
                        props[cid] = float(v)
            if "SEM" in observations_by_modality:
                for cid, v in observations_by_modality["SEM"].items():
                    if v is not None and not isinstance(v, (dict, Mapping)):
                        sem_map[cid] = np.asarray(v, dtype=np.float64)

        obs_cap_ids = [cid for cid in (observed_property_ids or props.keys()) if cid in props and cid in composition_by_id]
        if len(obs_cap_ids) >= 3:
            X = np.array([composition_by_id[cid][:3] for cid in obs_cap_ids], dtype=np.float64)
            y = np.array([props[cid] for cid in obs_cap_ids], dtype=np.float64)
            self.gp_capacity.fit(X, y)
            self._fitted_capacity = True

        obs_sem_ids = [cid for cid in (observed_xrd_ids or sem_map.keys()) if cid in sem_map and cid in composition_by_id]
        if len(obs_sem_ids) >= 2:
            X_temp = np.array([[composition_by_id[cid][2]] for cid in obs_sem_ids], dtype=np.float64)
            Y_sem = np.array([sem_map[cid] for cid in obs_sem_ids], dtype=np.float64)
            self.gp_sem.fit(X_temp, Y_sem)
            self._fitted_sem = True

        self._training_sample_count = len(obs_cap_ids) + len(obs_sem_ids)

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray,
        observed_xrd_embedding: np.ndarray | None = None,
        observed_modalities: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> PredictiveDistribution:
        act_norm = normalize_action_type(action_type)
        if act_norm == "CAPACITY_TEST":
            comp3 = composition[:3].reshape(1, -1)
            if self._fitted_capacity:
                mean_val, std_val = self.gp_capacity.predict(comp3, return_std=True)
                mean = np.array([float(mean_val[0])])
                variance = np.array([float(max(std_val[0] ** 2, 1.0))])
            else:
                temp = float(composition[2]) if len(composition) > 2 else 800.0
                temp_factor = np.exp(-0.5 * ((temp - 800.0) / 40.0) ** 2)
                base = 130.0 + 50.0 * temp_factor + 20.0 * float(composition[0])
                mean = np.array([base])
                variance = np.array([25.0])
            return PredictiveDistribution(
                hypothesis_id=self.hypothesis_id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean,
                variance=variance,
            )
        else:  # SEM morphology
            temp = float(composition[2]) if len(composition) > 2 else 800.0
            if self._fitted_sem:
                t_in = np.array([[temp]], dtype=np.float64)
                mean_sem, std_sem = self.gp_sem.predict(t_in, return_std=True)
                return PredictiveDistribution(
                    hypothesis_id=self.hypothesis_id,
                    candidate_id=candidate_id,
                    action_type=action_type,
                    mean=mean_sem[0],
                    variance=np.maximum(std_sem[0] ** 2, 0.01),
                )
            else:
                grain_size = float(np.clip((temp - 600.0) / 400.0, 0.1, 1.0))
                return PredictiveDistribution(
                    hypothesis_id=self.hypothesis_id,
                    candidate_id=candidate_id,
                    action_type=action_type,
                    mean=np.array([grain_size, 0.6, 0.4, grain_size * 0.8]),
                    variance=np.array([0.1, 0.1, 0.1, 0.1]),
                )

    def log_predictive_density(
        self,
        observation: np.ndarray | float,
        prediction: PredictiveDistribution,
    ) -> float:
        return prediction.log_pdf(observation)

    def supports_action(self, action_type: ActionType) -> bool:
        return normalize_action_type(action_type) in ("CAPACITY_TEST", "SEM")

    def falsification_summary(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray,
    ) -> str:
        return f"Refuted if temperature variation fails to modulate capacity on candidate {candidate_id}."


class MicrostructureInformedHypothesis:
    """Hypothesis predicting capacity informed by observed SEM characterization features."""

    def __init__(self) -> None:
        self.gp_capacity = GaussianProcessRegressor(
            kernel=ConstantKernel(100.0) * RBF(length_scale=0.5) + WhiteKernel(noise_level=2.0),
            normalize_y=True,
            random_state=42,
        )
        self._fitted_capacity = False
        self._training_sample_count = 0

    @property
    def hypothesis_id(self) -> str:
        return "microstructure_informed"

    @property
    def title(self) -> str:
        return "Microstructure-Informed Performance Hypothesis"

    @property
    def statement(self) -> str:
        return "Observed particle morphology and grain connectivity directly govern high-rate capacity."

    @property
    def assumptions(self) -> list[str]:
        return ["SEM morphology vector reflects lithium ion diffusion path lengths"]

    @property
    def is_fitted(self) -> bool:
        return self._fitted_capacity

    @property
    def fitted_capacity(self) -> bool:
        return self._fitted_capacity

    @property
    def training_sample_count(self) -> int:
        return self._training_sample_count

    def fit_context(self, context: HypothesisTrainingContext) -> None:
        """Fits hypothesis using generic domain training context."""
        cap_obs = context.observations_by_modality.get("CAPACITY_TEST", {})
        sem_obs = context.observations_by_modality.get("SEM", {})
        comp_map = context.candidate_features_by_id
        self.fit(composition_by_id=comp_map, property_by_id=cap_obs, xrd_embedding_by_id=sem_obs)

    def fit(
        self,
        composition_by_id: Mapping[str, np.ndarray],
        property_by_id: Mapping[str, float] | None = None,
        xrd_embedding_by_id: Mapping[str, np.ndarray] | None = None,
        observed_xrd_ids: set[str] | None = None,
        observed_property_ids: set[str] | None = None,
        observations_by_modality: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        props = dict(property_by_id or {})
        char_map = dict(xrd_embedding_by_id or {})

        if observations_by_modality:
            if "CAPACITY_TEST" in observations_by_modality:
                for cid, v in observations_by_modality["CAPACITY_TEST"].items():
                    if isinstance(v, (int, float, np.number)):
                        props[cid] = float(v)
            if "SEM" in observations_by_modality:
                for cid, v in observations_by_modality["SEM"].items():
                    if v is not None and not isinstance(v, (dict, Mapping)):
                        char_map[cid] = np.asarray(v, dtype=np.float64)

        obs_ids = [cid for cid in (observed_property_ids or props.keys()) if cid in props and cid in char_map]
        if len(obs_ids) >= 2:
            X = np.array([char_map[cid] for cid in obs_ids], dtype=np.float64)
            y = np.array([props[cid] for cid in obs_ids], dtype=np.float64)
            self.gp_capacity.fit(X, y)
            self._fitted_capacity = True
            self._training_sample_count = len(obs_ids)
        else:
            self._training_sample_count = len(obs_ids)

    def predict_observation(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray,
        observed_xrd_embedding: np.ndarray | None = None,
        observed_modalities: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> PredictiveDistribution:
        act_norm = normalize_action_type(action_type)
        # Check if candidate has observed SEM in observed_modalities
        emb = observed_xrd_embedding
        if emb is None and observed_modalities is not None and "SEM" in observed_modalities:
            sem_data = observed_modalities["SEM"]
            if isinstance(sem_data, Mapping) and candidate_id in sem_data:
                emb = np.asarray(sem_data[candidate_id], dtype=np.float64)

        if act_norm == "CAPACITY_TEST":
            if emb is not None and self._fitted_capacity:
                emb_2d = emb.reshape(1, -1)
                mean_val, std_val = self.gp_capacity.predict(emb_2d, return_std=True)
                mean = np.array([float(mean_val[0])])
                variance = np.array([float(max(std_val[0] ** 2, 1.0))])
            elif emb is not None:
                morph_score = float(np.sum(emb))
                mean = np.array([135.0 + 25.0 * morph_score])
                variance = np.array([12.0])
            else:
                mean = np.array([155.0])
                variance = np.array([49.0])
            return PredictiveDistribution(
                hypothesis_id=self.hypothesis_id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=mean,
                variance=variance,
            )
        else:  # SEM morphology
            return PredictiveDistribution(
                hypothesis_id=self.hypothesis_id,
                candidate_id=candidate_id,
                action_type=action_type,
                mean=np.array([0.7, 0.3, 0.8, 0.5]),
                variance=np.array([0.08, 0.08, 0.08, 0.08]),
            )

    def log_predictive_density(
        self,
        observation: np.ndarray | float,
        prediction: PredictiveDistribution,
    ) -> float:
        return prediction.log_pdf(observation)

    def supports_action(self, action_type: ActionType) -> bool:
        return normalize_action_type(action_type) in ("CAPACITY_TEST", "SEM")

    def falsification_summary(
        self,
        candidate_id: str,
        action_type: ActionType,
        composition: np.ndarray,
    ) -> str:
        return f"Refuted if SEM morphology on {candidate_id} shows no correlation with measured capacity."


class ToyMaterialHypothesisProvider:
    """Hypothesis provider for the toy material domain."""

    def build_hypotheses(self) -> Mapping[str, ScientificHypothesisModel]:
        return {
            "composition_only": CompositionOnlyHypothesis(),
            "temperature_mediated": TemperatureMediatedHypothesis(),
            "microstructure_informed": MicrostructureInformedHypothesis(),
        }
