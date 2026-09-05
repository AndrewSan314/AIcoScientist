from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol

import numpy as np

from src.science.multimodal.measurement_models import PredictiveObservableDistribution
from src.science.multimodal.ontology import observable_names_for_modality
from src.science.multimodal.schemas import ScientificObservable


class MultimodalScientificHypothesis(Protocol):
    """Contract for a hypothesis with shared latent state across modalities."""

    hypothesis_id: str
    title: str
    assumptions: list[str]

    def fit(self, candidate_features_by_id: Mapping[str, Any], observed_context: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        ...

    def predict_observable_distribution(
        self,
        candidate_id: str,
        modality: str,
        observed_context: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> PredictiveObservableDistribution:
        ...

    def log_likelihood(self, observable: ScientificObservable, observed_context: Mapping[str, Any] | None = None) -> float:
        ...

    def falsification_signature(self) -> dict[str, list[str]]:
        ...


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


class _ALabHypothesisBase(ABC):
    """CPU-safe shared latent-state model used for controlled and replay benchmarks."""

    hypothesis_id = ""
    title = ""
    assumptions: list[str] = []
    _variance_scale = 1.0

    def __init__(self) -> None:
        self._candidate_features: dict[str, np.ndarray] = {}
        self.training_count = 0

    def fit(
        self,
        candidate_features_by_id: Mapping[str, Any] | None = None,
        observed_context: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        features = candidate_features_by_id or kwargs.get("composition_by_id") or {}
        self._candidate_features = {
            str(cid): np.asarray(values, dtype=np.float64)
            for cid, values in features.items()
        }
        context = observed_context or kwargs.get("observations_by_modality") or {}
        self.training_count = sum(
            len(values) for values in context.values() if isinstance(values, Mapping)
        )

    def _features(self, candidate_id: str, supplied: Any | None = None) -> np.ndarray:
        if supplied is not None:
            return np.asarray(supplied, dtype=np.float64)
        return self._candidate_features.get(candidate_id, np.zeros(49, dtype=np.float64))

    @abstractmethod
    def latent_state(self, features: np.ndarray) -> dict[str, float]:
        """Returns an interpretable modelled latent state, not a causal claim."""

    def _context_value(self, candidate_id: str, modality: str, context: Mapping[str, Any] | None) -> np.ndarray | None:
        if not context:
            return None
        values = context.get(modality)
        if isinstance(values, Mapping) and candidate_id in values:
            raw = values[candidate_id]
            if isinstance(raw, ScientificObservable):
                raw = raw.value
            return np.atleast_1d(np.asarray(raw, dtype=np.float64))
        return None

    def _means(self, state: Mapping[str, float], modality: str, context: Mapping[str, Any] | None, candidate_id: str) -> np.ndarray:
        purity = float(np.clip(state["phase_purity"], 0.0, 1.0))
        homogeneity = float(np.clip(state["composition_homogeneity"], 0.0, 1.0))
        morphology = float(np.clip(state["morphology_quality"], 0.0, 1.0))
        completion = float(np.clip(state["reaction_completion"], 0.0, 1.0))
        kinetics = float(np.clip(state["kinetic_trapping"], 0.0, 1.0))

        # Conditional factorization: later modality means can use already observed XRD.
        xrd = self._context_value(candidate_id, "XRD", context)
        xrd_signal = float(np.clip(xrd[0], 0.0, 1.0)) if xrd is not None else purity
        if modality == "XRD":
            return np.array([
                0.18 + 0.45 * (1.0 - purity),
                0.20 + 0.35 * (1.0 - purity) + 0.05 * kinetics,
                0.08 + 0.65 * (1.0 - purity),
                0.20 + 0.50 * (1.0 - purity),
                1.0 + 4.0 * (1.0 - purity),
            ], dtype=np.float64)
        if modality == "REFINEMENT":
            return np.array([purity, 1.0 - purity, 1.0 - completion, 0.1 + 0.7 * (1.0 - purity)], dtype=np.float64)
        if modality == "SEM":
            return np.array([1.0 / max(morphology, 0.05), 0.18 + 0.30 * (1.0 - morphology), 1.0 - morphology, 0.25 + 0.5 * kinetics], dtype=np.float64)
        if modality == "EDS":
            return np.array([1.0 - homogeneity, 1.0 - homogeneity, 1.0 - homogeneity, homogeneity], dtype=np.float64)
        if modality in {"OUTCOME_TEST", "SYNTHESIS_OUTCOME", "PROPERTY"}:
            # XRD evidence conditions the outcome without pretending modalities are independent.
            return np.array([np.clip(0.45 * completion + 0.35 * xrd_signal + 0.20 * morphology, 0.0, 1.0)])
        raise ValueError(f"Unsupported multimodal modality: {modality}")

    def predict_observable_distribution(
        self,
        candidate_id: str,
        modality: str,
        observed_context: Mapping[str, Any] | None = None,
        candidate_features: Any | None = None,
        **_: Any,
    ) -> PredictiveObservableDistribution:
        features = self._features(candidate_id, candidate_features)
        state = self.latent_state(features)
        means = self._means(state, str(modality).upper(), observed_context, candidate_id)
        variance = np.full_like(means, 0.04 * self._variance_scale, dtype=np.float64)
        if means.size > 1:
            variance[1:] = 0.06 * self._variance_scale
        names = observable_names_for_modality(str(modality).upper())
        if means.size != len(names):
            raise ValueError(f"prediction schema has {len(names)} names for {means.size} values")
        return PredictiveObservableDistribution(
            hypothesis_id=self.hypothesis_id,
            candidate_id=candidate_id,
            modality=str(modality).upper(),
            mean=means,
            variance=variance,
            observable_names=names,
            metadata={
                "latent_state": state,
                "training_count": self.training_count,
                "variance_convention": "PREDICTIVE_VARIANCE_IS_TOTAL_OBSERVATION_VARIANCE",
                "measurement_uncertainty_semantics": "additional_measurement_error_only",
            },
        )

    def log_likelihood(self, observable: ScientificObservable, observed_context: Mapping[str, Any] | None = None) -> float:
        prediction = self.predict_observable_distribution(
            observable.candidate_id,
            observable.modality,
            observed_context,
        )
        observed_names = tuple(observable.observable_names or ())
        if not observed_names:
            raise ValueError("vector observation must declare observable_names")
        return prediction.log_pdf(
            observable.value,
            observed_names=observed_names,
            measurement_uncertainty=observable.uncertainty,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "assumptions": list(self.assumptions),
            "predicted_observables": {
                modality: list(observable_names_for_modality(modality))
                for modality in ("XRD", "REFINEMENT", "SEM", "EDS", "OUTCOME_TEST")
            },
            "falsification_signature": self.falsification_signature(),
            "training_count": self.training_count,
        }


class PhasePurityLimitedHypothesis(_ALabHypothesisBase):
    hypothesis_id = "H1_PHASE_PURITY_LIMITED"
    title = "Phase-purity limited"
    assumptions = ["Intended crystalline phase formation is the primary limiting state."]
    _variance_scale = 0.9

    def latent_state(self, features: np.ndarray) -> dict[str, float]:
        energy = float(features[0]) if len(features) else 0.0
        temp = float(features[1]) if len(features) > 1 else 0.5
        precursor_signal = float(np.mean(np.abs(features[3:]))) if len(features) > 3 else 0.5
        purity = np.clip(0.50 + 0.20 * _sigmoid(-energy) + 0.20 * temp - 0.05 * precursor_signal, 0.0, 1.0)
        return {"phase_purity": float(purity), "composition_homogeneity": 0.62, "morphology_quality": 0.55, "reaction_completion": float(purity), "kinetic_trapping": 1.0 - float(purity)}

    def falsification_signature(self) -> dict[str, list[str]]:
        return {
            "strongly_supporting_patterns": ["high target phase fraction", "low impurity signal", "successful outcome"],
            "strongly_falsifying_patterns": ["high impurity despite predicted purity", "failure with high refined target fraction"],
            "ambiguous_patterns": ["near-nominal EDS composition without phase assignment"],
        }


class CompositionHomogeneityLimitedHypothesis(_ALabHypothesisBase):
    hypothesis_id = "H2_COMPOSITION_HOMOGENEITY_LIMITED"
    title = "Composition-homogeneity limited"
    assumptions = ["Spatial composition variance can limit outcome despite acceptable bulk phase identity."]
    _variance_scale = 1.1

    def latent_state(self, features: np.ndarray) -> dict[str, float]:
        prec = np.asarray(features[3:], dtype=np.float64) if len(features) > 3 else np.zeros(1)
        spread = float(np.std(prec)) if prec.size else 0.5
        homogeneity = np.clip(0.92 - 1.8 * spread, 0.0, 1.0)
        purity = np.clip(0.58 + 0.08 * homogeneity, 0.0, 1.0)
        return {"phase_purity": float(purity), "composition_homogeneity": float(homogeneity), "morphology_quality": 0.58, "reaction_completion": float(0.45 + 0.45 * homogeneity), "kinetic_trapping": 0.35}

    def falsification_signature(self) -> dict[str, list[str]]:
        return {
            "strongly_supporting_patterns": ["high EDS spatial variance or segregation", "near-nominal bulk composition", "poor outcome tracks heterogeneity"],
            "strongly_falsifying_patterns": ["uniform EDS map with failed outcome", "high segregation predicted but absent"],
            "ambiguous_patterns": ["single-point EDS composition"],
        }


class MorphologyKineticsLimitedHypothesis(_ALabHypothesisBase):
    hypothesis_id = "H3_MORPHOLOGY_KINETICS_LIMITED"
    title = "Morphology-kinetics limited"
    assumptions = ["Thermal pathway and morphology materially affect outcome beyond bulk phase identity."]
    _variance_scale = 1.0

    def latent_state(self, features: np.ndarray) -> dict[str, float]:
        temp = float(features[1]) if len(features) > 1 else 0.5
        duration = float(features[2]) if len(features) > 2 else 0.5
        kinetic_trapping = np.clip(1.0 - abs(temp - 0.65) - 0.6 * abs(duration - 0.65), 0.0, 1.0)
        morphology = np.clip(0.35 + 0.60 * kinetic_trapping, 0.0, 1.0)
        completion = np.clip(0.40 + 0.45 * kinetic_trapping, 0.0, 1.0)
        return {"phase_purity": 0.63, "composition_homogeneity": 0.64, "morphology_quality": float(morphology), "reaction_completion": float(completion), "kinetic_trapping": float(1.0 - kinetic_trapping)}

    def falsification_signature(self) -> dict[str, list[str]]:
        return {
            "strongly_supporting_patterns": ["SEM morphology tracks outcome", "process window predicts completion", "phase identity remains similar"],
            "strongly_falsifying_patterns": ["morphology changes without outcome effect", "outcome follows phase purity while process is held fixed"],
            "ambiguous_patterns": ["low-resolution SEM with no process metadata"],
        }


def build_alab_multimodal_hypotheses() -> dict[str, MultimodalScientificHypothesis]:
    return {
        h.hypothesis_id: h
        for h in (
            PhasePurityLimitedHypothesis(),
            CompositionHomogeneityLimitedHypothesis(),
            MorphologyKineticsLimitedHypothesis(),
        )
    }


__all__ = [
    "CompositionHomogeneityLimitedHypothesis",
    "MorphologyKineticsLimitedHypothesis",
    "MultimodalScientificHypothesis",
    "PhasePurityLimitedHypothesis",
    "build_alab_multimodal_hypotheses",
]
