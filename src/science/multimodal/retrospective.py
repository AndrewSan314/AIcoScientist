"""Leakage-safe retrospective hypothesis calibration for real A-Lab observations.

The controlled hypotheses intentionally remain in ``hypotheses.py``.  This module
contains the separate, fitted model used for retrospective replay and its
evaluation metrics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from src.science.multimodal.measurement_models import PredictiveObservableDistribution
from src.science.multimodal.ontology import OBSERVABLE_REGISTRY, observable_names_for_modality
from src.science.multimodal.schemas import ScientificObservable
from src.domains.alab.chemistry import parse_chemical_formula


REAL_MODALITIES = ("XRD", "REFINEMENT", "OUTCOME_TEST", "SEM", "EDS")
OUTCOME_CATEGORIES = ("completely_reacted", "transformed", "partially_reacted", "unreacted")
OUTCOME_UTILITIES = {
    "completely_reacted": 1.0,
    "transformed": 0.75,
    "partially_reacted": 0.5,
    "unreacted": 0.0,
}
_Z50 = NormalDist().inv_cdf(0.75)
_Z90 = NormalDist().inv_cdf(0.95)
CALIBRATION_ACCEPTANCE_THRESHOLDS = {
    "coverage50_abs_error_max": 0.15,
    "coverage90_abs_error_max": 0.10,
}


def _id_hash(ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(item) for item in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_numeric_observation(value: Any, names: tuple[str, ...]) -> np.ndarray:
    if isinstance(value, ScientificObservable):
        if tuple(value.observable_names) != names:
            raise ValueError(f"observation schema mismatch: expected {names}, got {value.observable_names}")
        value = value.value
    elif isinstance(value, Mapping):
        value = value.get("value")
    result = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if result.shape != (len(names),) or not np.all(np.isfinite(result)):
        raise ValueError(f"observation must be finite with shape {(len(names),)}, got {result.shape}")
    return result


def _observed_context_value(candidate_id: str, modality: str, context: Mapping[str, Any] | None) -> np.ndarray | None:
    if not context:
        return None
    values = context.get(modality)
    if not isinstance(values, Mapping) or candidate_id not in values:
        return None
    raw = values[candidate_id]
    if isinstance(raw, ScientificObservable):
        raw = raw.value
    try:
        result = np.atleast_1d(np.asarray(raw, dtype=np.float64))
    except (TypeError, ValueError):
        return None
    return result if np.all(np.isfinite(result)) else None


@dataclass(frozen=True)
class RetrospectiveObservationSet:
    """Canonical observations indexed by modality and candidate ID."""

    by_modality: Mapping[str, Mapping[str, ScientificObservable]]
    coverage: Mapping[str, Any]


def canonical_formula(formula: str) -> str:
    """Canonical text key for grouping formulas without treating spelling as chemistry."""
    try:
        counts = parse_chemical_formula(str(formula))
    except (TypeError, ValueError):
        return "".join(str(formula).split())
    parts = []
    for element, count in sorted(counts.items()):
        rendered = str(int(round(count))) if abs(count - round(count)) < 1e-9 else f"{count:.8g}"
        parts.append(f"{element}{rendered}")
    return "".join(parts)


def reaction_signature(target_formula: str, precursor_formulas: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    return canonical_formula(target_formula), tuple(sorted(canonical_formula(item) for item in precursor_formulas if str(item).strip()))


def build_group_holdout_protocols(sample_metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build deterministic sample, reaction-signature, and target holdouts."""
    ids = sorted(str(item) for item in sample_metadata)
    descriptors = {
        "SAMPLE_ID_INTERPOLATION_HOLDOUT": {cid: cid for cid in ids},
        "REACTION_SIGNATURE_GROUP_HOLDOUT": {
            cid: json.dumps(
                reaction_signature(
                    str(sample_metadata[cid].get("target_compound", "")),
                    tuple(sample_metadata[cid].get("precursor_formulas", ())),
                ),
                separators=(",", ":"),
            )
            for cid in ids
        },
        "TARGET_COMPOUND_GROUP_HOLDOUT": {
            cid: canonical_formula(str(sample_metadata[cid].get("target_compound", "")))
            for cid in ids
        },
    }
    protocols: dict[str, dict[str, Any]] = {}
    for name, group_by_id in descriptors.items():
        groups: dict[str, list[str]] = {}
        for cid, group in group_by_id.items():
            groups.setdefault(group, []).append(cid)
        calibration_groups = {
            group for group in groups
            if int(hashlib.sha256(group.encode("utf-8")).hexdigest()[-1], 16) % 2 == 0
        }
        evaluation_groups = set(groups) - calibration_groups
        if not calibration_groups or not evaluation_groups:
            ordered = sorted(groups)
            calibration_groups = set(ordered[::2])
            evaluation_groups = set(ordered[1::2])
        calibration_ids = sorted(cid for group in calibration_groups for cid in groups[group])
        evaluation_ids = sorted(cid for group in evaluation_groups for cid in groups[group])
        protocols[name] = {
            "split_protocol": name,
            "group_key": "sample_id" if name.startswith("SAMPLE") else "reaction_signature" if name.startswith("REACTION") else "target_compound",
            "calibration_ids": calibration_ids,
            "evaluation_ids": evaluation_ids,
            "calibration_groups": sorted(calibration_groups),
            "evaluation_groups": sorted(evaluation_groups),
            "calibration_count": len(calibration_ids),
            "evaluation_count": len(evaluation_ids),
            "group_overlap": sorted(calibration_groups.intersection(evaluation_groups)),
            "target_overlap": sorted(
                {descriptors["TARGET_COMPOUND_GROUP_HOLDOUT"][cid] for cid in calibration_ids}
                .intersection(descriptors["TARGET_COMPOUND_GROUP_HOLDOUT"][cid] for cid in evaluation_ids)
            ),
            "precursor_signature_overlap": sorted(
                {descriptors["REACTION_SIGNATURE_GROUP_HOLDOUT"][cid] for cid in calibration_ids}
                .intersection(descriptors["REACTION_SIGNATURE_GROUP_HOLDOUT"][cid] for cid in evaluation_ids)
            ),
            "deterministic_assignment": "SHA256(group_key) parity with deterministic alternating fallback",
        }
    return protocols


class RetrospectiveCalibratedHypothesisModel:
    """Interpretable CPU-safe model fitted exclusively on a declared ID split.

    Each hypothesis uses a fixed scientific feature family.  A standardized
    ridge regression predicts named observable vectors, with residual and
    leverage uncertainty.  The structural H2/H3 distributions are pooled
    nuisance predictions because the real dataset has no candidate-linked
    EDS/SEM evidence; they cannot create mechanistic evidence that is absent.
    The role is explicit shared nuisance evidence, never negative evidence.
    """

    model_kind = "RETROSPECTIVE_CALIBRATED_HYPOTHESIS_MODEL"
    model_version = "1.0.0"

    def __init__(
        self,
        hypothesis_id: str,
        feature_indices: Sequence[int],
        identifiability_by_modality: Mapping[str, str],
        *,
        ridge_alpha: float = 1.0,
    ) -> None:
        self.hypothesis_id = str(hypothesis_id)
        self.feature_indices = tuple(int(index) for index in feature_indices)
        self.identifiability_by_modality = {str(k).upper(): str(v) for k, v in identifiability_by_modality.items()}
        self.ridge_alpha = float(ridge_alpha)
        self._candidate_features: dict[str, np.ndarray] = {}
        self._feature_mean = np.zeros(len(self.feature_indices), dtype=np.float64)
        self._feature_scale = np.ones(len(self.feature_indices), dtype=np.float64)
        self._parameters: dict[str, dict[str, Any]] = {}
        self._pooled: dict[str, dict[str, Any]] = {}
        self._fit_ids: tuple[str, ...] = ()
        self._preprocessing_fit_ids: tuple[str, ...] = ()
        self._fitted = False
        self.training_count = 0

    @property
    def fitted_ids(self) -> tuple[str, ...]:
        return self._fit_ids

    def fit(
        self,
        candidate_features_by_id: Mapping[str, Any] | None = None,
        observed_context: Mapping[str, Any] | None = None,
        *,
        training_ids: Sequence[str] | None = None,
        **_: Any,
    ) -> None:
        """Fit once; subsequent engine lifecycle calls cannot refit parameters."""
        if self._fitted:
            if candidate_features_by_id:
                self._candidate_features = {
                    str(cid): np.asarray(values, dtype=np.float64)
                    for cid, values in candidate_features_by_id.items()
                }
            return
        if training_ids is None:
            raise ValueError("retrospective calibration requires explicit training_ids")
        training = tuple(sorted({str(item) for item in training_ids}))
        if not training:
            raise ValueError("retrospective calibration requires non-empty training_ids")
        features = {
            str(cid): np.asarray(values, dtype=np.float64)
            for cid, values in (candidate_features_by_id or {}).items()
        }
        missing_features = sorted(set(training) - set(features))
        if missing_features:
            raise ValueError(f"training features missing IDs: {missing_features[:3]}")
        self._candidate_features = features
        X_raw = np.asarray([features[cid][list(self.feature_indices)] for cid in training], dtype=np.float64)
        if X_raw.ndim != 2 or X_raw.shape[1] != len(self.feature_indices) or not np.all(np.isfinite(X_raw)):
            raise ValueError("training features must be finite and match the declared feature family")
        self._feature_mean = np.mean(X_raw, axis=0)
        self._feature_scale = np.std(X_raw, axis=0)
        self._feature_scale = np.where(self._feature_scale > 1e-12, self._feature_scale, 1.0)
        X = self._design(X_raw)
        observations = observed_context or {}
        for modality in REAL_MODALITIES:
            names = observable_names_for_modality(modality)
            rows = []
            targets = []
            for cid in training:
                raw = observations.get(modality, {}) if isinstance(observations, Mapping) else {}
                if not isinstance(raw, Mapping) or cid not in raw:
                    continue
                rows.append(X[training.index(cid)])
                targets.append(_as_numeric_observation(raw[cid], names))
            if targets:
                y = np.asarray(targets, dtype=np.float64)
                self._pooled[modality] = self._pooled_parameters(y)
                if self.identifiability_by_modality.get(modality, "").startswith("NOT_"):
                    continue
                self._parameters[modality] = self._fit_linear(np.asarray(rows), y)
        self._fit_ids = training
        self._preprocessing_fit_ids = training
        self.training_count = sum(
            len(values) for modality, values in observations.items()
            if modality in REAL_MODALITIES and isinstance(values, Mapping) and set(values).intersection(training)
        )
        self._fitted = True

    def _design(self, X_raw: np.ndarray) -> np.ndarray:
        standardized = (X_raw - self._feature_mean) / self._feature_scale
        return np.column_stack([np.ones(len(standardized)), standardized])

    def _fit_linear(self, X: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
        penalty = np.eye(X.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        gram = X.T @ X + self.ridge_alpha * penalty
        inverse = np.linalg.pinv(gram)
        coefficients = inverse @ X.T @ Y
        residuals = Y - X @ coefficients
        variance = np.maximum(np.mean(residuals**2, axis=0), 1e-4)
        return {"coefficients": coefficients, "gram_inverse": inverse, "variance": variance, "n": int(len(Y))}

    @staticmethod
    def _pooled_parameters(Y: np.ndarray) -> dict[str, Any]:
        return {
            "mean": np.mean(Y, axis=0),
            "variance": np.maximum(np.var(Y, axis=0), 0.04),
            "n": int(len(Y)),
        }

    def _features_for_prediction(self, candidate_id: str, supplied: Any | None) -> np.ndarray:
        raw = supplied if supplied is not None else self._candidate_features.get(candidate_id)
        if raw is None:
            raise KeyError(f"candidate features missing for {candidate_id}")
        values = np.asarray(raw, dtype=np.float64)
        selected = values[list(self.feature_indices)]
        if not np.all(np.isfinite(selected)):
            raise ValueError("candidate features must be finite")
        return selected

    def _base_prediction(self, candidate_id: str, modality: str, candidate_features: Any | None) -> tuple[np.ndarray, np.ndarray, str]:
        names = observable_names_for_modality(modality)
        selected = self._features_for_prediction(candidate_id, candidate_features)
        design = self._design(selected.reshape(1, -1))[0]
        parameters = self._parameters.get(modality)
        identifiability = self.identifiability_by_modality.get(modality, "NOT_IDENTIFIABLE_FROM_AVAILABLE_DATA")
        if parameters is not None:
            mean = design @ parameters["coefficients"]
            leverage = float(design @ parameters["gram_inverse"] @ design)
            variance = parameters["variance"] * max(1.0, 1.0 + leverage)
            role = "DIAGNOSTIC:fitted_hypothesis_model"
        elif modality in self._pooled:
            pooled = self._pooled[modality]
            mean = np.asarray(pooled["mean"], dtype=np.float64)
            variance = np.asarray(pooled["variance"], dtype=np.float64) * 1.5
            role = "SHARED_NUISANCE:pooled_nuisance_not_identifiable"
        else:
            mean = np.asarray([
                ((definition.value_range[0] + definition.value_range[1]) / 2.0)
                if definition.value_range is not None else 0.5
                for definition in (OBSERVABLE_REGISTRY[name] for name in names)
            ], dtype=np.float64)
            variance = np.full(len(names), 1.0, dtype=np.float64)
            role = "UNINFORMATIVE:uninformative_prior_not_evaluated"
        for index, name in enumerate(names):
            bounds = OBSERVABLE_REGISTRY[name].value_range
            if bounds is not None:
                mean[index] = np.clip(mean[index], *bounds)
        return mean, np.maximum(variance, 1e-6), f"{identifiability}:{role}"

    def modality_role(self, modality: str) -> str:
        """Return the explicit evidence role used for posterior discrimination."""
        modality = str(modality).upper()
        identifiability = self.identifiability_by_modality.get(modality, "NOT_IDENTIFIABLE_FROM_AVAILABLE_DATA")
        if identifiability.startswith("NOT_"):
            return "UNAVAILABLE" if identifiability.startswith("NOT_EVALUATED") else "SHARED_NUISANCE"
        return "DIAGNOSTIC"

    def latent_state(self, candidate_id: str, candidate_features: Any | None = None) -> dict[str, float | None]:
        outcome, _, _ = self._base_prediction(candidate_id, "OUTCOME_TEST", candidate_features)
        refinement, _, _ = self._base_prediction(candidate_id, "REFINEMENT", candidate_features)
        return {
            "phase_quality": float(np.clip(refinement[0], 0.0, 1.0)),
            "phase_purity": float(np.clip(refinement[0], 0.0, 1.0)),
            "precursor_residual": float(np.clip(refinement[1], 0.0, 1.0)),
            "structural_disorder": float(np.clip(refinement[2], 0.0, 1.0)),
            "composition_homogeneity": None,
            "morphology_quality": None,
            "synthesis_success_propensity": float(np.clip(outcome[0], 0.0, 1.0)),
        }

    def _uncertainty_components(self, candidate_id: str, modality: str, candidate_features: Any | None) -> tuple[np.ndarray, np.ndarray]:
        selected = self._features_for_prediction(candidate_id, candidate_features)
        design = self._design(selected.reshape(1, -1))[0]
        parameters = self._parameters.get(modality)
        if parameters is not None:
            observation = np.asarray(parameters["variance"], dtype=np.float64)
            epistemic = observation * max(0.0, float(design @ parameters["gram_inverse"] @ design))
            return epistemic, observation
        _, variance, _ = self._base_prediction(candidate_id, modality, candidate_features)
        return np.asarray(variance, dtype=np.float64) * (1.0 / 3.0), np.asarray(variance, dtype=np.float64) * (2.0 / 3.0)

    def predict_observable_distribution(
        self,
        candidate_id: str,
        modality: str,
        observed_context: Mapping[str, Any] | None = None,
        *,
        candidate_features: Any | None = None,
        **_: Any,
    ) -> PredictiveObservableDistribution:
        modality = str(modality).upper()
        mean, variance, role = self._base_prediction(candidate_id, modality, candidate_features)
        epistemic_variance, observation_variance = self._uncertainty_components(candidate_id, modality, candidate_features)
        conditioned_on: list[str] = []
        if modality == "REFINEMENT":
            xrd_observed = _observed_context_value(candidate_id, "XRD", observed_context)
            if xrd_observed is not None:
                xrd_mean, _, _ = self._base_prediction(candidate_id, "XRD", candidate_features)
                residual = float(xrd_observed[0] - xrd_mean[0])
                mean = mean.copy()
                mean[0] = np.clip(mean[0] + 0.30 * residual, 0.0, 1.0)
                mean[1] = np.clip(mean[1] - 0.20 * residual, 0.0, 1.0)
                conditioned_on.append("XRD")
        elif modality == "OUTCOME_TEST":
            refinement_observed = _observed_context_value(candidate_id, "REFINEMENT", observed_context)
            xrd_observed = _observed_context_value(candidate_id, "XRD", observed_context)
            mean = mean.copy()
            if refinement_observed is not None:
                refinement_mean, _, _ = self._base_prediction(candidate_id, "REFINEMENT", candidate_features)
                mean[0] = np.clip(mean[0] + 0.35 * float(refinement_observed[0] - refinement_mean[0]), 0.0, 1.0)
                conditioned_on.append("REFINEMENT")
            elif xrd_observed is not None:
                xrd_mean, _, _ = self._base_prediction(candidate_id, "XRD", candidate_features)
                mean[0] = np.clip(mean[0] + 0.10 * float(xrd_observed[0] - xrd_mean[0]), 0.0, 1.0)
                conditioned_on.append("XRD")
        return PredictiveObservableDistribution(
            hypothesis_id=self.hypothesis_id,
            candidate_id=str(candidate_id),
            modality=modality,
            mean=mean,
            variance=variance,
            observable_names=observable_names_for_modality(modality),
            metadata={
                "model_kind": self.model_kind,
                "model_version": self.model_version,
                "prediction_role": role,
                "modality_role": self.modality_role(modality),
                "likelihood_mode": "shared_nuisance" if self.modality_role(modality) == "SHARED_NUISANCE" else "mechanistic_fitted",
                "identifiability": self.identifiability_by_modality.get(modality),
                "conditioned_on": conditioned_on,
                "latent_state": self.latent_state(candidate_id, candidate_features),
                "training_ids_sha256": _id_hash(self._fit_ids),
                "uncertainty_components": {
                    "epistemic_model_variance": epistemic_variance.tolist(),
                    "observation_variance": observation_variance.tolist(),
                    "total_predictive_variance": variance.tolist(),
                },
            },
        )

    def predict_category_probabilities(self, candidate_id: str, candidate_features: Any | None = None) -> np.ndarray:
        prediction = self.predict_observable_distribution(candidate_id, "OUTCOME_TEST", candidate_features=candidate_features)
        sigma = float(np.sqrt(prediction.variance[0] + 0.05**2))
        utilities = np.asarray([OUTCOME_UTILITIES[category] for category in OUTCOME_CATEGORIES], dtype=np.float64)
        log_weights = -0.5 * ((utilities - prediction.mean[0]) / sigma) ** 2
        log_weights -= np.max(log_weights)
        probabilities = np.exp(log_weights)
        return probabilities / np.sum(probabilities)

    def log_likelihood(self, observable: ScientificObservable, observed_context: Mapping[str, Any] | None = None) -> float:
        prediction = self.predict_observable_distribution(
            observable.candidate_id,
            observable.modality,
            observed_context,
        )
        return prediction.log_pdf(
            observable.value,
            observed_names=tuple(observable.observable_names),
            measurement_uncertainty=observable.uncertainty,
        )

    def falsification_signature(self) -> dict[str, list[str]]:
        if self.hypothesis_id.startswith("H1_"):
            return {
                "strongly_supporting_patterns": ["held-out structural observables match the fitted phase-purity model"],
                "strongly_falsifying_patterns": ["held-out structural residuals exceed the model uncertainty"],
                "ambiguous_patterns": ["outcome agreement without direct phase attribution"],
            }
        if self.hypothesis_id.startswith("H2_"):
            return {
                "strongly_supporting_patterns": ["outcome proxy tracks precursor-composition features"],
                "strongly_falsifying_patterns": ["uniform outcome proxy despite composition-feature variation"],
                "ambiguous_patterns": ["no candidate-linked EDS observation"],
            }
        return {
            "strongly_supporting_patterns": ["outcome proxy tracks the process-window features"],
            "strongly_falsifying_patterns": ["outcome is insensitive to process-window features"],
            "ambiguous_patterns": ["no candidate-linked SEM observation"],
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "model_kind": self.model_kind,
            "model_version": self.model_version,
            "feature_indices": list(self.feature_indices),
            "feature_preprocessing": "standardization fit on calibration IDs only",
            "ridge_alpha": self.ridge_alpha,
            "fit_id_count": len(self._fit_ids),
            "fit_ids_sha256": _id_hash(self._fit_ids),
            "preprocessing_fit_ids_sha256": _id_hash(self._preprocessing_fit_ids),
            "training_count": self.training_count,
            "identifiability_by_modality": dict(self.identifiability_by_modality),
            "modality_roles": {modality: self.modality_role(modality) for modality in REAL_MODALITIES},
            "fitted_modalities": sorted(self._parameters),
            "pooled_nuisance_modalities": sorted(self._pooled),
        }


class RetrospectiveDiscoveryModel:
    """Calibration-only predictor for pre-reveal reaction-outcome utility."""

    model_kind = "RETROSPECTIVE_CALIBRATED_DISCOVERY_MODEL"
    model_version = "1.0.0"

    def __init__(self, feature_indices: Sequence[int] = tuple(range(49)), ridge_alpha: float = 1.0) -> None:
        self.feature_indices = tuple(int(index) for index in feature_indices)
        self.ridge_alpha = float(ridge_alpha)
        self._mean = np.zeros(len(self.feature_indices), dtype=np.float64)
        self._scale = np.ones(len(self.feature_indices), dtype=np.float64)
        self._coefficients = np.zeros(len(self.feature_indices) + 1, dtype=np.float64)
        self._variance = 1.0
        self._fit_ids: tuple[str, ...] = ()
        self._candidate_features: dict[str, np.ndarray] = {}
        self._fitted = False

    @property
    def fitted_ids(self) -> tuple[str, ...]:
        return self._fit_ids

    def fit(
        self,
        candidate_features_by_id: Mapping[str, Any],
        observations: Mapping[str, Mapping[str, ScientificObservable]],
        training_ids: Sequence[str],
    ) -> None:
        if self._fitted:
            self._candidate_features = {str(cid): np.asarray(values, dtype=np.float64) for cid, values in candidate_features_by_id.items()}
            return
        training = tuple(sorted({str(item) for item in training_ids}))
        outcomes = observations.get("OUTCOME_TEST", {})
        usable = [cid for cid in training if cid in candidate_features_by_id and cid in outcomes]
        if not usable:
            raise ValueError("discovery calibration requires training outcomes")
        self._candidate_features = {str(cid): np.asarray(values, dtype=np.float64) for cid, values in candidate_features_by_id.items()}
        raw = np.asarray([self._candidate_features[cid][list(self.feature_indices)] for cid in usable], dtype=np.float64)
        target = np.asarray([_as_numeric_observation(outcomes[cid], observable_names_for_modality("OUTCOME_TEST"))[0] for cid in usable], dtype=np.float64)
        self._mean = np.mean(raw, axis=0)
        self._scale = np.where(np.std(raw, axis=0) > 1e-12, np.std(raw, axis=0), 1.0)
        design = np.column_stack([np.ones(len(raw)), (raw - self._mean) / self._scale])
        penalty = np.eye(design.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        inverse = np.linalg.pinv(design.T @ design + self.ridge_alpha * penalty)
        self._coefficients = inverse @ design.T @ target
        residuals = target - design @ self._coefficients
        self._variance = float(max(np.mean(residuals ** 2), 1e-4))
        self._fit_ids = training
        self._fitted = True

    def predict(self, candidate_id: str, candidate_features: Any | None = None) -> float:
        raw = np.asarray(candidate_features if candidate_features is not None else self._candidate_features[candidate_id], dtype=np.float64)
        selected = raw[list(self.feature_indices)]
        design = np.concatenate(([1.0], (selected - self._mean) / self._scale))
        return float(np.clip(design @ self._coefficients, 0.0, 1.0))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "model_version": self.model_version,
            "feature_indices": list(self.feature_indices),
            "fit_id_count": len(self._fit_ids),
            "fit_ids_sha256": _id_hash(self._fit_ids),
            "preprocessing_fit_on": "calibration_ids_only",
            "prediction_target": "reaction_outcome_utility",
            "residual_variance": self._variance,
        }


def build_retrospective_hypotheses() -> dict[str, RetrospectiveCalibratedHypothesisModel]:
    """Build fixed, interpretable competing hypotheses for retrospective fitting."""
    unavailable = "NOT_EVALUATED_INSUFFICIENT_LINKAGE"
    not_identifiable = "NOT_IDENTIFIABLE_FROM_AVAILABLE_DATA"
    return {
        "H1_PHASE_PURITY_LIMITED": RetrospectiveCalibratedHypothesisModel(
            "H1_PHASE_PURITY_LIMITED",
            (0, 1, 2),
            {"XRD": "CALIBRATED_DIRECT_STRUCTURAL", "REFINEMENT": "CALIBRATED_DIRECT_STRUCTURAL", "OUTCOME_TEST": "CALIBRATED_OUTCOME_LINKED", "SEM": unavailable, "EDS": unavailable},
        ),
        "H2_COMPOSITION_HOMOGENEITY_LIMITED": RetrospectiveCalibratedHypothesisModel(
            "H2_COMPOSITION_HOMOGENEITY_LIMITED",
            tuple(range(3, 49)),
            {"XRD": not_identifiable, "REFINEMENT": not_identifiable, "OUTCOME_TEST": "CALIBRATED_OUTCOME_PROXY_WEAK", "SEM": unavailable, "EDS": unavailable},
        ),
        "H3_MORPHOLOGY_KINETICS_LIMITED": RetrospectiveCalibratedHypothesisModel(
            "H3_MORPHOLOGY_KINETICS_LIMITED",
            (1, 2),
            {"XRD": not_identifiable, "REFINEMENT": not_identifiable, "OUTCOME_TEST": "CALIBRATED_OUTCOME_PROXY_WEAK", "SEM": unavailable, "EDS": unavailable},
        ),
    }


def assert_no_evaluation_leakage(
    models: Mapping[str, RetrospectiveCalibratedHypothesisModel],
    calibration_ids: Sequence[str],
    evaluation_ids: Sequence[str],
) -> None:
    calibration = {str(item) for item in calibration_ids}
    evaluation = {str(item) for item in evaluation_ids}
    if calibration.intersection(evaluation):
        raise AssertionError("calibration and evaluation IDs overlap")
    for model in models.values():
        fitted = set(model.fitted_ids)
        if not fitted.issubset(calibration):
            raise AssertionError(f"{model.hypothesis_id} fitted outside calibration IDs")
        if fitted.intersection(evaluation):
            raise AssertionError(f"{model.hypothesis_id} contains evaluation IDs")
        preprocessing_ids = set(model._preprocessing_fit_ids)
        if preprocessing_ids != fitted or preprocessing_ids.intersection(evaluation):
            raise AssertionError(f"{model.hypothesis_id} preprocessing contains evaluation IDs")


def _metric_record(
    model: RetrospectiveCalibratedHypothesisModel,
    modality: str,
    calibration_observations: Mapping[str, ScientificObservable],
    evaluation_observations: Mapping[str, ScientificObservable],
    candidate_features: Mapping[str, Any],
) -> dict[str, Any]:
    identifiability = model.identifiability_by_modality.get(modality, "NOT_IDENTIFIABLE_FROM_AVAILABLE_DATA")
    base = {
        "status": identifiability,
        "identifiability": identifiability,
        "N_calibration": len(calibration_observations),
        "N_evaluation": len(evaluation_observations),
        "MAE": None,
        "RMSE": None,
        "mean_log_predictive_density": None,
        "coverage_50": None,
        "coverage_90": None,
        "calibration_error": None,
        "Brier": None,
        "log_loss": None,
        "per_observable": {},
        "calibration_coverage_status": "NOT_EVALUATED",
    }
    if identifiability.startswith("NOT_") or not evaluation_observations:
        if not evaluation_observations and not identifiability.startswith("NOT_"):
            base["status"] = "NOT_EVALUATED_INSUFFICIENT_LINKAGE"
        return base
    names = observable_names_for_modality(modality)
    errors = []
    squared_errors = []
    log_densities = []
    covered_50 = []
    covered_90 = []
    category_brier = []
    category_log_loss = []
    per_observable: dict[str, dict[str, Any]] = {
        name: {"MAE": [], "RMSE": [], "NLL": [], "coverage_50": [], "coverage_90": []}
        for name in observable_names_for_modality(modality)
    }
    calibration_targets = []
    for observed in calibration_observations.values():
        calibration_targets.append(_as_numeric_observation(observed, names=observable_names_for_modality(modality)))
    calibration_array = np.asarray(calibration_targets, dtype=np.float64) if calibration_targets else np.empty((0, len(per_observable)))
    scales = np.ptp(calibration_array, axis=0) if len(calibration_array) else np.ones(len(per_observable))
    scales = np.where(scales > 1e-12, scales, np.std(calibration_array, axis=0) if len(calibration_array) else 1.0)
    scales = np.where(np.asarray(scales) > 1e-12, scales, 1.0)
    names = observable_names_for_modality(modality)
    for cid, observed in sorted(evaluation_observations.items()):
        prediction = model.predict_observable_distribution(cid, modality, candidate_features=candidate_features[cid])
        target = _as_numeric_observation(observed, names)
        residual = target - prediction.mean
        std = np.sqrt(prediction.variance)
        errors.extend(np.abs(residual).tolist())
        squared_errors.extend((residual**2).tolist())
        log_densities.append(prediction.log_pdf(target, observed_names=names, measurement_uncertainty=observed.uncertainty))
        covered_50.extend((np.abs(residual) <= _Z50 * std).tolist())
        covered_90.extend((np.abs(residual) <= _Z90 * std).tolist())
        measurement_uncertainty = np.atleast_1d(np.asarray(observed.uncertainty, dtype=np.float64))
        if measurement_uncertainty.size == 1:
            measurement_uncertainty = np.full(len(names), float(measurement_uncertainty[0]))
        total_variance = prediction.variance + measurement_uncertainty ** 2
        per_nll = 0.5 * (np.log(2.0 * np.pi * total_variance) + residual ** 2 / total_variance)
        for index, name in enumerate(names):
            row = per_observable[name]
            row["MAE"].append(float(abs(residual[index])))
            row["RMSE"].append(float(residual[index] ** 2))
            row["NLL"].append(float(per_nll[index]))
            row["coverage_50"].append(bool(abs(residual[index]) <= _Z50 * std[index]))
            row["coverage_90"].append(bool(abs(residual[index]) <= _Z90 * std[index]))
        if modality == "OUTCOME_TEST":
            category = str(observed.provenance.get("reaction_category", ""))
            if category in OUTCOME_CATEGORIES:
                probabilities = model.predict_category_probabilities(cid, candidate_features[cid])
                one_hot = np.asarray([float(item == category) for item in OUTCOME_CATEGORIES])
                category_brier.append(float(np.mean((probabilities - one_hot) ** 2)))
                category_log_loss.append(float(-np.log(max(probabilities[OUTCOME_CATEGORIES.index(category)], 1e-12))))
    base.update({
        "status": identifiability,
        "MAE": float(np.mean(errors)),
        "RMSE": float(np.sqrt(np.mean(squared_errors))),
        "mean_log_predictive_density": float(np.mean(log_densities)),
        "coverage_50": float(np.mean(covered_50)),
        "coverage_90": float(np.mean(covered_90)),
        "calibration_error": float(0.5 * abs(float(np.mean(covered_50)) - 0.5) + 0.5 * abs(float(np.mean(covered_90)) - 0.9)),
        "per_observable": {
            name: {
                "MAE": float(np.mean(row["MAE"])),
                "RMSE": float(np.sqrt(np.mean(row["RMSE"]))),
                "NLL": float(np.mean(row["NLL"])),
                "coverage50": float(np.mean(row["coverage_50"])),
                "coverage90": float(np.mean(row["coverage_90"])),
                "calibration_error": float(0.5 * abs(float(np.mean(row["coverage_50"])) - 0.5) + 0.5 * abs(float(np.mean(row["coverage_90"])) - 0.9)),
                "normalization": {"method": "calibration_range_then_std", "scale": float(scales[index])},
                "NRMSE": float(np.sqrt(np.mean(row["RMSE"])) / scales[index]),
            }
            for index, (name, row) in enumerate(per_observable.items())
        },
    })
    base["calibration_coverage_status"] = (
        "CALIBRATION_COVERAGE_PASS"
        if abs(float(np.mean(covered_50)) - 0.5) <= CALIBRATION_ACCEPTANCE_THRESHOLDS["coverage50_abs_error_max"]
        and abs(float(np.mean(covered_90)) - 0.9) <= CALIBRATION_ACCEPTANCE_THRESHOLDS["coverage90_abs_error_max"]
        else "CALIBRATION_COVERAGE_FAIL"
    )
    if category_brier:
        base["Brier"] = float(np.mean(category_brier))
        base["log_loss"] = float(np.mean(category_log_loss))
    return base


def evaluate_retrospective_models(
    models: Mapping[str, RetrospectiveCalibratedHypothesisModel],
    calibration: RetrospectiveObservationSet,
    evaluation: RetrospectiveObservationSet,
    candidate_features: Mapping[str, Any],
    calibration_ids: Sequence[str],
    evaluation_ids: Sequence[str],
) -> dict[str, Any]:
    assert_no_evaluation_leakage(models, calibration_ids, evaluation_ids)
    per_hypothesis_modality: dict[str, dict[str, Any]] = {}
    for hypothesis_id, model in models.items():
        per_hypothesis_modality[hypothesis_id] = {}
        for modality in REAL_MODALITIES:
            train_obs = calibration.by_modality.get(modality, {})
            eval_obs = evaluation.by_modality.get(modality, {})
            per_hypothesis_modality[hypothesis_id][modality] = _metric_record(
                model, modality, train_obs, eval_obs, candidate_features,
            )
    supported_records = [
        record
        for modality_records in per_hypothesis_modality.values()
        for modality, record in modality_records.items()
        if not str(record.get("identifiability", "")).startswith("NOT_") and record.get("N_evaluation", 0) > 0
    ]
    coverage_pass = bool(supported_records) and all(
        record.get("calibration_coverage_status") == "CALIBRATION_COVERAGE_PASS"
        for record in supported_records
    )
    return {
        "status": "A_LAB_MODELS_EVALUATED",
        "retrospective_model_evaluation_status": "A_LAB_MODELS_EVALUATED",
        "calibration_coverage_status": "A_LAB_CALIBRATION_COVERAGE_PASS" if coverage_pass else "A_LAB_CALIBRATION_PARTIAL",
        "calibration_acceptance_thresholds": dict(CALIBRATION_ACCEPTANCE_THRESHOLDS),
        "model_kind": RetrospectiveCalibratedHypothesisModel.model_kind,
        "model_version": RetrospectiveCalibratedHypothesisModel.model_version,
        "split": {
            "group_key": "sample_id",
            "calibration_n": len(set(calibration_ids)),
            "evaluation_n": len(set(evaluation_ids)),
            "calibration_ids_sha256": _id_hash(calibration_ids),
            "evaluation_ids_sha256": _id_hash(evaluation_ids),
            "disjoint": set(calibration_ids).isdisjoint(evaluation_ids),
        },
        "fit_contract": {
            "preprocessing_fit_on": "calibration_ids_only",
            "parameters_fit_on": "calibration_observations_only",
            "variance_fit_on": "calibration_residuals_only",
            "thresholds_fit_on": False,
            "calibration_curves_fit_on": False,
            "evaluation_used_for_model_selection": False,
        },
        "hypotheses": {hypothesis_id: model.diagnostics() for hypothesis_id, model in models.items()},
        "per_hypothesis_modality": per_hypothesis_modality,
        "coverage": {"calibration": dict(calibration.coverage), "evaluation": dict(evaluation.coverage)},
    }


__all__ = [
    "OUTCOME_CATEGORIES",
    "REAL_MODALITIES",
    "CALIBRATION_ACCEPTANCE_THRESHOLDS",
    "RetrospectiveCalibratedHypothesisModel",
    "RetrospectiveDiscoveryModel",
    "RetrospectiveObservationSet",
    "assert_no_evaluation_leakage",
    "build_group_holdout_protocols",
    "build_retrospective_hypotheses",
    "canonical_formula",
    "evaluate_retrospective_models",
    "reaction_signature",
]
