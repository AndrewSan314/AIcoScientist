from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.science.multimodal.ontology import observable_names_for_modality


@dataclass(frozen=True)
class PredictiveObservableDistribution:
    """Diagonal-Gaussian predictive distribution for scalar or vector observables.

    ``variance`` is the total predictive observation variance.  The optional
    ``measurement_uncertainty`` accepted by :meth:`log_pdf` is an additional
    measurement-error term and must not repeat variance already represented by
    the predictive distribution.
    """

    hypothesis_id: str
    candidate_id: str
    modality: str
    mean: np.ndarray
    variance: np.ndarray
    observable_names: tuple[str, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    distribution_kind: str = "gaussian"
    categories: tuple[str, ...] = ()
    probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        mean = np.atleast_1d(np.asarray(self.mean, dtype=np.float64))
        variance = np.atleast_1d(np.asarray(self.variance, dtype=np.float64))
        if mean.shape != variance.shape:
            raise ValueError("mean and variance must have the same shape")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("predictive moments must be finite")
        if np.any(variance <= 0):
            raise ValueError("predictive variance must be positive")
        names = tuple(self.observable_names or ())
        if not names:
            try:
                registered = observable_names_for_modality(self.modality)
            except ValueError:
                registered = ()
            names = registered if len(registered) == len(mean) else tuple(
                f"{str(self.modality).upper()}.value_{i}" for i in range(len(mean))
            )
        if len(names) != len(mean) or len(set(names)) != len(names):
            raise ValueError("observable_names must have one unique name per predictive output")
        if self.distribution_kind not in {"gaussian", "categorical"}:
            raise ValueError("distribution_kind must be gaussian or categorical")
        probs = None
        categories = tuple(self.categories)
        if self.distribution_kind == "categorical":
            if not categories:
                raise ValueError("categorical distributions require categories")
            probs = np.asarray(self.probabilities, dtype=np.float64)
            if probs.shape != (len(categories),) or not np.all(np.isfinite(probs)) or np.any(probs < 0) or not np.sum(probs) > 0:
                raise ValueError("categorical probabilities must be finite and non-negative")
            probs = probs / np.sum(probs)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "observable_names", names)
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "probabilities", probs)

    def sample(self, rng: np.random.Generator | None = None) -> Any:
        generator = rng or np.random.default_rng()
        if self.distribution_kind == "categorical":
            return self.categories[int(generator.choice(len(self.categories), p=self.probabilities))]
        return generator.normal(self.mean, np.sqrt(self.variance))

    def log_pdf(
        self,
        observation: Any,
        *,
        observed_names: tuple[str, ...] | None = None,
        measurement_uncertainty: Any | None = None,
    ) -> float:
        if observed_names is not None and tuple(observed_names) != self.observable_names:
            raise ValueError(
                f"observable schema mismatch: expected {self.observable_names}, got {tuple(observed_names)}"
            )
        if self.distribution_kind == "categorical":
            value = observation
            if isinstance(value, Mapping):
                value = value.get("reaction_category", value.get("value"))
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.reshape(-1)[0]
            if value not in self.categories:
                raise ValueError(f"unknown categorical observation {value!r}")
            return float(np.log(max(float(self.probabilities[self.categories.index(value)]), 1e-300)))
        obs = np.atleast_1d(np.asarray(observation, dtype=np.float64))
        if obs.shape != self.mean.shape:
            raise ValueError(f"observation dimension mismatch: expected {self.mean.shape}, got {obs.shape}")
        if not np.all(np.isfinite(obs)):
            raise ValueError("observation must be finite")
        effective_variance = self.variance.copy()
        if measurement_uncertainty is not None:
            errors = np.asarray(measurement_uncertainty, dtype=np.float64)
            if errors.ndim == 0:
                errors = np.full_like(self.mean, float(errors))
            else:
                errors = np.atleast_1d(errors)
            if errors.shape != self.mean.shape or not np.all(np.isfinite(errors)) or np.any(errors < 0):
                raise ValueError("measurement uncertainty must match the prediction and be finite/non-negative")
            effective_variance += errors**2
        value = -0.5 * np.sum(
            np.log(2.0 * np.pi * effective_variance) + ((obs - self.mean) ** 2) / effective_variance
        )
        if not np.isfinite(value):
            raise ValueError("non-finite Gaussian log likelihood")
        return float(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "candidate_id": self.candidate_id,
            "modality": self.modality,
            "observable_names": list(self.observable_names),
            "mean": self.mean.tolist(),
            "variance": self.variance.tolist(),
            "distribution_kind": self.distribution_kind,
            "categories": list(self.categories),
            "probabilities": None if self.probabilities is None else self.probabilities.tolist(),
            "metadata": dict(self.metadata),
        }


__all__ = ["PredictiveObservableDistribution"]
