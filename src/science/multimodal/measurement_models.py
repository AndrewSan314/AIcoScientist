from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PredictiveObservableDistribution:
    """Diagonal-Gaussian predictive distribution for scalar or vector observables."""

    hypothesis_id: str
    candidate_id: str
    modality: str
    mean: np.ndarray
    variance: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mean = np.atleast_1d(np.asarray(self.mean, dtype=np.float64))
        variance = np.atleast_1d(np.asarray(self.variance, dtype=np.float64))
        if mean.shape != variance.shape:
            raise ValueError("mean and variance must have the same shape")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("predictive moments must be finite")
        if np.any(variance <= 0):
            raise ValueError("predictive variance must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng or np.random.default_rng()
        return generator.normal(self.mean, np.sqrt(self.variance))

    def log_pdf(self, observation: Any) -> float:
        obs = np.atleast_1d(np.asarray(observation, dtype=np.float64))
        if obs.shape != self.mean.shape:
            raise ValueError(f"observation dimension mismatch: expected {self.mean.shape}, got {obs.shape}")
        value = -0.5 * np.sum(
            np.log(2.0 * np.pi * self.variance) + ((obs - self.mean) ** 2) / self.variance
        )
        return float(value) if np.isfinite(value) else -1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "candidate_id": self.candidate_id,
            "modality": self.modality,
            "mean": self.mean.tolist(),
            "variance": self.variance.tolist(),
            "metadata": dict(self.metadata),
        }


__all__ = ["PredictiveObservableDistribution"]
