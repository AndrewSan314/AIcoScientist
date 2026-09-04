from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ProbabilisticHypothesisBackend(Protocol):
    """Backend contract; it supplies distributions but never selects experiments."""

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        ...

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...

    def sample_predictive(self, X: np.ndarray, n_samples: int = 1, seed: int | None = None) -> np.ndarray:
        ...

    def diagnostics(self) -> dict[str, Any]:
        ...


__all__ = ["ProbabilisticHypothesisBackend"]
