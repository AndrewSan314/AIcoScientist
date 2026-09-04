from __future__ import annotations

from typing import Any

import numpy as np


class GPaxBackend:
    """Lazy optional GPax adapter; GPax remains downstream of AIcoScientist policy."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.model: Any | None = None
        self._gpax: Any | None = None

    @property
    def available(self) -> bool:
        try:
            import gpax  # type: ignore
        except ImportError:
            return False
        self._gpax = gpax
        return True

    def _require(self) -> Any:
        if not self.available:
            raise ImportError("GPax is optional; install the hypothesis-gp extra to use GPaxBackend")
        return self._gpax

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        gpax = self._require()
        model_cls = getattr(gpax, "ExactGP", None)
        if model_cls is None:
            raise ImportError("installed GPax does not expose ExactGP")
        self.model = model_cls(np.asarray(X).shape[1], kernel=kwargs.get("kernel", "RBF"))
        raise RuntimeError("GPaxBackend requires an explicit JAX/NumPyro inference configuration; no hidden training is performed")

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("GPaxBackend is not fitted")
        raise NotImplementedError("Use the configured GPax model's predictive API through a domain adapter")

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        mean, variance = self.predict_distribution(X)
        obs = np.asarray(y, dtype=np.float64)
        return -0.5 * (np.log(2 * np.pi * variance) + ((obs - mean) ** 2) / variance)

    def sample_predictive(self, X: np.ndarray, n_samples: int = 1, seed: int | None = None) -> np.ndarray:
        mean, variance = self.predict_distribution(X)
        return np.random.default_rng(seed).normal(mean, np.sqrt(variance), size=(n_samples, len(mean)))

    def diagnostics(self) -> dict[str, Any]:
        return {"backend": "gpax", "available": self.available, "fitted": self.model is not None, "fallback": "sklearn_gaussian_process"}


__all__ = ["GPaxBackend"]
