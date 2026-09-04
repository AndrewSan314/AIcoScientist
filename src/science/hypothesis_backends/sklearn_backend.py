from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


class SklearnGaussianBackend:
    """CPU-safe probabilistic backend used when optional GPax is unavailable."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: GaussianProcessRegressor | None = None
        self.noise_variance = 1e-4

    def fit(self, X: np.ndarray, y: np.ndarray, **_: Any) -> None:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(X_arr) != len(y_arr) or len(y_arr) < 2:
            raise ValueError("X and y need at least two aligned observations")
        self.model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.05),
            normalize_y=True,
            random_state=self.random_state,
            n_restarts_optimizer=0,
        )
        self.model.fit(X_arr, y_arr)
        self.noise_variance = float(max(self.model.kernel_.k2.noise_level if hasattr(self.model.kernel_, "k2") else 1e-4, 1e-8))

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("backend is not fitted")
        mean, std = self.model.predict(np.asarray(X, dtype=np.float64), return_std=True)
        return np.asarray(mean), np.maximum(np.asarray(std) ** 2, 1e-8)

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        mean, variance = self.predict_distribution(X)
        obs = np.asarray(y, dtype=np.float64).reshape(-1)
        return -0.5 * (np.log(2 * np.pi * variance) + ((obs - mean) ** 2) / variance)

    def sample_predictive(self, X: np.ndarray, n_samples: int = 1, seed: int | None = None) -> np.ndarray:
        mean, variance = self.predict_distribution(X)
        return np.random.default_rng(seed).normal(mean, np.sqrt(variance), size=(n_samples, len(mean)))

    def diagnostics(self) -> dict[str, Any]:
        return {"backend": "sklearn_gaussian_process", "fitted": self.model is not None, "noise_variance": self.noise_variance}


__all__ = ["SklearnGaussianBackend"]
