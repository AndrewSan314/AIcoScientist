from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


class TreeEnsembleBaseline:
    """Strong small-data scalar baseline with ensemble predictive spread."""

    def __init__(self, kind: str = "extra_trees", *, n_estimators: int = 200, random_state: int = 42) -> None:
        model = ExtraTreesRegressor if kind == "extra_trees" else RandomForestRegressor if kind == "random_forest" else None
        if model is None:
            raise ValueError("kind must be extra_trees or random_forest")
        self.model = model(n_estimators=n_estimators, random_state=random_state, n_jobs=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeEnsembleBaseline":
        self.model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self.model, "estimators_"):
            raise RuntimeError("fit must be called before prediction")
        predictions = np.asarray([tree.predict(X) for tree in self.model.estimators_], dtype=float)
        return predictions.mean(axis=0), predictions.std(axis=0, ddof=0)


class GaussianProcessBaseline:
    """Small-data process GP; scales controls using the training fold only."""

    def __init__(self, *, random_state: int = 42) -> None:
        self.scaler = StandardScaler()
        self.model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.05),
            normalize_y=True,
            random_state=random_state,
            n_restarts_optimizer=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessBaseline":
        features, targets = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if len(features) < 2 or len(features) != len(targets):
            raise ValueError("X and y need at least two aligned observations")
        self.model.fit(self.scaler.fit_transform(features), targets)
        return self

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self.model, "kernel_"):
            raise RuntimeError("fit must be called before prediction")
        mean, std = self.model.predict(self.scaler.transform(np.asarray(X, dtype=float)), return_std=True)
        return np.asarray(mean), np.maximum(np.asarray(std), 1e-8)
