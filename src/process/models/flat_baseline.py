from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


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
