from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


class MaskedMultiOutputRegressor:
    """One auditable tree model per available target; absent labels are never imputed."""

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = random_state
        self.models: dict[str, ExtraTreesRegressor] = {}

    def fit(self, X: np.ndarray, targets: Mapping[str, np.ndarray]) -> "MaskedMultiOutputRegressor":
        features = np.asarray(X, dtype=float)
        self.models = {}
        for name, raw in targets.items():
            y = np.asarray(raw, dtype=float)
            mask = np.isfinite(y)
            if mask.sum() < 2:
                continue
            model = ExtraTreesRegressor(n_estimators=200, random_state=self.random_state, n_jobs=1)
            model.fit(features[mask], y[mask])
            self.models[name] = model
        if not self.models:
            raise ValueError("at least one target requires two observed labels")
        return self

    def predict_distribution(self, X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        features = np.asarray(X, dtype=float)
        return {
            name: (np.asarray([tree.predict(features) for tree in model.estimators_]).mean(axis=0), np.asarray([tree.predict(features) for tree in model.estimators_]).std(axis=0))
            for name, model in self.models.items()
        }
