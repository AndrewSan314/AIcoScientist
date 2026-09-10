from __future__ import annotations

import numpy as np

from src.process.models.multioutput import MaskedMultiOutputRegressor


def test_multioutput_preserves_per_target_missing_masks() -> None:
    model = MaskedMultiOutputRegressor().fit(np.arange(12).reshape(6, 2), {"capacity": np.arange(6, dtype=float), "impedance": np.array([1, np.nan, 3, 4, 5, 6])})
    assert set(model.predict_distribution([[1, 2]])) == {"capacity", "impedance"}
