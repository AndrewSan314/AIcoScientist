from __future__ import annotations

import numpy as np

from src.process.fusion.encoders import ImageFeatureEncoder, SignalFeatureEncoder, TabularEncoder


def test_baseline_encoders_emit_finite_features() -> None:
    assert TabularEncoder().encode({"a": 1.0}).shape == (1,)
    assert SignalFeatureEncoder().encode([1, 2, 3]).shape == (6,)
    assert np.isfinite(ImageFeatureEncoder().encode(np.ones((2, 2)))).all()
