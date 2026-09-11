from __future__ import annotations

import pandas as pd

from src.process.optimization.process_space import ProcessSearchSpace


def test_recipe_identity_is_stable_across_reconstruction() -> None:
    frame = pd.DataFrame([{"recipe_id": "r1", "pressure": 1.0}])
    assert ProcessSearchSpace.from_finite_pool(frame).recipe("r1") == ProcessSearchSpace.from_finite_pool(frame.copy()).recipe("r1")
