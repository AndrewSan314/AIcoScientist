from __future__ import annotations

import pandas as pd

from src.process.optimization.process_space import ProcessSearchSpace


def test_process_space_proposes_only_exact_recorded_recipes() -> None:
    space = ProcessSearchSpace.from_finite_pool(pd.DataFrame([{"recipe_id": "a", "temperature": 100}, {"recipe_id": "b", "temperature": 120}]))
    assert space.validate_recipe({"temperature": 100})
    assert not space.validate_recipe({"temperature": 110})
