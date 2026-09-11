from __future__ import annotations

import pandas as pd

from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.optimization.process_space import ProcessSearchSpace


def test_process_space_proposes_only_exact_recorded_recipes() -> None:
    space = ProcessSearchSpace.from_finite_pool(pd.DataFrame([{"recipe_id": "a", "temperature": 100}, {"recipe_id": "b", "temperature": 120}]))
    assert space.validate_recipe({"temperature": 100})
    assert not space.validate_recipe({"temperature": 110})


def test_official_process_scaling_uses_known_controls_without_changing_recipe_ids() -> None:
    space = ProcessSearchSpace(pd.DataFrame({"recipe_id": ["a", "b"], "speed": [10.0, 20.0], "gap": [0.1, 0.1]}))
    observations = pd.DataFrame({"recipe_id": ["a"], "speed": [10.0], "gap": [0.1], "target": [1.0]})

    scaled_observations, scaled_space = ProcessOptimizationCoordinator._scale_official_botorch_inputs(observations, space)

    assert scaled_space.candidates["recipe_id"].tolist() == ["a", "b"]
    assert scaled_space.candidates["speed"].tolist() == [0.0, 1.0]
    assert scaled_space.candidates["gap"].tolist() == [0.0, 0.0]
    assert scaled_observations.loc[0, "target"] == 1.0
