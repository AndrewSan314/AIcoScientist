from __future__ import annotations

import pandas as pd

from src.process.evaluation import reveal_one


def test_replay_reveals_exactly_one_hidden_source_outcome() -> None:
    observed = pd.DataFrame([{"recipe_id": "a", "capacity": 1.0}])
    hidden = pd.DataFrame([{"recipe_id": "b", "capacity": 2.0}])
    next_observed, next_hidden, step = reveal_one(observed, hidden, recipe_id="b", id_column="recipe_id", target="capacity")
    assert len(next_observed) == 2 and next_hidden.empty and step.revealed_target == 2.0
