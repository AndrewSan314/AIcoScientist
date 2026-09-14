"""Tests verifying recipe/protocol grouping across replicates."""

from __future__ import annotations
import pytest
import pandas as pd
from src.process.benchmarks.rediscovery import RecipeAggregation

def test_aggregate_replicates_correctness() -> None:
    df = pd.DataFrame([
        {"candidate_id": "r1", "speed": 0.2, "gap": 100.0, "d30": 400.0},
        {"candidate_id": "r2", "speed": 0.2, "gap": 100.0, "d30": 404.0},
        {"candidate_id": "r3", "speed": 0.4, "gap": 150.0, "d30": 350.0},
    ])
    agg = RecipeAggregation.aggregate_recipes(
        df,
        control_columns=["speed", "gap"],
        target_column="d30",
        id_column="candidate_id",
        agg="mean",
    )
    assert len(agg) == 2
    r1 = agg[agg["speed"] == 0.2].iloc[0]
    assert pytest.approx(r1["d30"], abs=1e-5) == 402.0
