"""Tests for RecipeAggregation replicate handling."""

from __future__ import annotations

import pandas as pd
import pytest

from src.process.benchmarks.rediscovery import RecipeAggregation


def test_aggregate_recipes_with_replicates() -> None:
    df = pd.DataFrame([
        {"candidate_id": "r1", "speed": 100.0, "gap": 0.2, "capacity": 8.0},
        {"candidate_id": "r2", "speed": 100.0, "gap": 0.2, "capacity": 10.0},
        {"candidate_id": "r3", "speed": 120.0, "gap": 0.3, "capacity": 12.0},
    ])
    aggregated = RecipeAggregation.aggregate_recipes(
        df,
        control_columns=["speed", "gap"],
        target_column="capacity",
        id_column="candidate_id",
        agg="mean",
    )
    assert len(aggregated) == 2
    match_100 = aggregated.loc[(aggregated["speed"] == 100.0) & (aggregated["gap"] == 0.2)]
    assert len(match_100) == 1
    assert pytest.approx(match_100.iloc[0]["capacity"]) == 9.0


def test_aggregate_recipes_unique_controls_preserved() -> None:
    df = pd.DataFrame([
        {"candidate_id": "r1", "speed": 80.0, "gap": 0.1, "capacity": 5.0},
        {"candidate_id": "r2", "speed": 100.0, "gap": 0.2, "capacity": 7.0},
        {"candidate_id": "r3", "speed": 120.0, "gap": 0.3, "capacity": 11.0},
    ])
    aggregated = RecipeAggregation.aggregate_recipes(
        df,
        control_columns=["speed", "gap"],
        target_column="capacity",
        id_column="candidate_id",
    )
    assert len(aggregated) == 3


def test_aggregate_recipes_empty() -> None:
    df = pd.DataFrame(columns=["candidate_id", "speed", "gap", "capacity"])
    aggregated = RecipeAggregation.aggregate_recipes(
        df,
        control_columns=["speed", "gap"],
        target_column="capacity",
    )
    assert aggregated.empty
    assert list(aggregated.columns) == ["candidate_id", "capacity", "speed", "gap"]
