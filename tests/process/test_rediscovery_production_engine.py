"""Tests verifying that rediscovery routes through the production engine stack."""

from __future__ import annotations
import pytest
import pandas as pd
from src.process.benchmarks.rediscovery import RediscoveryReplay

def test_production_engine_invocation() -> None:
    pool = pd.DataFrame([
        {"recipe_id": "p1", "speed": 0.1, "gap": 100.0, "d30": 200.0},
        {"recipe_id": "p2", "speed": 0.2, "gap": 150.0, "d30": 300.0},
        {"recipe_id": "p3", "speed": 0.3, "gap": 200.0, "d30": 400.0},
        {"recipe_id": "p4", "speed": 0.4, "gap": 250.0, "d30": 250.0},
    ])
    replay = RediscoveryReplay(
        pool,
        candidate_id_column="recipe_id",
        target_column="d30",
        control_columns=["speed", "gap"],
        minimize=False,
    )
    traj = replay.run(
        strategy="AICOSCIENTIST_PROCESS_ENGINE",
        seed=11,
        initial_size=2,
        max_steps=2,
    )
    assert traj.policy == "AICOSCIENTIST_PROCESS_ENGINE"
    assert len(traj.steps) == 2
    for step in traj.steps:
        assert step.predicted_mean is not None
        assert step.predicted_std is not None
