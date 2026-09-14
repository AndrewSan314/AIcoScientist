import pytest
import pandas as pd

from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosGraphiteAdapter
from src.process.benchmarks.rediscovery import RediscoveryReplay


def _get_toy_pool():
    # 6 candidates, controls x1 and x2, target y
    # Candidate "best" has y=100.0, others have lower values
    return pd.DataFrame([
        {"recipe_id": "c1", "x1": 1.0, "x2": 10.0, "y": 10.0},
        {"recipe_id": "c2", "x1": 2.0, "x2": 20.0, "y": 20.0},
        {"recipe_id": "c3", "x1": 3.0, "x2": 30.0, "y": 30.0},
        {"recipe_id": "c4", "x1": 4.0, "x2": 40.0, "y": 40.0},
        {"recipe_id": "c5", "x1": 5.0, "x2": 50.0, "y": 50.0},
        {"recipe_id": "best", "x1": 6.0, "x2": 60.0, "y": 100.0},
    ])


def test_rank_semantics_random_policy():
    pool = _get_toy_pool()
    replay = RediscoveryReplay(
        candidate_pool=pool,
        candidate_id_column="recipe_id",
        target_column="y",
        control_columns=["x1", "x2"],
    )

    traj = replay.run(strategy="random", seed=42, initial_size=2, max_steps=4)
    found_step = traj.experiments_to_best

    for step in traj.steps:
        if found_step is not None and step.step > found_step:
            # All steps after reveal MUST have hidden_best_rank = None
            assert step.hidden_best_rank is None, f"Step {step.step} after reveal should have hidden_best_rank None"
        elif found_step is not None and step.step == found_step:
            # The step that selects it records pre-selection rank
            assert step.hidden_best_rank is not None
            assert 1 <= step.hidden_best_rank <= len(pool)


def test_rank_semantics_production_surrogate_policy():
    pool = _get_toy_pool()
    replay = RediscoveryReplay(
        candidate_pool=pool,
        candidate_id_column="recipe_id",
        target_column="y",
        control_columns=["x1", "x2"],
    )

    traj = replay.run(strategy="AICOSCIENTIST_PROCESS_SURROGATE", seed=42, initial_size=2, max_steps=4)
    found_step = traj.experiments_to_best

    for step in traj.steps:
        if found_step is not None and step.step > found_step:
            # All steps after reveal MUST have hidden_best_rank = None
            assert step.hidden_best_rank is None
        elif found_step is not None and step.step == found_step:
            assert step.hidden_best_rank is not None
            assert 1 <= step.hidden_best_rank <= len(pool)
