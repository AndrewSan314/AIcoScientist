"""Tests verifying distinct engine_path tracking and artifact identity across benchmark policies."""
from __future__ import annotations

import pandas as pd
import pytest

from src.process.benchmarks.rediscovery import (
    RediscoveryReplay,
    summarize_trajectories,
)


@pytest.fixture
def mock_pool() -> pd.DataFrame:
    return pd.DataFrame([
        {"recipe_id": "r1", "coating_speed_m_per_min": 0.1, "coating_gap_um": 100.0, "discharge_specific_capacity_cycle30_mah_g": 200.0},
        {"recipe_id": "r2", "coating_speed_m_per_min": 0.2, "coating_gap_um": 150.0, "discharge_specific_capacity_cycle30_mah_g": 350.0},
        {"recipe_id": "r3", "coating_speed_m_per_min": 0.3, "coating_gap_um": 200.0, "discharge_specific_capacity_cycle30_mah_g": 400.0},
        {"recipe_id": "r4", "coating_speed_m_per_min": 0.15, "coating_gap_um": 120.0, "discharge_specific_capacity_cycle30_mah_g": 280.0},
        {"recipe_id": "r5", "coating_speed_m_per_min": 0.25, "coating_gap_um": 180.0, "discharge_specific_capacity_cycle30_mah_g": 380.0},
    ])


def test_engine_path_distinction(mock_pool: pd.DataFrame) -> None:
    replay = RediscoveryReplay(
        mock_pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
        minimize=False,
    )

    # 1. AICOSCIENTIST_PROCESS_SURROGATE
    traj_surrogate = replay.run(strategy="AICOSCIENTIST_PROCESS_SURROGATE", seed=11, initial_size=2, max_steps=2)
    assert traj_surrogate.engine_path == "AICOSCIENTIST_PROCESS_SURROGATE"
    assert traj_surrogate.steps[0].engine_path == "AICOSCIENTIST_PROCESS_SURROGATE"
    assert traj_surrogate.steps[0].surrogate_artifact_fingerprint is not None

    # 2. DIRECT_BOTORCH_BASELINE
    traj_botorch = replay.run(strategy="DIRECT_BOTORCH_BASELINE", seed=11, initial_size=2, max_steps=2)
    assert traj_botorch.engine_path == "DIRECT_BOTORCH_BASELINE"
    assert traj_botorch.steps[0].engine_path == "DIRECT_BOTORCH_BASELINE"
    assert traj_botorch.steps[0].surrogate_artifact_fingerprint is None

    # 3. Random
    traj_random = replay.run(strategy="random", seed=11, initial_size=2, max_steps=2)
    assert traj_random.engine_path == "RANDOM_BASELINE"
    assert traj_random.steps[0].engine_path == "RANDOM_BASELINE"
    assert traj_random.steps[0].surrogate_artifact_fingerprint is None


def test_engine_path_serialization_and_summary(mock_pool: pd.DataFrame) -> None:
    replay = RediscoveryReplay(
        mock_pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
    )
    traj = replay.run(strategy="AICOSCIENTIST_PROCESS_SURROGATE", seed=42, initial_size=2, max_steps=2)
    t_dict = traj.to_dict()
    assert t_dict["engine_path"] == "AICOSCIENTIST_PROCESS_SURROGATE"
    assert t_dict["steps"][0]["engine_path"] == "AICOSCIENTIST_PROCESS_SURROGATE"

    summary = summarize_trajectories([traj])
    assert summary.engine_path == "AICOSCIENTIST_PROCESS_SURROGATE"
    s_dict = summary.to_dict()
    assert s_dict["engine_path"] == "AICOSCIENTIST_PROCESS_SURROGATE"
