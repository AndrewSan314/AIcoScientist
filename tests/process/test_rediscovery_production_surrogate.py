"""Tests verifying genuine production ProcessSurrogate execution in rediscovery benchmark."""
from __future__ import annotations

import pandas as pd
import pytest

from src.process.benchmarks.rediscovery import RediscoveryReplay
from src.process.stages import ProcessStage


@pytest.fixture
def mock_pool() -> pd.DataFrame:
    return pd.DataFrame([
        {"recipe_id": "r1", "coating_speed_m_per_min": 0.1, "coating_gap_um": 100.0, "calendering_applied": 0.0, "discharge_specific_capacity_cycle30_mah_g": 210.0},
        {"recipe_id": "r2", "coating_speed_m_per_min": 0.2, "coating_gap_um": 150.0, "calendering_applied": 1.0, "discharge_specific_capacity_cycle30_mah_g": 340.0},
        {"recipe_id": "r3", "coating_speed_m_per_min": 0.3, "coating_gap_um": 200.0, "calendering_applied": 1.0, "discharge_specific_capacity_cycle30_mah_g": 405.0},  # hidden best
        {"recipe_id": "r4", "coating_speed_m_per_min": 0.15, "coating_gap_um": 120.0, "calendering_applied": 0.0, "discharge_specific_capacity_cycle30_mah_g": 290.0},
        {"recipe_id": "r5", "coating_speed_m_per_min": 0.25, "coating_gap_um": 180.0, "calendering_applied": 1.0, "discharge_specific_capacity_cycle30_mah_g": 375.0},
    ])


def test_production_surrogate_pipeline_provenance(mock_pool: pd.DataFrame) -> None:
    """Verify production ProcessSurrogate fits at each step and records artifact fingerprint."""
    replay = RediscoveryReplay(
        mock_pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um", "calendering_applied"],
        decision_stage=ProcessStage.COATING,
    )
    traj = replay.run(
        strategy="AICOSCIENTIST_PROCESS_SURROGATE",
        seed=101,
        initial_size=2,
        max_steps=2,
    )
    assert traj.engine_path == "AICOSCIENTIST_PROCESS_SURROGATE"
    assert len(traj.steps) == 2

    for step in traj.steps:
        assert step.engine_path == "AICOSCIENTIST_PROCESS_SURROGATE"
        assert step.surrogate_artifact_fingerprint is not None
        assert len(step.surrogate_artifact_fingerprint) == 64  # SHA-256
        assert step.information_horizon == "InformationHorizon(COATING)"
        assert step.predicted_mean is not None
        assert step.predicted_std is not None
        assert step.training_view_summary is not None
        assert step.training_view_summary["decision_stage"] == "COATING"
        assert step.dataset_fingerprint is not None


def test_process_surrogate_excludes_downstream_controls_under_coating_horizon(mock_pool: pd.DataFrame) -> None:
    """Verify InformationHorizon(COATING) excludes downstream CALENDERING controls from surrogate input."""
    replay = RediscoveryReplay(
        mock_pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um", "calendering_applied"],
        decision_stage=ProcessStage.COATING,
    )
    traj = replay.run(
        strategy="AICOSCIENTIST_PROCESS_SURROGATE",
        seed=42,
        initial_size=2,
        max_steps=1,
    )
    step1 = traj.steps[0]
    # Under COATING horizon, calendering_applied is downstream and should be excluded from observable controls
    assert "calendering_applied" not in step1.training_view_summary["observable_controls"]
    assert "coating_speed_m_per_min" in step1.training_view_summary["observable_controls"]
    assert "coating_gap_um" in step1.training_view_summary["observable_controls"]
