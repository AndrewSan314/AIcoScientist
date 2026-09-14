"""Tests verifying ProcessOptimizationCoordinator routing and hypergeometric analytics in rediscovery benchmark."""

from __future__ import annotations

import pandas as pd
import pytest

from src.process.benchmarks.rediscovery import (
    ProductionProcessRediscoveryRunner,
    RediscoveryReplay,
    calculate_hypergeometric_baseline,
)
from src.process.coordinator import ProcessOptimizationCoordinator


@pytest.fixture
def mock_recipe_pool() -> pd.DataFrame:
    return pd.DataFrame([
        {"recipe_id": "r1", "coating_speed_m_per_min": 0.1, "coating_gap_um": 100.0, "discharge_specific_capacity_cycle30_mah_g": 200.0},
        {"recipe_id": "r2", "coating_speed_m_per_min": 0.2, "coating_gap_um": 150.0, "discharge_specific_capacity_cycle30_mah_g": 350.0},
        {"recipe_id": "r3", "coating_speed_m_per_min": 0.3, "coating_gap_um": 200.0, "discharge_specific_capacity_cycle30_mah_g": 400.0},  # hidden best
        {"recipe_id": "r4", "coating_speed_m_per_min": 0.15, "coating_gap_um": 120.0, "discharge_specific_capacity_cycle30_mah_g": 280.0},
        {"recipe_id": "r5", "coating_speed_m_per_min": 0.25, "coating_gap_um": 180.0, "discharge_specific_capacity_cycle30_mah_g": 380.0},
    ])


def test_calculate_hypergeometric_baseline_exact() -> None:
    curve = calculate_hypergeometric_baseline(total_candidates=10, initial_size=2, budget=4, top_k=1)
    assert pytest.approx(curve[1], abs=1e-5) == 0.125
    assert pytest.approx(curve[2], abs=1e-5) == 0.25
    assert pytest.approx(curve[4], abs=1e-5) == 0.5


def test_calculate_hypergeometric_baseline_top3() -> None:
    curve = calculate_hypergeometric_baseline(total_candidates=10, initial_size=2, budget=2, top_k=3)
    assert pytest.approx(curve[1], abs=1e-5) == 0.375
    assert pytest.approx(curve[2], abs=1e-5) == 18.0 / 28.0


def test_rediscovery_coordinator_policy(mock_recipe_pool: pd.DataFrame) -> None:
    replay = RediscoveryReplay(
        mock_recipe_pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
        minimize=False,
    )
    trajectory = replay.run(
        strategy="AICOSCIENTIST_PROCESS_ENGINE",
        seed=42,
        initial_size=2,
        max_steps=2,
    )
    assert len(trajectory.steps) == 2
    assert trajectory.policy == "AICOSCIENTIST_PROCESS_ENGINE"
    assert trajectory.hidden_best_id == "r3"
    assert trajectory.hidden_best_value == 400.0
    for step in trajectory.steps:
        assert step.predicted_mean is not None
        assert step.predicted_std is not None
        assert 1 <= step.hidden_best_rank <= 3


def test_production_runner_smoke(mock_recipe_pool: pd.DataFrame) -> None:
    runner = ProductionProcessRediscoveryRunner(
        mock_recipe_pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
    )
    res = runner.run(
        policies=["AICOSCIENTIST_PROCESS_ENGINE", "random"],
        seeds=[42],
        initial_size=2,
        budget=2,
    )
    assert "AICOSCIENTIST_PROCESS_ENGINE" in res["trajectories"]
    assert "random" in res["trajectories"]
    assert "analytic_hypergeometric" in res
    assert 1 in res["analytic_hypergeometric"]["top1_hit_rate_by_step"]


def test_direct_botorch_baseline_policy(mock_recipe_pool: pd.DataFrame) -> None:
    replay = RediscoveryReplay(
        mock_recipe_pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
        minimize=False,
    )
    trajectory = replay.run(
        strategy="DIRECT_BOTORCH_BASELINE",
        seed=42,
        initial_size=2,
        max_steps=2,
    )
    assert len(trajectory.steps) == 2
    assert trajectory.policy == "DIRECT_BOTORCH_BASELINE"
    assert trajectory.hidden_best_id == "r3"
    assert trajectory.hidden_best_value == 400.0
    for step in trajectory.steps:
        assert step.predicted_mean is not None
        assert step.predicted_std is not None
        assert 1 <= step.hidden_best_rank <= 3

