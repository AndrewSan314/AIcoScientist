"""Tests verifying genuine ProcessOptimizationCoordinator execution during rediscovery replay."""
from __future__ import annotations

from unittest.mock import patch
import pandas as pd
import pytest

from src.process.benchmarks.rediscovery import (
    FrozenSurrogateOptimizerBackend,
    RediscoveryReplay,
)
from src.process.coordinator import ProcessOptimizationCoordinator


@pytest.fixture
def mock_pool() -> pd.DataFrame:
    return pd.DataFrame([
        {"recipe_id": "r1", "coating_speed_m_per_min": 0.1, "coating_gap_um": 100.0, "discharge_specific_capacity_cycle30_mah_g": 200.0},
        {"recipe_id": "r2", "coating_speed_m_per_min": 0.2, "coating_gap_um": 150.0, "discharge_specific_capacity_cycle30_mah_g": 350.0},
        {"recipe_id": "r3", "coating_speed_m_per_min": 0.3, "coating_gap_um": 200.0, "discharge_specific_capacity_cycle30_mah_g": 400.0},
        {"recipe_id": "r4", "coating_speed_m_per_min": 0.15, "coating_gap_um": 120.0, "discharge_specific_capacity_cycle30_mah_g": 280.0},
        {"recipe_id": "r5", "coating_speed_m_per_min": 0.25, "coating_gap_um": 180.0, "discharge_specific_capacity_cycle30_mah_g": 380.0},
    ])


def test_coordinator_propose_recipes_actually_called(mock_pool: pd.DataFrame) -> None:
    """Verify coordinator.propose_recipes is actually invoked at every sequential BO step."""
    replay = RediscoveryReplay(
        mock_pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
        minimize=False,
    )

    original_propose = ProcessOptimizationCoordinator.propose_recipes
    call_records: list[dict] = []

    def spy_propose(self, observations, space, objective, **kwargs):
        call_records.append({
            "observations_count": len(observations),
            "space_candidates_count": len(space.candidates),
            "backend": type(self.scalar_backend).__name__,
            "strategy": kwargs.get("strategy"),
        })
        return original_propose(self, observations, space, objective, **kwargs)

    with patch.object(ProcessOptimizationCoordinator, "propose_recipes", side_effect=spy_propose, autospec=True):
        traj = replay.run(
            strategy="AICOSCIENTIST_PROCESS_SURROGATE",
            seed=42,
            initial_size=2,
            max_steps=2,
        )

    # Exactly 2 steps executed -> propose_recipes called exactly 2 times
    assert len(call_records) == 2
    assert len(traj.steps) == 2

    # Step 1: 2 initial observations revealed -> 3 visible candidates remain
    assert call_records[0]["observations_count"] == 2
    assert call_records[0]["space_candidates_count"] == 3
    assert call_records[0]["backend"] == "FrozenSurrogateOptimizerBackend"
    assert call_records[0]["strategy"] == "expected_improvement"

    # Step 2: 3 observations revealed -> 2 visible candidates remain
    assert call_records[1]["observations_count"] == 3
    assert call_records[1]["space_candidates_count"] == 2
    assert call_records[1]["backend"] == "FrozenSurrogateOptimizerBackend"
    assert call_records[1]["strategy"] == "expected_improvement"


def test_frozen_surrogate_optimizer_backend_contract(mock_pool: pd.DataFrame) -> None:
    """Verify FrozenSurrogateOptimizerBackend adheres to the scalar backend contract."""
    replay = RediscoveryReplay(
        mock_pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_speed_m_per_min", "coating_gap_um"],
    )
    traj = replay.run(
        strategy="AICOSCIENTIST_PROCESS_SURROGATE",
        seed=11,
        initial_size=2,
        max_steps=1,
    )
    step1 = traj.steps[0]
    assert step1.predicted_mean is not None
    assert step1.predicted_std is not None
    assert step1.acquisition_value is not None
    assert step1.engine_path == "AICOSCIENTIST_PROCESS_SURROGATE"
    assert step1.surrogate_artifact_fingerprint is not None
    assert replay.execution_trace.coordinator_calls >= 1
    assert replay.execution_trace.surrogates_fitted >= 1
