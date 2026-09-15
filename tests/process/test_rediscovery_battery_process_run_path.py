"""Tests verifying BatteryProcessRun routing through PreManufacturingRecipeSelectionHorizon in rediscovery replay."""
from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosGraphiteAdapter
from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.information_horizon import DecisionHorizon, PreManufacturingRecipeSelectionHorizon
from src.process.benchmarks.rediscovery import (
    FrozenSurrogateOptimizerBackend,
    RediscoveryReplay,
)
from src.process.stages import ProcessStage


def _create_mock_run(run_id: str, recipe_id: str, gap: float, speed: float, d30: float, mass: float) -> BatteryProcessRun:
    prov = ProvenanceRecord("PHYSICAL_HISTORICAL", "doi:10.17632/4dh2h3tsf4.1")
    stages = [
        StageRecord(f"{run_id}:formulation", ProcessStage.FORMULATION, 0, {"active_material_fraction_pct": ParameterValue(96.0, "%")}, {}, [], provenance=prov),
        StageRecord(f"{run_id}:mixing", ProcessStage.MIXING, 1, {"mixing_solids_pct": ParameterValue(48.0, "%")}, {}, [], f"{run_id}:formulation", provenance=prov),
        StageRecord(f"{run_id}:coating", ProcessStage.COATING, 2, {"coating_gap_um": ParameterValue(gap, "um"), "coating_speed_m_per_min": ParameterValue(speed, "m/min")}, {}, [], f"{run_id}:mixing", provenance=prov),
        StageRecord(f"{run_id}:drying", ProcessStage.DRYING, 3, {"drying_temperature_c": ParameterValue(60.0, "C")}, {}, [], f"{run_id}:coating", provenance=prov),
        StageRecord(f"{run_id}:calendering", ProcessStage.CALENDERING, 4, {"calendering_applied": ParameterValue(0.0, "binary")}, {"active_mass_mg": MeasurementValue(mass, "mg")}, [], f"{run_id}:drying", provenance=prov),
    ]
    return BatteryProcessRun(
        run_id=run_id,
        cell_id=run_id,
        batch_id=recipe_id,
        chemistry_id="graphite Li-ion electrode",
        equipment_context={},
        environment_context={},
        stages=stages,
        final_kpis={"discharge_specific_capacity_cycle30_mah_g": MeasurementValue(d30, "mAh/g")},
        provenance=prov,
    )


def test_battery_process_run_horizon_projection() -> None:
    """Verify run projection under PreManufacturingRecipeSelectionHorizon exposes controls and hides outcomes."""
    run = _create_mock_run("cell-1", "rec-1", gap=100.0, speed=0.2, d30=390.0, mass=12.5)
    horizon = PreManufacturingRecipeSelectionHorizon()

    view = horizon.project_for_recipe_selection(run)
    # All planned controls visible
    assert "coating_gap_um" in view.controls
    assert "coating_speed_m_per_min" in view.controls
    assert "drying_temperature_c" in view.controls
    assert "active_material_fraction_pct" in view.controls
    assert "mixing_solids_pct" in view.controls

    # Intermediate properties (mass) and final KPIs (D30) strictly hidden
    assert len(view.intermediate_properties) == 0
    assert "active_mass_mg" not in view.controls
    assert "discharge_specific_capacity_cycle30_mah_g" not in view.controls


def test_rediscovery_replay_with_explicit_runs_by_recipe() -> None:
    """Verify RediscoveryReplay consumes runs_by_recipe and executes full engine pipeline."""
    runs_by_recipe = {
        "rec-1": [
            _create_mock_run("c1-a", "rec-1", gap=100.0, speed=0.1, d30=220.0, mass=11.0),
            _create_mock_run("c1-b", "rec-1", gap=100.0, speed=0.1, d30=230.0, mass=11.2),
        ],
        "rec-2": [
            _create_mock_run("c2-a", "rec-2", gap=150.0, speed=0.2, d30=340.0, mass=14.0),
            _create_mock_run("c2-b", "rec-2", gap=150.0, speed=0.2, d30=350.0, mass=14.1),
        ],
        "rec-3": [
            _create_mock_run("c3-a", "rec-3", gap=200.0, speed=0.3, d30=400.0, mass=18.0),
            _create_mock_run("c3-b", "rec-3", gap=200.0, speed=0.3, d30=410.0, mass=18.2),
        ],
        "rec-4": [
            _create_mock_run("c4-a", "rec-4", gap=120.0, speed=0.15, d30=280.0, mass=12.0),
        ],
        "rec-5": [
            _create_mock_run("c5-a", "rec-5", gap=180.0, speed=0.25, d30=370.0, mass=16.0),
        ],
    }

    pool = pd.DataFrame([
        {"recipe_id": "rec-1", "coating_gap_um": 100.0, "coating_speed_m_per_min": 0.1, "discharge_specific_capacity_cycle30_mah_g": 225.0},
        {"recipe_id": "rec-2", "coating_gap_um": 150.0, "coating_speed_m_per_min": 0.2, "discharge_specific_capacity_cycle30_mah_g": 345.0},
        {"recipe_id": "rec-3", "coating_gap_um": 200.0, "coating_speed_m_per_min": 0.3, "discharge_specific_capacity_cycle30_mah_g": 405.0},
        {"recipe_id": "rec-4", "coating_gap_um": 120.0, "coating_speed_m_per_min": 0.15, "discharge_specific_capacity_cycle30_mah_g": 280.0},
        {"recipe_id": "rec-5", "coating_gap_um": 180.0, "coating_speed_m_per_min": 0.25, "discharge_specific_capacity_cycle30_mah_g": 370.0},
    ])

    replay = RediscoveryReplay(
        candidate_pool=pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_gap_um", "coating_speed_m_per_min"],
        runs_by_recipe=runs_by_recipe,
        decision_stage=DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION,
    )

    traj = replay.run(
        strategy="AICOSCIENTIST_PROCESS_SURROGATE",
        seed=101,
        initial_size=2,
        max_steps=2,
    )

    assert len(traj.steps) == 2
    assert traj.engine_path == "AICOSCIENTIST_PROCESS_SURROGATE"

    trace = replay.execution_trace
    assert trace.runs_loaded >= 4  # At least 2 initial recipes * 2 runs
    assert trace.horizon_projections >= 4
    assert trace.samples_created >= 4
    assert trace.surrogates_fitted == 2
    assert trace.artifacts_created == 2
    assert trace.coordinator_calls == 2
    assert trace.proposals_generated >= 2
    assert trace.oracle_reveals == 4  # 2 initial + 2 sequential steps


def test_missing_battery_process_run_fails_closed_in_full_engine() -> None:
    """Verify that physical historical full engine strictly fails closed if source runs are missing."""
    pool = pd.DataFrame([
        {"recipe_id": "r1", "coating_gap_um": 100.0, "coating_speed_m_per_min": 0.1, "discharge_specific_capacity_cycle30_mah_g": 225.0},
        {"recipe_id": "r2", "coating_gap_um": 150.0, "coating_speed_m_per_min": 0.2, "discharge_specific_capacity_cycle30_mah_g": 345.0},
        {"recipe_id": "r3", "coating_gap_um": 200.0, "coating_speed_m_per_min": 0.3, "discharge_specific_capacity_cycle30_mah_g": 405.0},
    ])
    # Empty runs mapping with allow_flat_fallback=False
    replay = RediscoveryReplay(
        candidate_pool=pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_gap_um", "coating_speed_m_per_min"],
        runs_by_recipe={},  # Missing runs for all recipes
        allow_flat_fallback=False,
    )
    with pytest.raises(RuntimeError, match="SOURCE_BATTERY_PROCESS_RUN_NOT_FOUND"):
        replay.run(
            strategy="AICOSCIENTIST_PROCESS_SURROGATE",
            seed=42,
            initial_size=2,
            max_steps=1,
        )


def test_process_surrogate_nei_alias_fails_closed() -> None:
    """Verify that requesting NEI on FrozenSurrogate raises explicit unsupported error."""
    pool = pd.DataFrame([
        {"recipe_id": "r1", "coating_gap_um": 100.0, "discharge_specific_capacity_cycle30_mah_g": 225.0},
        {"recipe_id": "r2", "coating_gap_um": 150.0, "discharge_specific_capacity_cycle30_mah_g": 345.0},
        {"recipe_id": "r3", "coating_gap_um": 200.0, "discharge_specific_capacity_cycle30_mah_g": 405.0},
    ])
    replay = RediscoveryReplay(
        candidate_pool=pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_gap_um"],
        allow_flat_fallback=True,
    )
    with pytest.raises(ValueError, match="NEI_NOT_IMPLEMENTED_FOR_FROZEN_SURROGATE"):
        replay.run(strategy="AICOSCIENTIST_PROCESS_SURROGATE_NEI", seed=42, initial_size=2, max_steps=1)

    with pytest.raises(ValueError, match="NEI_NOT_IMPLEMENTED_FOR_FROZEN_SURROGATE"):
        replay.run(strategy="aicointel_nei", seed=42, initial_size=2, max_steps=1)


def test_process_surrogate_records_expected_improvement() -> None:
    """Verify that AICOSCIENTIST_PROCESS_SURROGATE records EXPECTED_IMPROVEMENT acquisition."""
    pool = pd.DataFrame([
        {"recipe_id": "r1", "coating_gap_um": 100.0, "discharge_specific_capacity_cycle30_mah_g": 225.0},
        {"recipe_id": "r2", "coating_gap_um": 150.0, "discharge_specific_capacity_cycle30_mah_g": 345.0},
        {"recipe_id": "r3", "coating_gap_um": 200.0, "discharge_specific_capacity_cycle30_mah_g": 405.0},
    ])
    replay = RediscoveryReplay(
        candidate_pool=pool,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=["coating_gap_um"],
        allow_flat_fallback=True,
    )
    traj = replay.run(strategy="AICOSCIENTIST_PROCESS_SURROGATE", seed=42, initial_size=2, max_steps=1)
    step1 = traj.steps[0]
    assert step1.training_view_summary is not None
    assert step1.training_view_summary["acquisition_strategy"] == "EXPECTED_IMPROVEMENT"

