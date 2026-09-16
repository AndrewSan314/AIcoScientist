"""Tests validating Warwick NMC622 Calendering Dataset Adapter and Benchmark Guarantees."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.datasets.battery_process.warwick_nmc622_calendering import WarwickNMC622CalenderingAdapter
from src.process.contracts import BatteryProcessRun
from src.process.information_horizon import PreManufacturingRecipeSelectionHorizon
from src.process.stages import ProcessStage


@pytest.fixture(scope="module")
def nmc622_adapter() -> WarwickNMC622CalenderingAdapter:
    repo_root = Path(__file__).resolve().parent.parent.parent
    raw_archive = (
        repo_root
        / "data"
        / "external"
        / "warwick_nmc622_calendering"
        / "raw"
        / "Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale.zip"
    )
    if not raw_archive.exists():
        pytest.skip(f"Raw archive not found at {raw_archive}")
    return WarwickNMC622CalenderingAdapter(raw_archive)


def test_nmc622_adapter_load_runs(nmc622_adapter: WarwickNMC622CalenderingAdapter) -> None:
    """Verify adapter extracts exactly 54 runs spanning 18 conditions (3 replicates each)."""
    runs = nmc622_adapter.load_runs()
    assert len(runs) == 54

    # Group by batch_id (condition ID)
    runs_by_condition: dict[str, list[BatteryProcessRun]] = {}
    for run in runs:
        runs_by_condition.setdefault(run.batch_id, []).append(run)

    assert len(runs_by_condition) == 18
    for cond_id, cond_runs in runs_by_condition.items():
        assert len(cond_runs) == 3, f"Condition {cond_id} has {len(cond_runs)} replicates, expected 3"


def test_nmc622_provenance_and_stages(nmc622_adapter: WarwickNMC622CalenderingAdapter) -> None:
    """Verify physical pilot line provenance and required process stages."""
    runs = nmc622_adapter.load_runs()
    sample_run = runs[0]

    assert sample_run.provenance.evidence_kind == "PILOT_LINE_HISTORICAL"
    assert "10.17632/wwhm2frfmy.1" in sample_run.provenance.source_doi
    assert "NMC622" in sample_run.chemistry_id

    stage_types = [s.stage_type for s in sample_run.stages]
    assert ProcessStage.COATING in stage_types
    assert ProcessStage.CALENDERING in stage_types
    assert ProcessStage.FINAL_CHARACTERIZATION in stage_types


def test_nmc622_information_horizon_firewall(nmc622_adapter: WarwickNMC622CalenderingAdapter) -> None:
    """Verify PreManufacturingRecipeSelectionHorizon strictly masks measured intermediate and final KPIs."""
    runs = nmc622_adapter.load_runs()
    horizon = PreManufacturingRecipeSelectionHorizon()

    for run in runs[:5]:
        view = horizon.project_for_recipe_selection(run)
        # Pre-manufacturing planned recipe controls must be exposed
        assert "roll_temperature_c" in view.controls
        assert "target_density_g_cm3" in view.controls
        assert "target_coating_weight_gsm" in view.controls

        # Physical execution parameters, measured intermediate cell properties, and final KPIs must be strictly hidden
        assert "roll_gap_um" not in view.controls
        assert "number_of_passes" not in view.controls
        assert "target_porosity_pct" not in view.controls
        assert "rate_performance_5c_over_0_2c" not in view.controls
        assert "rate_performance_3c_over_0_2c" not in view.controls
        assert "calendered_density_g_cm3" not in view.controls
        assert "calendered_thickness_um" not in view.controls
        assert len(view.intermediate_properties) == 0

        # Verify hidden_observations contains execution and intermediate properties
        hidden = horizon.hidden_observations(run)
        assert "roll_gap_um" in hidden
        assert "number_of_passes" in hidden
        assert "calendered_density_g_cm3" in hidden
        assert "rate_performance_5c_over_0_2c" in hidden


def test_nmc622_source_observed_best_consistency(nmc622_adapter: WarwickNMC622CalenderingAdapter) -> None:
    """Verify EXP_03 is the unique source-observed maximum for rate performance 5C:0.2C."""
    runs = nmc622_adapter.load_runs()
    cond_means: dict[str, list[float]] = {}

    for run in runs:
        cond_id = run.batch_id
        val = run.final_kpis["rate_performance_5c_over_0_2c"].value
        cond_means.setdefault(cond_id, []).append(val)

    mean_per_cond = {k: float(np.mean(v)) for k, v in cond_means.items()}
    best_cond = max(mean_per_cond.items(), key=lambda x: x[1])

    assert best_cond[0] == "EXP_03"
    assert pytest.approx(best_cond[1], rel=1e-3) == 0.7947
