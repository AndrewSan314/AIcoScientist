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
from src.process.benchmarks.rediscovery import RediscoveryReplay
from src.process.information_horizon import PreManufacturingRecipeSelectionHorizon
from src.process.stages import ProcessStage


@pytest.fixture(scope="module")
def nmc622_adapter() -> WarwickNMC622CalenderingAdapter:
    repo_root = Path(__file__).resolve().parent.parent.parent
    extracted_root = (
        repo_root
        / "data"
        / "external"
        / "warwick_nmc622_calendering"
        / "raw"
        / "Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale"
    )
    if (extracted_root / "1- Tables").exists():
        return WarwickNMC622CalenderingAdapter(extracted_root)
    raw_dir = repo_root / "data" / "external" / "warwick_nmc622_calendering" / "raw"
    if (raw_dir / "1- Tables").exists() or (raw_dir / "Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale" / "1- Tables").exists():
        return WarwickNMC622CalenderingAdapter(raw_dir)
    pytest.fail(f"Warwick NMC622 dataset tables not found at {extracted_root}")


def test_nmc622_target_units_and_hidden_features(nmc622_adapter: WarwickNMC622CalenderingAdapter) -> None:
    """Verify rate_performance_5c_over_0_2c units are dimensionless_ratio and not mAh/g."""
    runs = nmc622_adapter.load_runs()
    sample_run = runs[0]
    kpi = sample_run.final_kpis["rate_performance_5c_over_0_2c"]
    assert kpi.units == "dimensionless_ratio"
    assert kpi.units != "mAh/g"

    pool = nmc622_adapter.get_candidate_pool()
    runs_by_recipe = nmc622_adapter.get_runs_by_recipe()
    replay = RediscoveryReplay(
        candidate_pool=pool,
        candidate_id_column="recipe_id",
        target_column="rate_performance_5c_over_0_2c",
        control_columns=["roll_temperature_c", "target_density_g_cm3", "target_coating_weight_gsm"],
        dataset_id="warwick_nmc622_calendering",
        runs_by_recipe=runs_by_recipe,
    )
    assert replay._resolve_target_units() == "dimensionless_ratio"
    assert replay._resolve_target_units() != "mAh/g"


def test_nmc622_dynamic_hidden_observations_audit(nmc622_adapter: WarwickNMC622CalenderingAdapter) -> None:
    """Verify hidden observations are dynamically source-derived and contain execution quantities."""
    pool = nmc622_adapter.get_candidate_pool()
    runs_by_recipe = nmc622_adapter.get_runs_by_recipe()
    replay = RediscoveryReplay(
        candidate_pool=pool,
        candidate_id_column="recipe_id",
        target_column="rate_performance_5c_over_0_2c",
        control_columns=["roll_temperature_c", "target_density_g_cm3", "target_coating_weight_gsm"],
        dataset_id="warwick_nmc622_calendering",
        runs_by_recipe=runs_by_recipe,
    )
    hidden_obs: set[str] = set()
    for r_list in runs_by_recipe.values():
        for r in r_list:
            for st in r.stages:
                hidden_obs.update(st.intermediate_properties.keys())
            hidden_obs.update(r.final_kpis.keys())
    ctrls = {"roll_temperature_c", "target_density_g_cm3", "target_coating_weight_gsm"}
    hidden_names = sorted(hidden_obs - ctrls)

    assert "roll_gap_um" in hidden_names
    assert "number_of_passes" in hidden_names
    assert "target_porosity_pct" in hidden_names
    assert "rate_performance_5c_over_0_2c" in hidden_names
    # Verify Drakopoulos-only features are NOT present in NMC
    assert "active_mass_mg" not in hidden_names
    assert "discharge_specific_capacity_cycle30_mah_g" not in hidden_names


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
