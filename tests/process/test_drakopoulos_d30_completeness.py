"""Tests verifying D30 replicate completeness tracking and survivorship-bias fix (v3)."""

from __future__ import annotations
import pytest
from unittest.mock import MagicMock

from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosRecipeGroup
from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.stages import ProcessStage


def _make_run(run_id: str, d30_value: float | None, mass: float = 7.0) -> BatteryProcessRun:
    """Helper to create a minimal BatteryProcessRun with controllable D30 value."""
    prov = ProvenanceRecord("PHYSICAL_HISTORICAL", "", "", "1", {}, "2")
    form_ctrl = {
        "active_material_fraction_pct": ParameterValue(95.0, "%"),
        "conductive_additive_fraction_pct": ParameterValue(1.0, "%"),
        "binder_cmc_fraction_pct": ParameterValue(2.0, "%"),
        "binder_sbr_fraction_pct": ParameterValue(2.0, "%"),
    }
    mixing_ctrl = {"mixing_solids_pct": ParameterValue(49.5, "%")}
    coating_ctrl = {"coating_speed_m_per_min": ParameterValue(0.2, "m/min"), "coating_gap_um": ParameterValue(150.0, "um")}
    drying_ctrl = {"drying_temperature_c": ParameterValue(80.0, "C")}
    cal_ctrl = {"calendering_applied": ParameterValue(0.0, "binary")}
    metrology = {"active_mass_mg": MeasurementValue(mass, "mg")} if mass else {}
    stages = [
        StageRecord(f"{run_id}:formulation", ProcessStage.FORMULATION, 0, form_ctrl, {}, [], provenance=prov),
        StageRecord(f"{run_id}:mixing", ProcessStage.MIXING, 1, mixing_ctrl, {}, [], f"{run_id}:formulation", provenance=prov),
        StageRecord(f"{run_id}:coating", ProcessStage.COATING, 2, coating_ctrl, {}, [], f"{run_id}:mixing", provenance=prov),
        StageRecord(f"{run_id}:drying", ProcessStage.DRYING, 3, drying_ctrl, {}, [], f"{run_id}:coating", provenance=prov),
        StageRecord(f"{run_id}:calendering", ProcessStage.CALENDERING, 4, cal_ctrl, metrology, [], f"{run_id}:drying", provenance=prov),
    ]
    final_kpis = {}
    if d30_value is not None:
        final_kpis["discharge_specific_capacity_cycle30_mah_g"] = MeasurementValue(d30_value, "mAh/g")
    return BatteryProcessRun(
        run_id=run_id, cell_id=run_id, batch_id="recipe-A",
        chemistry_id="graphite", equipment_context={}, environment_context={},
        stages=stages, final_kpis=final_kpis, provenance=prov,
    )


class TestDrakopoulosRecipeGroupCompleteness:
    """Verify that DrakopoulosRecipeGroup correctly tracks D30 replicate completeness."""

    def test_completeness_fields_present(self) -> None:
        """DrakopoulosRecipeGroup must expose all completeness tracking fields."""
        g = DrakopoulosRecipeGroup(
            recipe_id="test", controls={}, calendered=False,
            replicate_count=3, mean_d30_specific_capacity=350.0,
            std_d30_specific_capacity=5.0, mean_active_mass_mg=7.0,
            cell_ids=("A", "B", "C"),
            total_replicates=3, valid_d30_replicates=3,
            zero_d30_replicates=0, missing_d30_replicates=0,
            recipe_eligibility_status="STRICT_COMPLETE_RECIPE",
        )
        assert g.total_replicates == 3
        assert g.valid_d30_replicates == 3
        assert g.zero_d30_replicates == 0
        assert g.missing_d30_replicates == 0
        assert g.recipe_eligibility_status == "STRICT_COMPLETE_RECIPE"

    def test_strict_complete_recipe_all_measured(self) -> None:
        """All cells have D30 measurements → STRICT_COMPLETE_RECIPE."""
        g = DrakopoulosRecipeGroup(
            recipe_id="r1", controls={}, calendered=False,
            replicate_count=3, mean_d30_specific_capacity=350.0,
            std_d30_specific_capacity=5.0, mean_active_mass_mg=7.0,
            cell_ids=("A", "B", "C"),
            total_replicates=3, valid_d30_replicates=3,
            zero_d30_replicates=0, missing_d30_replicates=0,
            recipe_eligibility_status="STRICT_COMPLETE_RECIPE",
        )
        assert g.recipe_eligibility_status == "STRICT_COMPLETE_RECIPE"

    def test_partial_d30_when_some_missing(self) -> None:
        """Some cells lack D30 → PARTIAL_D30."""
        g = DrakopoulosRecipeGroup(
            recipe_id="r2", controls={}, calendered=False,
            replicate_count=3, mean_d30_specific_capacity=340.0,
            std_d30_specific_capacity=10.0, mean_active_mass_mg=7.0,
            cell_ids=("A", "B", "C"),
            total_replicates=3, valid_d30_replicates=2,
            zero_d30_replicates=0, missing_d30_replicates=1,
            recipe_eligibility_status="PARTIAL_D30",
        )
        assert g.recipe_eligibility_status == "PARTIAL_D30"
        assert g.missing_d30_replicates == 1

    def test_no_d30_when_all_missing(self) -> None:
        """No cells have D30 → NO_D30."""
        g = DrakopoulosRecipeGroup(
            recipe_id="r3", controls={}, calendered=False,
            replicate_count=3, mean_d30_specific_capacity=0.0,
            std_d30_specific_capacity=0.0, mean_active_mass_mg=7.0,
            cell_ids=("A", "B", "C"),
            total_replicates=3, valid_d30_replicates=0,
            zero_d30_replicates=0, missing_d30_replicates=3,
            recipe_eligibility_status="NO_D30",
        )
        assert g.recipe_eligibility_status == "NO_D30"
        assert g.valid_d30_replicates == 0


class TestSurvivorshipBiasFix:
    """Verify that zero-capacity cells are included in D30 mean (not filtered out)."""

    def test_zero_d30_included_in_mean(self) -> None:
        """A zero D30 cell must be included in mean, not silently dropped."""
        # 2 cells: one with D30=400, one with D30=0
        # Correct mean = 200, biased mean (old code) = 400
        from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosGraphiteAdapter
        adapter = DrakopoulosGraphiteAdapter.__new__(DrakopoulosGraphiteAdapter)
        
        runs = [
            _make_run("cell-1", d30_value=400.0),
            _make_run("cell-2", d30_value=0.0),
        ]
        # Mock load_partition to return our test runs
        adapter.load_partition = lambda partition="PROSPECTIVE_MODEL_VALIDATION": runs
        groups = adapter.load_recipe_groups()
        
        assert len(groups) == 1
        g = groups[0]
        # V3: zero D30 included → mean = 200, not 400
        assert pytest.approx(g.mean_d30_specific_capacity, abs=0.01) == 200.0
        assert g.zero_d30_replicates == 1
        assert g.valid_d30_replicates == 2
        assert g.missing_d30_replicates == 0
        assert g.recipe_eligibility_status == "STRICT_COMPLETE_RECIPE"

    def test_missing_d30_tracked_correctly(self) -> None:
        """Cells without D30 KPI are counted as missing, not zero."""
        runs = [
            _make_run("cell-1", d30_value=380.0),
            _make_run("cell-2", d30_value=None),  # Missing D30
            _make_run("cell-3", d30_value=360.0),
        ]
        from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosGraphiteAdapter
        adapter = DrakopoulosGraphiteAdapter.__new__(DrakopoulosGraphiteAdapter)
        adapter.load_partition = lambda partition="PROSPECTIVE_MODEL_VALIDATION": runs
        groups = adapter.load_recipe_groups()
        
        assert len(groups) == 1
        g = groups[0]
        assert g.valid_d30_replicates == 2
        assert g.missing_d30_replicates == 1
        assert g.total_replicates == 3
        assert g.recipe_eligibility_status == "PARTIAL_D30"
        # Mean computed from valid only: (380 + 360) / 2 = 370
        assert pytest.approx(g.mean_d30_specific_capacity, abs=0.01) == 370.0

    def test_strict_complete_recipe_filter(self) -> None:
        """STRICT_COMPLETE_RECIPE filtering should only keep fully-measured recipes."""
        runs_a = [_make_run("a-1", 350.0), _make_run("a-2", 360.0)]  # Complete
        runs_b = [_make_run("b-1", 380.0), _make_run("b-2", None)]   # Partial
        
        # Give different batch_ids
        for r in runs_a:
            object.__setattr__(r, 'batch_id', 'recipe-complete')
        for r in runs_b:
            object.__setattr__(r, 'batch_id', 'recipe-partial')
        
        from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosGraphiteAdapter
        adapter = DrakopoulosGraphiteAdapter.__new__(DrakopoulosGraphiteAdapter)
        adapter.load_partition = lambda partition="PROSPECTIVE_MODEL_VALIDATION": runs_a + runs_b
        groups = adapter.load_recipe_groups()
        
        strict = [g for g in groups if g.recipe_eligibility_status == "STRICT_COMPLETE_RECIPE"]
        partial = [g for g in groups if g.recipe_eligibility_status == "PARTIAL_D30"]
        
        assert len(strict) == 1
        assert strict[0].recipe_id == "recipe-complete"
        assert len(partial) == 1
        assert partial[0].recipe_id == "recipe-partial"
