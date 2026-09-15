"""Tests verifying zero-D30 semantics and replicate completeness."""

from __future__ import annotations
import pytest

from src.datasets.battery_process.drakopoulos_graphite import (
    DrakopoulosGraphiteAdapter,
    DrakopoulosRecipeGroup,
)
from src.process.contracts import (
    BatteryProcessRun,
    MeasurementValue,
    ParameterValue,
    ProvenanceRecord,
    StageRecord,
)
from src.process.stages import ProcessStage


def _make_run(
    run_id: str,
    recipe_id: str,
    d30_value: float | None,
    mass: float = 7.5,
) -> BatteryProcessRun:
    """Helper to create a minimal BatteryProcessRun with controllable D30 value."""
    prov = ProvenanceRecord(
        evidence_kind="PHYSICAL_HISTORICAL",
        source_url="",
        source_doi="",
        source_version="1",
        raw_hashes={},
        adapter_version="2",
        processing_parameters={"partition": "PROSPECTIVE_MODEL_VALIDATION"},
    )
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
    final_kpis: dict[str, MeasurementValue] = {}
    if d30_value is not None:
        final_kpis["discharge_specific_capacity_cycle30_mah_g"] = MeasurementValue(
            value=d30_value,
            units="mAh/g",
            source_name="Discharge Specific Capacity Cycle 30",
        )
    return BatteryProcessRun(
        run_id=run_id,
        cell_id=run_id,
        batch_id=recipe_id,
        chemistry_id="graphite",
        equipment_context={},
        environment_context={},
        stages=stages,
        final_kpis=final_kpis,
        provenance=prov,
    )


class TestZeroD30Semantics:
    """Scientific verification of measured zero vs missing D30."""

    def test_measured_zero_distinct_from_missing(self) -> None:
        """Measured zero must be numeric 0.0, not None / missing."""
        run_zero = _make_run("c1", "r1", d30_value=0.0)
        run_missing = _make_run("c2", "r1", d30_value=None)

        kpi_zero = run_zero.final_kpis.get("discharge_specific_capacity_cycle30_mah_g")
        kpi_missing = run_missing.final_kpis.get("discharge_specific_capacity_cycle30_mah_g")

        assert kpi_zero is not None
        assert kpi_zero.value == 0.0
        assert isinstance(kpi_zero.value, (int, float))

        assert kpi_missing is None

    def test_zero_contributes_to_recipe_mean_and_is_strict_complete(self) -> None:
        """A recipe with [250, 230, 0] must have valid=3, zero=1, mean=160, STRICT_COMPLETE_RECIPE."""
        runs = [
            _make_run("c1", "r-zero-mix", d30_value=250.0),
            _make_run("c2", "r-zero-mix", d30_value=230.0),
            _make_run("c3", "r-zero-mix", d30_value=0.0),
        ]
        adapter = DrakopoulosGraphiteAdapter.__new__(DrakopoulosGraphiteAdapter)
        adapter.load_partition = lambda partition="PROSPECTIVE_MODEL_VALIDATION": runs

        groups = adapter.load_recipe_groups()
        assert len(groups) == 1
        g = groups[0]

        assert g.total_replicates == 3
        assert g.valid_d30_replicates == 3
        assert g.positive_d30_replicates == 2
        assert g.zero_d30_replicates == 1
        assert g.missing_d30_replicates == 0
        assert g.failed_before_d30_replicates == 0
        assert g.recipe_eligibility_status == "STRICT_COMPLETE_RECIPE"
        assert pytest.approx(g.mean_d30_specific_capacity, abs=0.01) == 160.0

    def test_missing_replicate_produces_partial_d30(self) -> None:
        """A recipe with [250, 230, None] must have valid=2, missing=1, PARTIAL_D30."""
        runs = [
            _make_run("c1", "r-partial", d30_value=250.0),
            _make_run("c2", "r-partial", d30_value=230.0),
            _make_run("c3", "r-partial", d30_value=None),
        ]
        adapter = DrakopoulosGraphiteAdapter.__new__(DrakopoulosGraphiteAdapter)
        adapter.load_partition = lambda partition="PROSPECTIVE_MODEL_VALIDATION": runs

        groups = adapter.load_recipe_groups()
        assert len(groups) == 1
        g = groups[0]

        assert g.total_replicates == 3
        assert g.valid_d30_replicates == 2
        assert g.positive_d30_replicates == 2
        assert g.zero_d30_replicates == 0
        assert g.missing_d30_replicates == 1
        assert g.recipe_eligibility_status == "PARTIAL_D30"
        assert g.missing_d30_reason == "MISSING_D30_UNRESOLVED"
        assert pytest.approx(g.mean_d30_specific_capacity, abs=0.01) == 240.0

    def test_all_missing_produces_no_d30(self) -> None:
        """A recipe with all missing replicates must be NO_D30."""
        runs = [
            _make_run("c1", "r-nod30", d30_value=None),
            _make_run("c2", "r-nod30", d30_value=None),
        ]
        adapter = DrakopoulosGraphiteAdapter.__new__(DrakopoulosGraphiteAdapter)
        adapter.load_partition = lambda partition="PROSPECTIVE_MODEL_VALIDATION": runs

        groups = adapter.load_recipe_groups()
        assert len(groups) == 1
        g = groups[0]

        assert g.total_replicates == 2
        assert g.valid_d30_replicates == 0
        assert g.missing_d30_replicates == 2
        assert g.recipe_eligibility_status == "NO_D30"
        assert g.mean_d30_specific_capacity == 0.0

    def test_all_zero_replicates_is_strict_complete_with_zero_mean(self) -> None:
        """A recipe where all replicates are measured zero is STRICT_COMPLETE_RECIPE with mean=0."""
        runs = [
            _make_run("c1", "r-all-zero", d30_value=0.0),
            _make_run("c2", "r-all-zero", d30_value=0.0),
        ]
        adapter = DrakopoulosGraphiteAdapter.__new__(DrakopoulosGraphiteAdapter)
        adapter.load_partition = lambda partition="PROSPECTIVE_MODEL_VALIDATION": runs

        groups = adapter.load_recipe_groups()
        assert len(groups) == 1
        g = groups[0]

        assert g.total_replicates == 2
        assert g.valid_d30_replicates == 2
        assert g.positive_d30_replicates == 0
        assert g.zero_d30_replicates == 2
        assert g.missing_d30_replicates == 0
        assert g.recipe_eligibility_status == "STRICT_COMPLETE_RECIPE"
        assert g.mean_d30_specific_capacity == 0.0
