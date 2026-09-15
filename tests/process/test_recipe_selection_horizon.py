"""Tests verifying PreManufacturingRecipeSelectionHorizon visibility and invariant checks."""

from __future__ import annotations
import pytest

from src.process.contracts import (
    BatteryProcessRun,
    MeasurementValue,
    ParameterValue,
    ProvenanceRecord,
    StageRecord,
)
from src.process.information_horizon import (
    DecisionHorizon,
    InformationHorizon,
    InformationHorizonError,
    PreManufacturingRecipeSelectionHorizon,
)
from src.process.stages import ProcessStage


def _make_full_run(run_id: str = "run-001") -> BatteryProcessRun:
    """Create a realistic BatteryProcessRun with controls and metrology across stages."""
    prov = ProvenanceRecord(
        evidence_kind="PHYSICAL_HISTORICAL",
        source_url="",
        source_doi="",
        source_version="1",
        raw_hashes={},
        adapter_version="2",
        processing_parameters={"partition": "PROSPECTIVE_MODEL_VALIDATION"},
    )
    stages = [
        StageRecord(
            stage_id=f"{run_id}:formulation",
            stage_type=ProcessStage.FORMULATION,
            sequence_index=0,
            controls={
                "active_material_fraction_pct": ParameterValue(95.0, "%"),
                "conductive_additive_fraction_pct": ParameterValue(1.0, "%"),
                "binder_cmc_fraction_pct": ParameterValue(2.0, "%"),
                "binder_sbr_fraction_pct": ParameterValue(2.0, "%"),
            },
            intermediate_properties={},
            modalities=[],
            provenance=prov,
        ),
        StageRecord(
            stage_id=f"{run_id}:mixing",
            stage_type=ProcessStage.MIXING,
            sequence_index=1,
            controls={
                "mixing_solids_pct": ParameterValue(49.5, "%"),
            },
            intermediate_properties={
                "slurry_viscosity_pa_s": MeasurementValue(2.5, "Pa.s"),
            },
            modalities=[],
            upstream_stage_id=f"{run_id}:formulation",
            provenance=prov,
        ),
        StageRecord(
            stage_id=f"{run_id}:coating",
            stage_type=ProcessStage.COATING,
            sequence_index=2,
            controls={
                "coating_speed_m_per_min": ParameterValue(0.2, "m/min"),
                "coating_gap_um": ParameterValue(150.0, "um"),
            },
            intermediate_properties={
                "wet_thickness_um": MeasurementValue(145.0, "um"),
            },
            modalities=[],
            upstream_stage_id=f"{run_id}:mixing",
            provenance=prov,
        ),
        StageRecord(
            stage_id=f"{run_id}:drying",
            stage_type=ProcessStage.DRYING,
            sequence_index=3,
            controls={
                "drying_temperature_c": ParameterValue(80.0, "C"),
            },
            intermediate_properties={
                "electrode_thickness_um": MeasurementValue(65.0, "um"),
            },
            modalities=[],
            upstream_stage_id=f"{run_id}:coating",
            provenance=prov,
        ),
        StageRecord(
            stage_id=f"{run_id}:calendering",
            stage_type=ProcessStage.CALENDERING,
            sequence_index=4,
            controls={
                "calendering_applied": ParameterValue(1.0, "binary"),
            },
            intermediate_properties={
                "active_mass_mg": MeasurementValue(7.5, "mg"),
                "calendered_thickness_um": MeasurementValue(45.0, "um"),
                "porosity_pct": MeasurementValue(32.0, "%"),
            },
            modalities=[],
            upstream_stage_id=f"{run_id}:drying",
            provenance=prov,
        ),
    ]
    final_kpis = {
        "discharge_specific_capacity_cycle30_mah_g": MeasurementValue(350.0, "mAh/g"),
        "cell_capacity_mah": MeasurementValue(2.625, "mAh"),
    }
    return BatteryProcessRun(
        run_id=run_id,
        cell_id=run_id,
        batch_id="recipe-01",
        chemistry_id="graphite",
        equipment_context={},
        environment_context={},
        stages=stages,
        final_kpis=final_kpis,
        provenance=prov,
    )


class TestRecipeSelectionHorizon:
    """Verify PRE_MANUFACTURING_RECIPE_SELECTION horizon semantics."""

    def test_decision_horizon_enum(self) -> None:
        assert DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION.value == "PRE_MANUFACTURING_RECIPE_SELECTION"
        assert DecisionHorizon.STAGE_LOCAL.value == "STAGE_LOCAL"

    def test_all_planned_controls_are_visible(self) -> None:
        run = _make_full_run()
        horizon = PreManufacturingRecipeSelectionHorizon()
        ctrls = horizon.visible_planned_controls(run)

        expected_planned_controls = {
            "active_material_fraction_pct",
            "conductive_additive_fraction_pct",
            "binder_cmc_fraction_pct",
            "binder_sbr_fraction_pct",
            "mixing_solids_pct",
            "coating_speed_m_per_min",
            "coating_gap_um",
            "drying_temperature_c",
            "calendering_applied",
        }
        for expected in expected_planned_controls:
            assert expected in ctrls, f"Expected planned control {expected} to be visible"
            assert isinstance(ctrls[expected], ParameterValue)

    def test_post_process_observations_are_hidden(self) -> None:
        run = _make_full_run()
        horizon = PreManufacturingRecipeSelectionHorizon()
        hidden = horizon.hidden_observations(run)

        expected_hidden = {
            "slurry_viscosity_pa_s",
            "wet_thickness_um",
            "electrode_thickness_um",
            "active_mass_mg",
            "calendered_thickness_um",
            "porosity_pct",
            "discharge_specific_capacity_cycle30_mah_g",
            "cell_capacity_mah",
        }
        for exp in expected_hidden:
            assert exp in hidden, f"Expected post-process observation {exp} to be hidden"

        view = horizon.project_for_recipe_selection(run)
        assert len(view.intermediate_properties) == 0
        assert len(view.modalities) == 0

    def test_assert_visible_rejects_hidden_observations(self) -> None:
        run = _make_full_run()
        horizon = PreManufacturingRecipeSelectionHorizon()

        # Planned controls should pass
        horizon.assert_visible(run, ["coating_speed_m_per_min", "coating_gap_um", "calendering_applied"])

        # Hidden observations must raise InformationHorizonError
        with pytest.raises(InformationHorizonError, match="Features unavailable before manufacturing execution"):
            horizon.assert_visible(run, ["active_mass_mg"])

        with pytest.raises(InformationHorizonError, match="Features unavailable before manufacturing execution"):
            horizon.assert_visible(run, ["porosity_pct"])

        with pytest.raises(InformationHorizonError, match="Features unavailable before manufacturing execution"):
            horizon.assert_visible(run, ["discharge_specific_capacity_cycle30_mah_g"])

    def test_information_horizon_compatibility(self) -> None:
        """InformationHorizon initialized with DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION."""
        run = _make_full_run()
        horizon = InformationHorizon(DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION)
        assert horizon.is_recipe_selection()

        view = horizon.project(run)
        assert "coating_gap_um" in view.controls
        assert "drying_temperature_c" in view.controls
        assert "calendering_applied" in view.controls
        assert len(view.intermediate_properties) == 0

        with pytest.raises(InformationHorizonError):
            horizon.assert_visible(run, ["active_mass_mg"])
