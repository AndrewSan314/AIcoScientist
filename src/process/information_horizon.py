from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .contracts import BatteryProcessRun, MeasurementValue, ParameterValue, StageRecord
from .modalities import ModalityObservation
from .stages import ProcessStage, stage_precedes


class InformationHorizonError(ValueError):
    pass


class DecisionHorizon(str, Enum):
    """Canonical decision horizon enumeration."""

    STAGE_LOCAL = "STAGE_LOCAL"
    PRE_MANUFACTURING_RECIPE_SELECTION = "PRE_MANUFACTURING_RECIPE_SELECTION"


@dataclass(frozen=True)
class HorizonView:
    decision_stage: ProcessStage | None
    controls: Mapping[str, ParameterValue]
    intermediate_properties: Mapping[str, MeasurementValue]
    modalities: tuple[ModalityObservation, ...]
    source_stage_ids: tuple[str, ...]
    source_stages: tuple[StageRecord, ...] = ()


@dataclass(frozen=True)
class PreManufacturingRecipeSelectionHorizon:
    """Strict pre-manufacturing recipe selection horizon.

    Scientific invariant:
        future planned control != future observation
    All planned manufacturing controls (formulation, mixing solids, coating gap/speed,
    drying temperature, calendering) are visible before execution begins.
    All post-manufacturing measurements (mass, thickness, porosity, D30, KPIs) are hidden.
    """

    decision_horizon: str = DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION.value

    def visible_planned_controls(self, run: BatteryProcessRun) -> dict[str, ParameterValue]:
        """Extract all planned process controls across all stages."""
        controls: dict[str, ParameterValue] = {}
        for stage in run.stages:
            for name, val in stage.controls.items():
                controls[name] = val
        return controls

    def hidden_observations(self, run: BatteryProcessRun) -> dict[str, MeasurementValue]:
        """Extract all post-manufacturing measurements and final KPIs that must remain hidden."""
        hidden: dict[str, MeasurementValue] = {}
        for stage in run.stages:
            for name, val in stage.intermediate_properties.items():
                hidden[name] = val
        for name, val in run.final_kpis.items():
            hidden[name] = val
        return hidden

    def project_for_recipe_selection(self, run: BatteryProcessRun) -> HorizonView:
        """Project a process run for pre-manufacturing recipe selection."""
        ctrls = self.visible_planned_controls(run)
        mapped_controls: dict[str, ParameterValue] = dict(ctrls)
        for stage in run.stages:
            prefix = f"{stage.stage_type.value.lower()}."
            for name, val in stage.controls.items():
                mapped_controls[f"{prefix}{name}"] = val
        return HorizonView(
            decision_stage=ProcessStage.FORMULATION,
            controls=mapped_controls,
            intermediate_properties={},
            modalities=(),
            source_stage_ids=tuple(s.stage_id for s in run.stages),
            source_stages=tuple(run.stages),
        )

    def project(self, run: BatteryProcessRun) -> HorizonView:
        return self.project_for_recipe_selection(run)

    def assert_visible(self, run: BatteryProcessRun, feature_names: Sequence[str]) -> None:
        hidden = self.hidden_observations(run)
        forbidden = [name for name in feature_names if name in hidden or name.split(".")[-1] in hidden]
        if forbidden:
            raise InformationHorizonError(
                f"Features unavailable before manufacturing execution: {sorted(forbidden)}. "
                "Post-process measurements and final KPIs must remain hidden."
            )


@dataclass(frozen=True)
class InformationHorizon:
    """Strict pre-decision visibility boundary for a manufacturing stage or recipe selection."""

    stage: ProcessStage | DecisionHorizon | str
    include_decision_stage_controls: bool = False

    @classmethod
    def pre_manufacturing_recipe_selection(cls) -> PreManufacturingRecipeSelectionHorizon:
        return PreManufacturingRecipeSelectionHorizon()

    def is_recipe_selection(self) -> bool:
        return (
            self.stage == DecisionHorizon.PRE_MANUFACTURING_RECIPE_SELECTION
            or str(self.stage) == "PRE_MANUFACTURING_RECIPE_SELECTION"
        )

    def can_observe_stage(self, stage: ProcessStage) -> bool:
        """Returns True if given stage is observable under this horizon."""
        if self.is_recipe_selection():
            return True
        if self.include_decision_stage_controls and stage == self.stage:
            return True
        return stage_precedes(stage, self.stage)  # type: ignore[arg-type]

    def visible_stages(self, run: BatteryProcessRun) -> list[StageRecord]:
        if self.is_recipe_selection():
            return list(run.stages)
        return [record for record in run.stages if stage_precedes(record.stage_type, self.stage)]  # type: ignore[arg-type]

    def visible_planned_controls(self, run: BatteryProcessRun) -> dict[str, ParameterValue]:
        return PreManufacturingRecipeSelectionHorizon().visible_planned_controls(run)

    def hidden_observations(self, run: BatteryProcessRun) -> dict[str, MeasurementValue]:
        return PreManufacturingRecipeSelectionHorizon().hidden_observations(run)

    def project_for_recipe_selection(self, run: BatteryProcessRun) -> HorizonView:
        return PreManufacturingRecipeSelectionHorizon().project_for_recipe_selection(run)

    def project(self, run: BatteryProcessRun) -> HorizonView:
        if self.is_recipe_selection():
            return PreManufacturingRecipeSelectionHorizon().project(run)
        controls: dict[str, ParameterValue] = {}
        measurements: dict[str, MeasurementValue] = {}
        modalities: list[ModalityObservation] = []
        stages = self.visible_stages(run)
        for record in stages:
            prefix = f"{record.stage_type.value.lower()}."
            controls.update({prefix + name: value for name, value in record.controls.items()})
            measurements.update({prefix + name: value for name, value in record.intermediate_properties.items()})
            modalities.extend(record.modalities)
        return HorizonView(
            decision_stage=self.stage if isinstance(self.stage, ProcessStage) else None,
            controls=controls,
            intermediate_properties=measurements,
            modalities=tuple(modalities),
            source_stage_ids=tuple(record.stage_id for record in stages),
            source_stages=tuple(stages),
        )

    def assert_visible(self, run: BatteryProcessRun, feature_names: Sequence[str]) -> None:
        if self.is_recipe_selection():
            PreManufacturingRecipeSelectionHorizon().assert_visible(run, feature_names)
            return
        allowed = set(self.project(run).controls) | set(self.project(run).intermediate_properties)
        forbidden = [name for name in feature_names if name not in allowed]
        if forbidden:
            stage_name = self.stage.value if hasattr(self.stage, "value") else str(self.stage)
            raise InformationHorizonError(
                f"Features unavailable before {stage_name}: {sorted(forbidden)}. Final KPIs are never horizon features."
            )
