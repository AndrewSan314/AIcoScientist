from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import BatteryProcessRun, MeasurementValue, ParameterValue, StageRecord
from .modalities import ModalityObservation
from .stages import ProcessStage, stage_precedes


class InformationHorizonError(ValueError):
    pass


@dataclass(frozen=True)
class HorizonView:
    decision_stage: ProcessStage
    controls: Mapping[str, ParameterValue]
    intermediate_properties: Mapping[str, MeasurementValue]
    modalities: tuple[ModalityObservation, ...]
    source_stage_ids: tuple[str, ...]


@dataclass(frozen=True)
class InformationHorizon:
    """Strict pre-decision visibility boundary for a manufacturing stage."""

    stage: ProcessStage
    include_decision_stage_controls: bool = False

    def visible_stages(self, run: BatteryProcessRun) -> list[StageRecord]:
        return [record for record in run.stages if stage_precedes(record.stage_type, self.stage)]

    def project(self, run: BatteryProcessRun) -> HorizonView:
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
            decision_stage=self.stage,
            controls=controls,
            intermediate_properties=measurements,
            modalities=tuple(modalities),
            source_stage_ids=tuple(record.stage_id for record in stages),
        )

    def assert_visible(self, run: BatteryProcessRun, feature_names: list[str]) -> None:
        allowed = set(self.project(run).controls) | set(self.project(run).intermediate_properties)
        forbidden = [name for name in feature_names if name not in allowed]
        if forbidden:
            raise InformationHorizonError(
                f"Features unavailable before {self.stage.value}: {sorted(forbidden)}. Final KPIs are never horizon features."
            )
