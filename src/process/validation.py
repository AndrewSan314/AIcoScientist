from __future__ import annotations

from dataclasses import dataclass

from .contracts import BatteryProcessRun
from .information_horizon import InformationHorizon, InformationHorizonError
from .stages import ProcessStage


@dataclass(frozen=True)
class ProcessValidationReport:
    valid: bool
    errors: tuple[str, ...] = ()


def validate_process_run(run: BatteryProcessRun) -> ProcessValidationReport:
    errors: list[str] = []
    try:
        # Reconstructing exercises the serialization boundary and checks exact identity fields.
        reconstructed = BatteryProcessRun.from_dict(run.to_dict())
        if reconstructed.identity_fingerprint != run.identity_fingerprint:
            errors.append("process run serialization is not identity stable")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    return ProcessValidationReport(valid=not errors, errors=tuple(errors))


def validate_information_horizon(run: BatteryProcessRun, stage: ProcessStage, feature_names: list[str]) -> ProcessValidationReport:
    try:
        InformationHorizon(stage).assert_visible(run, feature_names)
    except InformationHorizonError as exc:
        return ProcessValidationReport(valid=False, errors=(str(exc),))
    return ProcessValidationReport(valid=True)
