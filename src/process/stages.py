from __future__ import annotations

from enum import Enum


class ProcessStage(str, Enum):
    """Canonical ordered battery-manufacturing stages."""

    FORMULATION = "FORMULATION"
    MIXING = "MIXING"
    COATING = "COATING"
    DRYING = "DRYING"
    CALENDERING = "CALENDERING"
    ASSEMBLY = "ASSEMBLY"
    ELECTROLYTE_WETTING = "ELECTROLYTE_WETTING"
    FORMATION = "FORMATION"
    FINAL_CHARACTERIZATION = "FINAL_CHARACTERIZATION"


STAGE_ORDER = {stage: index for index, stage in enumerate(ProcessStage)}


def stage_precedes(left: ProcessStage, right: ProcessStage, *, allow_equal: bool = False) -> bool:
    return STAGE_ORDER[left] <= STAGE_ORDER[right] if allow_equal else STAGE_ORDER[left] < STAGE_ORDER[right]
