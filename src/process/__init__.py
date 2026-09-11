"""Default battery-manufacturing process optimization path.

This package deliberately has no dependency on the falsification/HIG stack.
"""

from .contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from .information_horizon import InformationHorizon, InformationHorizonError
from .modalities import ModalityObservation, ModalityType
from .stages import ProcessStage

DEFAULT_PRODUCT_MODE = "BATTERY_PROCESS_OPTIMIZATION"

__all__ = [
    "DEFAULT_PRODUCT_MODE",
    "BatteryProcessRun",
    "InformationHorizon",
    "InformationHorizonError",
    "MeasurementValue",
    "ModalityObservation",
    "ModalityType",
    "ParameterValue",
    "ProcessStage",
    "ProvenanceRecord",
    "StageRecord",
]
