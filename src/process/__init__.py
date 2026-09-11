"""Default battery-manufacturing process optimization path.

This package deliberately has no dependency on the falsification/HIG stack.
"""

from .contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from .evidence import EvidenceOption, EvidenceReplayResult, replay_blinded_evidence
from .information_horizon import InformationHorizon, InformationHorizonError
from .maspo import ContextualProcessState, MASPOPlan, MASPOProcessOptimizationCoordinator
from .modalities import ModalityObservation, ModalityType
from .robustness import RobustnessReport, evaluate_recipe_robustness
from .stages import ProcessStage

DEFAULT_PRODUCT_MODE = "BATTERY_PROCESS_OPTIMIZATION"

__all__ = [
    "DEFAULT_PRODUCT_MODE",
    "BatteryProcessRun",
    "ContextualProcessState",
    "EvidenceOption",
    "EvidenceReplayResult",
    "InformationHorizon",
    "InformationHorizonError",
    "MASPOPlan",
    "MASPOProcessOptimizationCoordinator",
    "MeasurementValue",
    "ModalityObservation",
    "ModalityType",
    "ParameterValue",
    "ProcessStage",
    "ProvenanceRecord",
    "StageRecord",
    "RobustnessReport",
    "evaluate_recipe_robustness",
    "replay_blinded_evidence",
]
