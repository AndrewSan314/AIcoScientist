"""Default battery-manufacturing process optimization path.

This package deliberately has no dependency on the falsification/HIG stack.
"""

from .contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from .evidence import EvidenceOption, EvidenceReplayResult, replay_blinded_evidence
from .information_horizon import InformationHorizon, InformationHorizonError
from .maspo import ContextualProcessState, MASPOPlan, MASPOProcessOptimizationCoordinator
from .modalities import ModalityObservation, ModalitySlotSpec, ModalityType, SourceBoundModalityInput, source_modality_fingerprint, tensor_fingerprint
from .models import LegalStageTransition, build_legal_stage_transitions, encode_legal_multimodal_state
from .optimization.state import ModelValidationStatus, OptimizationState, canonical_control_action_id, context_provenance_fingerprint, contextual_candidate_instance_id
from .robustness import RobustnessReport, evaluate_recipe_robustness
from .stages import ProcessStage

DEFAULT_PRODUCT_MODE = "BATTERY_PROCESS_OPTIMIZATION"

__all__ = [
    "DEFAULT_PRODUCT_MODE",
    "BatteryProcessRun",
    "ContextualProcessState",
    "OptimizationState",
    "ModelValidationStatus",
    "LegalStageTransition",
    "EvidenceOption",
    "EvidenceReplayResult",
    "InformationHorizon",
    "InformationHorizonError",
    "MASPOPlan",
    "MASPOProcessOptimizationCoordinator",
    "MeasurementValue",
    "ModalityObservation",
    "ModalitySlotSpec",
    "ModalityType",
    "SourceBoundModalityInput",
    "ParameterValue",
    "ProcessStage",
    "ProvenanceRecord",
    "StageRecord",
    "RobustnessReport",
    "evaluate_recipe_robustness",
    "canonical_control_action_id",
    "context_provenance_fingerprint",
    "contextual_candidate_instance_id",
    "source_modality_fingerprint",
    "tensor_fingerprint",
    "encode_legal_multimodal_state",
    "build_legal_stage_transitions",
    "replay_blinded_evidence",
]
