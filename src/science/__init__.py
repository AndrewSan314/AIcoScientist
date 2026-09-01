"""AIcoScientist Scientific Closed-Loop Experimentation Framework.

Provides domain-generic abstractions for closed-loop experimentation:
- Process (Controllable variables) -> Characterization (Structure) -> Performance (Properties)
- Append-only experiment ledger with tamper-evident SHA-256 hash chaining
- Multi-channel Stage A (Process -> Characterization) and Stage B (Process + Characterization -> Performance)
- Monte Carlo uncertainty propagation via Law of Total Variance
- Structured deterministic ScientificRationale
- Resumable closed-loop coordinator
"""

from __future__ import annotations

from src.science.actions import (
    ActionRecommendation,
    ActionType,
    CounterfactualAlternative,
    ExperimentActionType,
    ExperimentOutcome,
    ScientificAction,
    normalize_action_type,
)
from src.science.domain import (
    ExperimentExecutor,
    HypothesisProvider,
    HypothesisTrainingContext,
    MaterialDomainAdapter,
    MaterialDomainConfig,
    ModalityDefinition,
    ObjectiveDefinition,
    ObjectiveDirection,
)
from src.science.agents import (
    AgentPerspective,
    EvidenceAuditorAgent,
    ExperimentDesignerAgent,
    FalsificationScientistAgent,
    HypothesisScientistAgent,
    MultiAgentPresentationLayer,
)
from src.science.coordinator import PendingExperimentError, ScientificClosedLoopCoordinator
from src.science.direct_baseline import DirectPerformanceModel
from src.science.discovery_engine import AutonomousDiscoveryEngine
from src.science.decision_engine import ScientificDecisionEngine
from src.science.evaluation import evaluate_two_stage_model
from src.science.experiment_policy import NextBestExperimentPolicy
from src.science.hypotheses import (
    HypothesisEngine,
    HypothesisStatus,
    ScientificHypothesis,
    get_default_hypotheses,
)
from src.science.ledger import ExperimentLedger
from src.science.model_bundle import ScientificModelBundle
from src.science.provenance import (
    ScientificModelProvenance,
    build_benchmark_run_manifest,
    get_environment_provenance,
    get_git_provenance,
)
from src.science.rationale import ScientificRationale, generate_scientific_rationale
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.synthetic import SyntheticExperimentOracle, SyntheticScienceAdapter
from src.science.two_stage import (
    StageACharacterizationModel,
    StageBPerformanceModel,
    TwoStagePrediction,
    TwoStageScientificModel,
)
from src.science.validation import InformationHorizonError, validate_record_against_spec
from src.science.xrd_representation import XRDRepresentationExtractor

__all__ = [
    "ActionRecommendation",
    "ActionType",
    "AgentPerspective",
    "AutonomousDiscoveryEngine",
    "CounterfactualAlternative",
    "DirectPerformanceModel",
    "EvidenceAuditorAgent",
    "ExperimentActionType",
    "ExperimentDesignerAgent",
    "ExperimentExecutor",
    "ExperimentLedger",
    "ExperimentOutcome",
    "ExperimentStage",
    "FalsificationScientistAgent",
    "HypothesisEngine",
    "HypothesisProvider",
    "HypothesisScientistAgent",
    "HypothesisStatus",
    "HypothesisTrainingContext",
    "InformationHorizonError",
    "MaterialDomainAdapter",
    "MaterialDomainConfig",
    "ModalityDefinition",
    "MultiAgentPresentationLayer",
    "NextBestExperimentPolicy",
    "ObjectiveDefinition",
    "ObjectiveDirection",
    "PendingExperimentError",
    "ScientificAction",
    "ScientificClosedLoopCoordinator",
    "ScientificDecisionEngine",
    "ScientificExperimentRecord",
    "ScientificHypothesis",
    "ScientificModelBundle",
    "ScientificModelProvenance",
    "ScientificRationale",
    "StageACharacterizationModel",
    "StageBPerformanceModel",
    "SyntheticExperimentOracle",
    "SyntheticScienceAdapter",
    "TwoStagePrediction",
    "TwoStageScientificModel",
    "XRDRepresentationExtractor",
    "build_benchmark_run_manifest",
    "evaluate_two_stage_model",
    "generate_scientific_rationale",
    "get_default_hypotheses",
    "get_environment_provenance",
    "get_git_provenance",
    "normalize_action_type",
    "validate_record_against_spec",
]
