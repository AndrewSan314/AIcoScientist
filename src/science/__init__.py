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

from src.science.direct_baseline import DirectPerformanceModel
from src.science.evaluation import evaluate_two_stage_model
from src.science.model_bundle import ScientificModelBundle
from src.science.provenance import (
    ScientificModelProvenance,
    build_benchmark_run_manifest,
    get_environment_provenance,
    get_git_provenance,
)
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.two_stage import (
    StageACharacterizationModel,
    StageBPerformanceModel,
    TwoStagePrediction,
    TwoStageScientificModel,
)
from src.science.validation import InformationHorizonError, validate_record_against_spec

__all__ = [
    "DirectPerformanceModel",
    "ExperimentStage",
    "InformationHorizonError",
    "ScientificExperimentRecord",
    "ScientificModelBundle",
    "ScientificModelProvenance",
    "StageACharacterizationModel",
    "StageBPerformanceModel",
    "TwoStagePrediction",
    "TwoStageScientificModel",
    "build_benchmark_run_manifest",
    "evaluate_two_stage_model",
    "get_environment_provenance",
    "get_git_provenance",
    "validate_record_against_spec",
]
