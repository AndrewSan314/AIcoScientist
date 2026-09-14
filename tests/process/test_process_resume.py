from pathlib import Path

import pytest

from src.process.contracts import MeasurementValue, ParameterValue, StageRecord
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.lifecycle import ProcessClosedLoopCoordinator, ProcessLifecycleMode
from src.process.maspo import MASPOProcessOptimizationCoordinator
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.stages import ProcessStage
from .conftest import process_run
from .test_maspo_control import FirstCandidateBackend, _observations, _space


def _objective():
    return ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")])


def test_process_checkpoint_resume_verifies_all_fingerprints(tmp_path: Path):
    lifecycle = ProcessClosedLoopCoordinator(MASPOProcessOptimizationCoordinator(ProcessOptimizationCoordinator(scalar_backend=FirstCandidateBackend())))
    run = process_run(); path = tmp_path / "checkpoint.json"
    _, checkpoint = lifecycle.propose_next(run=run, current_stage=ProcessStage.COATING, remaining_control_spaces={ProcessStage.DRYING: _space()}, observations=_observations(), objective=_objective(), source_data_fingerprint="source-v1", model_artifact_fingerprint="artifact-v1", checkpoint_path=path, mode=ProcessLifecycleMode.OFFLINE_REPLAY, seed=7)
    resumed = lifecycle.resume(path, run=run, source_data_fingerprint="source-v1", model_artifact_fingerprint="artifact-v1", objective=_objective())
    assert resumed == checkpoint and resumed.proposal["stage"] == "DRYING"
    with pytest.raises(ValueError, match="source_data_fingerprint"):
        lifecycle.resume(path, run=run, source_data_fingerprint="source-v2", model_artifact_fingerprint="artifact-v1", objective=_objective())


def test_ingest_is_deterministic_and_refuses_duplicate_evidence(tmp_path: Path):
    lifecycle = ProcessClosedLoopCoordinator(MASPOProcessOptimizationCoordinator(ProcessOptimizationCoordinator(scalar_backend=FirstCandidateBackend())))
    run = process_run(); _, checkpoint = lifecycle.propose_next(run=run, current_stage=ProcessStage.COATING, remaining_control_spaces={ProcessStage.DRYING: _space()}, observations=_observations(), objective=_objective(), source_data_fingerprint="source-v1", model_artifact_fingerprint="artifact-v1", checkpoint_path=tmp_path / "checkpoint.json")
    observation = StageRecord("formation", ProcessStage.FORMATION, 7, {"current": ParameterValue(1.0)}, {"formation_capacity": MeasurementValue(150.0)}, [], "cal", provenance=run.provenance)
    updated, advanced = lifecycle.ingest_observation(checkpoint, run, observation, evidence_id="evidence-1")
    assert updated.stages[-1].stage_id == "formation" and advanced.revealed_observation_ids == ("evidence-1",)
    with pytest.raises(ValueError, match="twice"):
        lifecycle.ingest_observation(advanced, updated, observation, evidence_id="evidence-1")
