"""Checkpointed offline process-control lifecycle, independent of legacy HIG state."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .contracts import BatteryProcessRun, StageRecord
from .maspo import MASPOPlan, MASPOProcessOptimizationCoordinator
from .optimization.process_objective import ProcessOptimizationObjective
from .optimization.process_space import ProcessSearchSpace
from .stages import ProcessStage
from .validation import validate_process_run


class ProcessLifecycleMode(StrEnum):
    OFFLINE_REPLAY = "OFFLINE_REPLAY"
    SIMULATED_PHYSICS = "SIMULATED_PHYSICS"
    LIVE_PROCESS = "LIVE_PROCESS"


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _objective_fingerprint(objective: ProcessOptimizationObjective) -> str:
    return _fingerprint({"objectives": [item.__dict__ for item in objective.objectives], "constraints": [item.__dict__ for item in objective.constraints], "reference_point": objective.reference_point, "feasibility_threshold": objective.feasibility_threshold})


def _proposal_payload(plan: MASPOPlan) -> dict[str, Any]:
    proposal = plan.next_control
    return {"proposal_id": proposal.proposal_id, "stage": proposal.stage.value if proposal.stage else None, "controls": proposal.controls, "predicted_outputs": {name: value.__dict__ for name, value in proposal.predicted_outputs.items()}, "feasibility_probability": proposal.feasibility_probability, "acquisition_value": proposal.acquisition_value, "model_version": proposal.model_version, "data_fingerprint": proposal.data_fingerprint, "source_recipe_id": proposal.source_recipe_id, "candidate_instance_id": proposal.candidate_instance_id, "provenance": proposal.provenance}


@dataclass(frozen=True)
class ProcessLifecycleCheckpoint:
    run_id: str
    current_stage: str
    process_run_fingerprint: str
    source_data_fingerprint: str
    model_artifact_fingerprint: str
    process_graph_fingerprint: str | None
    optimizer_backend_version: str
    objective_constraint_fingerprint: str
    proposal: Mapping[str, Any]
    evidence_acquisition_decision: Mapping[str, Any] | None
    revealed_observation_ids: tuple[str, ...]
    seed: int | None
    timestamp: str
    completed_stage_ids: tuple[str, ...]
    mode: ProcessLifecycleMode

    def save(self, path: str | Path) -> None:
        destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.__dict__, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> "ProcessLifecycleCheckpoint":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("process lifecycle checkpoint must be a mapping")
        raw["revealed_observation_ids"] = tuple(raw.get("revealed_observation_ids", ()))
        raw["completed_stage_ids"] = tuple(raw.get("completed_stage_ids", ()))
        raw["mode"] = ProcessLifecycleMode(raw["mode"])
        return cls(**raw)

    def verify(self, *, run: BatteryProcessRun, source_data_fingerprint: str, model_artifact_fingerprint: str, process_graph_fingerprint: str | None, objective: ProcessOptimizationObjective) -> None:
        expected = {"run_id": run.run_id, "process_run_fingerprint": run.identity_fingerprint, "source_data_fingerprint": source_data_fingerprint, "model_artifact_fingerprint": model_artifact_fingerprint, "process_graph_fingerprint": process_graph_fingerprint, "objective_constraint_fingerprint": _objective_fingerprint(objective)}
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"process lifecycle checkpoint {name} mismatch")


class ProcessClosedLoopCoordinator:
    """Persist the next action before any reveal; LIVE_PROCESS never invokes hardware."""

    def __init__(self, coordinator: MASPOProcessOptimizationCoordinator | None = None) -> None:
        self.coordinator = coordinator or MASPOProcessOptimizationCoordinator()

    def propose_next(
        self, *, run: BatteryProcessRun, current_stage: ProcessStage, remaining_control_spaces: dict[ProcessStage, ProcessSearchSpace], observations: pd.DataFrame | dict[ProcessStage, pd.DataFrame], objective: ProcessOptimizationObjective, source_data_fingerprint: str, model_artifact_fingerprint: str, checkpoint_path: str | Path, mode: ProcessLifecycleMode = ProcessLifecycleMode.OFFLINE_REPLAY, evidence_acquisition_decision: Mapping[str, Any] | None = None, seed: int | None = None,
    ) -> tuple[MASPOPlan, ProcessLifecycleCheckpoint]:
        report = validate_process_run(run)
        if not report.valid:
            raise ValueError("invalid process run: " + "; ".join(report.errors))
        plan = self.coordinator.optimize_remaining_process(current_state=run, current_stage=current_stage, remaining_control_spaces=remaining_control_spaces, observations=observations, objective=objective, seed=seed)
        checkpoint = ProcessLifecycleCheckpoint(run.run_id, current_stage.value, run.identity_fingerprint, source_data_fingerprint, model_artifact_fingerprint, plan.graph_provenance.get("process_graph_fingerprint"), plan.next_control.model_version, _objective_fingerprint(objective), _proposal_payload(plan), dict(evidence_acquisition_decision) if evidence_acquisition_decision else None, (), seed, datetime.now(UTC).isoformat(), tuple(stage.stage_id for stage in run.stages), mode)
        checkpoint.save(checkpoint_path)
        return plan, checkpoint

    def resume(self, checkpoint_path: str | Path, *, run: BatteryProcessRun, source_data_fingerprint: str, model_artifact_fingerprint: str, objective: ProcessOptimizationObjective) -> ProcessLifecycleCheckpoint:
        checkpoint = ProcessLifecycleCheckpoint.load(checkpoint_path)
        checkpoint.verify(run=run, source_data_fingerprint=source_data_fingerprint, model_artifact_fingerprint=model_artifact_fingerprint, process_graph_fingerprint=self.coordinator.graph_provenance.get("process_graph_fingerprint"), objective=objective)
        return checkpoint

    @staticmethod
    def ingest_observation(checkpoint: ProcessLifecycleCheckpoint, run: BatteryProcessRun, observation: StageRecord, *, evidence_id: str) -> tuple[BatteryProcessRun, ProcessLifecycleCheckpoint]:
        if evidence_id in checkpoint.revealed_observation_ids:
            raise ValueError("process lifecycle checkpoint refuses to reveal an observation twice")
        if observation.stage_id in {item.stage_id for item in run.stages} or observation.sequence_index <= max(item.sequence_index for item in run.stages):
            raise ValueError("ingested observation must be one new downstream process stage")
        updated = replace(run, stages=[*run.stages, observation])
        report = validate_process_run(updated)
        if not report.valid:
            raise ValueError("ingested process observation is invalid: " + "; ".join(report.errors))
        return updated, replace(checkpoint, process_run_fingerprint=updated.identity_fingerprint, current_stage=observation.stage_type.value, revealed_observation_ids=(*checkpoint.revealed_observation_ids, evidence_id), completed_stage_ids=(*checkpoint.completed_stage_ids, observation.stage_id), timestamp=datetime.now(UTC).isoformat())
