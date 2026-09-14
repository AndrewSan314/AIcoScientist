"""Canonical, horizon-safe bridge from BPSS datasets to scalar surrogates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.datasets.battery_process import (
    ArtisticSimulationAdapter,
    BatteryProcessDatasetAdapter,
    DrakopoulosGraphiteAdapter,
    NaIonHTEAdapter,
    ProcessPredictionTask,
    WarwickNMC622Adapter,
    WarwickUltrasoundAdapter,
)
from src.process.contracts import BatteryProcessRun
from src.process.information_horizon import InformationHorizon
from src.process.modalities import ModalityObservation, source_modality_fingerprint
from src.process.stages import ProcessStage

from .core import DatasetManifest, DatasetValidationReport, ProcessSurrogateSample


_REGISTERED_DATASETS = {
    "drakopoulos_graphite": DrakopoulosGraphiteAdapter,
    "warwick_nmc622": WarwickNMC622Adapter,
    "warwick_ultrasound": WarwickUltrasoundAdapter,
    "naion_hte": NaIonHTEAdapter,
    "artistic": ArtisticSimulationAdapter,
}


def registered_battery_datasets() -> tuple[str, ...]:
    return tuple(sorted(_REGISTERED_DATASETS))


def registered_battery_dataset(dataset_id: str, *, root: str | Path | None = None) -> BatteryProcessDatasetAdapter:
    try:
        adapter_type = _REGISTERED_DATASETS[dataset_id]
    except KeyError as exc:
        raise ValueError(f"dataset must be one of {', '.join(registered_battery_datasets())}") from exc
    return adapter_type(root) if root is not None else adapter_type()


def _numeric(values: Mapping[str, Any], *, prefix: str) -> dict[str, float]:
    return {prefix + name: float(value.value) for name, value in values.items() if isinstance(value.value, (int, float))}


def _modality_values(observations: Sequence[ModalityObservation]) -> tuple[dict[str, float], tuple[dict[str, Any], ...]]:
    values: dict[str, float] = {}
    provenance: list[dict[str, Any]] = []
    for observation in observations:
        name = f"{observation.observed_at_stage.value.lower()}.{observation.modality_type.value.lower()}"
        values[name + ".observed"] = float(not observation.is_missing)
        provenance.append({"name": name, "modality_id": observation.modality_id, "missing_reason": observation.missing_reason, "source_fingerprint": source_modality_fingerprint(observation) if not observation.is_missing else None})
        if isinstance(observation.values, (int, float)):
            values[name + ".value"] = float(observation.values)
        elif isinstance(observation.values, Mapping):
            values.update({name + "." + str(key): float(value) for key, value in observation.values.items() if isinstance(value, (int, float))})
    return values, tuple(provenance)


class BatteryProcessSurrogateAdapter:
    """One registered BPSS source, one stage and one final KPI; never pooled."""

    ADAPTER_VERSION = "1"

    def __init__(self, source: BatteryProcessDatasetAdapter, *, task: ProcessPredictionTask) -> None:
        self.source, self.task, self.horizon = source, task, InformationHorizon(task.decision_stage)

    @classmethod
    def from_registered(cls, dataset_id: str, *, stage: ProcessStage, target: str, root: str | Path | None = None) -> "BatteryProcessSurrogateAdapter":
        return cls(registered_battery_dataset(dataset_id, root=root), task=ProcessPredictionTask(target, stage))

    def _run_sample(self, run: BatteryProcessRun) -> ProcessSurrogateSample | None:
        outcome = run.final_kpis.get(self.task.target)
        if outcome is None or not isinstance(outcome.value, (int, float)):
            return None
        view = self.horizon.project(run)
        controls = _numeric(view.controls, prefix="")
        observations = _numeric(view.intermediate_properties, prefix="")
        modality_state, modalities = _modality_values(view.modalities)
        if not controls and not observations and not modality_state:
            return None
        metadata = self.source.metadata()
        parameters = run.provenance.processing_parameters
        fidelity = str(parameters.get("fidelity_mode") or run.provenance.evidence_kind)
        return ProcessSurrogateSample(
            sample_id=run.run_id, run_id=run.run_id, recipe_id=run.batch_id or run.run_id,
            source_dataset=metadata.dataset_id, source_evidence_kind=run.provenance.evidence_kind,
            stage=self.task.decision_stage, group_id=run.batch_id or run.run_id,
            controls=controls, observations=observations, modality_state=modality_state,
            fidelity=fidelity, requested_horizon=int(parameters["requested_slurry_steps"]) if parameters.get("requested_slurry_steps") else None,
            physics_config_fingerprint=parameters.get("physics_config_fingerprint"), targets={self.task.target: float(outcome.value)},
            provenance={"run_identity_fingerprint": run.identity_fingerprint, "source_url": run.provenance.source_url, "source_doi": run.provenance.source_doi, "source_version": run.provenance.source_version, "raw_hashes": dict(run.provenance.raw_hashes), "horizon": {"decision_stage": self.task.decision_stage.value, "source_stage_ids": view.source_stage_ids}, "modalities": modalities},
        )

    def samples(self) -> list[ProcessSurrogateSample]:
        return [sample for run in self.source.load_runs() if (sample := self._run_sample(run)) is not None]

    def manifest(self) -> DatasetManifest:
        metadata, runs, samples = self.source.metadata(), self.source.load_runs(), self.samples()
        hashes = {f"{run.run_id}:{name}": digest for run in runs for name, digest in run.provenance.raw_hashes.items()}
        return DatasetManifest(metadata.dataset_id, metadata.version, metadata.source_doi, hashes, type(self).__name__, self.ADAPTER_VERSION, metadata.chemistry, (self.task.decision_stage.value,), tuple(sorted({name for sample in samples for name in sample.controls})), tuple(sorted({name for sample in samples for name in sample.observations})), (self.task.target,), tuple(sorted({name for sample in samples for name in sample.modality_state})), tuple(sorted({sample.fidelity for sample in samples} or {metadata.evidence_kind})), len(samples), "batch_id_or_run_id", "GROUP_ISOLATED_BY_SOURCE_BATCH_OR_RUN", {self.task.target: next((value.units or "UNSPECIFIED" for run in runs if (value := run.final_kpis.get(self.task.target)) is not None), "UNSPECIFIED")})

    def validate(self, *, targets: Sequence[str] | None = None) -> DatasetValidationReport:
        if targets is not None and tuple(targets) != (self.task.target,):
            return DatasetValidationReport("INVALID", 0, ("bridge supports exactly its configured final target",))
        source_report = self.source.validate()
        if not source_report.valid:
            return DatasetValidationReport("INVALID", 0, tuple(source_report.errors))
        try:
            samples = self.samples()
        except (OSError, ValueError) as exc:
            return DatasetValidationReport("INVALID", 0, (str(exc),))
        errors = []
        if not samples:
            errors.append("no horizon-valid rows with the configured final target")
        if len({sample.group_id for sample in samples}) < 3:
            errors.append("at least three source groups are required for train/validation/test")
        return DatasetValidationReport("VALID" if not errors else "INVALID", len(samples), tuple(errors), target_availability={self.task.target: len(samples)}, fidelity_distribution={name: sum(sample.fidelity == name for sample in samples) for name in sorted({sample.fidelity for sample in samples})}, stage_distribution={self.task.decision_stage.value: len(samples)})
