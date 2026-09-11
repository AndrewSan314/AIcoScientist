from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd

from src.process.contracts import BatteryProcessRun
from src.process.information_horizon import InformationHorizon
from src.process.optimization.process_space import ProcessSearchSpace
from src.process.stages import ProcessStage
from src.process.validation import ProcessValidationReport, validate_process_run


class RawDatasetUnavailableError(FileNotFoundError):
    """Raised instead of guessing a public source schema or inventing observations."""


@dataclass(frozen=True)
class BatteryDatasetMetadata:
    dataset_id: str
    display_name: str
    chemistry: str
    evidence_kind: str
    source_doi: str
    version: str
    license: str
    process_stages: tuple[ProcessStage, ...]
    modalities: tuple[str, ...]
    recommended_splits: tuple[str, ...]
    optimization_capable: bool
    multimodal_capable: bool
    limitations: str


@dataclass(frozen=True)
class ProcessPredictionTask:
    target: str
    decision_stage: ProcessStage
    split: str = "GROUP_BY_PROCESS_RECIPE"


@dataclass(frozen=True)
class ProcessOptimizationTask:
    target: str
    decision_stage: ProcessStage


@dataclass
class ProcessTrainingFrame:
    features: pd.DataFrame
    targets: pd.Series
    groups: pd.Series
    run_ids: pd.Series
    horizon: InformationHorizon
    target: str


class BatteryProcessDatasetAdapter(Protocol):
    def metadata(self) -> BatteryDatasetMetadata: ...
    def load_runs(self) -> list[BatteryProcessRun]: ...
    def validate(self) -> ProcessValidationReport: ...
    def build_training_view(self, task: ProcessPredictionTask) -> ProcessTrainingFrame: ...
    def build_optimization_space(self, task: ProcessOptimizationTask) -> ProcessSearchSpace: ...


class NormalizedRunAdapter:
    """Adapter base for audited JSON normalized from immutable raw source files.

    `normalized_runs.json` is a derived cache, never a replacement for raw data. The
    adjacent manifest pins every raw SHA-256 and source-to-schema mapping.
    """

    ADAPTER_VERSION = "1"
    SCHEMA_VERSION = "1"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or Path("data/external") / self.metadata().dataset_id / self.metadata().version)

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed" / self.ADAPTER_VERSION

    @property
    def normalized_runs_path(self) -> Path:
        return self.processed_dir / "normalized_runs.json"

    def load_runs(self) -> list[BatteryProcessRun]:
        if not self.normalized_runs_path.is_file():
            raise RawDatasetUnavailableError(
                f"{self.metadata().dataset_id} has not been source-audited. Expected {self.normalized_runs_path}. "
                "Download immutable raw files, write hashes/manifest and map source fields before training."
            )
        records = json.loads(self.normalized_runs_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("normalized_runs.json must contain a list of BatteryProcessRun records")
        runs = [BatteryProcessRun.from_dict(record) for record in records]
        report = self.validate_runs(runs)
        if not report.valid:
            raise ValueError("Invalid normalized process dataset: " + "; ".join(report.errors))
        return runs

    def validate_runs(self, runs: list[BatteryProcessRun]) -> ProcessValidationReport:
        errors: list[str] = []
        seen: set[str] = set()
        for run in runs:
            if run.run_id in seen:
                errors.append(f"duplicate run_id: {run.run_id}")
            seen.add(run.run_id)
            report = validate_process_run(run)
            errors.extend(f"{run.run_id}: {error}" for error in report.errors)
            if run.provenance.evidence_kind != self.metadata().evidence_kind:
                errors.append(f"{run.run_id}: evidence kind differs from adapter metadata")
        return ProcessValidationReport(valid=not errors, errors=tuple(errors))

    def validate(self) -> ProcessValidationReport:
        try:
            return self.validate_runs(self.load_runs())
        except (OSError, ValueError) as exc:
            return ProcessValidationReport(valid=False, errors=(str(exc),))

    def build_training_view(self, task: ProcessPredictionTask) -> ProcessTrainingFrame:
        rows: list[dict[str, float]] = []
        targets: list[float] = []
        groups: list[str] = []
        run_ids: list[str] = []
        horizon = InformationHorizon(task.decision_stage)
        for run in self.load_runs():
            target = run.final_kpis.get(task.target)
            if target is None or not isinstance(target.value, (int, float)):
                continue
            view = horizon.project(run)
            row = {
                name: float(value.value)
                for name, value in {**view.controls, **view.intermediate_properties}.items()
                if isinstance(value.value, (int, float))
            }
            if not row:
                continue
            rows.append(row)
            targets.append(float(target.value))
            groups.append(run.batch_id or run.run_id)
            run_ids.append(run.run_id)
        if not rows:
            raise ValueError(f"No numeric, horizon-valid training rows for target {task.target!r}")
        raw_features = pd.DataFrame(rows)
        missing = raw_features.isna()
        # A numeric placeholder is only usable by scalar baselines alongside its
        # explicit mask; it never means the source observed a physical zero.
        features = raw_features.fillna(0.0)
        for column in raw_features.columns[missing.any()]:
            features[f"{column}__observed"] = (~missing[column]).astype(float)
        return ProcessTrainingFrame(
            features=features,
            targets=pd.Series(targets, name=task.target),
            groups=pd.Series(groups, name="group"),
            run_ids=pd.Series(run_ids, name="run_id"),
            horizon=horizon,
            target=task.target,
        )

    def build_optimization_space(self, task: ProcessOptimizationTask) -> ProcessSearchSpace:
        rows: list[dict[str, object]] = []
        for run in self.load_runs():
            row: dict[str, object] = {"recipe_id": run.run_id}
            for record in run.stages:
                if record.stage_type == task.decision_stage:
                    row.update({name: parameter.value for name, parameter in record.controls.items()})
            if len(row) > 1:
                rows.append(row)
        if not rows:
            raise ValueError(f"No recorded controls for {task.decision_stage.value}")
        return ProcessSearchSpace.from_finite_pool(pd.DataFrame(rows), id_column="recipe_id")

    def build_replay_frame(self, task: ProcessOptimizationTask) -> pd.DataFrame:
        """Source recipe controls plus hidden final target for the replay oracle only."""
        rows: list[dict[str, object]] = []
        for run in self.load_runs():
            outcome = run.final_kpis.get(task.target)
            stage = next((record for record in run.stages if record.stage_type == task.decision_stage), None)
            if outcome is None or stage is None or not isinstance(outcome.value, (int, float)):
                continue
            rows.append({"recipe_id": run.run_id, **{name: value.value for name, value in stage.controls.items()}, task.target: float(outcome.value)})
        if not rows:
            raise ValueError(f"No source-backed replay rows for {task.target!r} at {task.decision_stage.value}")
        return pd.DataFrame(rows)

    def write_processed_cache(self, runs: list[BatteryProcessRun], *, raw_hashes: dict[str, str]) -> Path:
        if not raw_hashes:
            raise ValueError("raw_hashes are required; processed data cannot be detached from source evidence")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        payload = [run.to_dict() for run in runs]
        self.normalized_runs_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        source_manifest: dict[str, object] = {}
        raw_manifest_path = self.root / "manifest.json"
        if raw_manifest_path.is_file():
            parsed = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                source_manifest = parsed
        processed_hash = hashlib.sha256(self.normalized_runs_path.read_bytes()).hexdigest()
        manifest = {
            "dataset": self.metadata().dataset_id,
            "source_url": source_manifest.get("official_dataset_source"),
            "source_doi": self.metadata().source_doi,
            "version": self.metadata().version,
            "license": source_manifest.get("license"),
            "downloaded_at": source_manifest.get("downloaded_at"),
            "adapter_version": self.ADAPTER_VERSION,
            "adapter_git_sha": self._adapter_git_sha(),
            "schema_version": self.SCHEMA_VERSION,
            "processing_parameters": {"normalization": "typed BatteryProcessRun JSON"},
            "raw_hashes": dict(sorted(raw_hashes.items())),
            "processed_hashes": {"normalized_runs.json": processed_hash},
            "processed_hash": processed_hash,
        }
        (self.processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return self.normalized_runs_path

    @staticmethod
    def _adapter_git_sha() -> str:
        root = Path(__file__).resolve().parents[3]
        try:
            result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        except OSError:
            return "unknown"
        return result.stdout.strip() if result.returncode == 0 else "unknown"
