from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from collections import deque
from pathlib import Path

from .base import BatteryDatasetMetadata, NormalizedRunAdapter, RawDatasetUnavailableError
from src.process.contracts import BatteryProcessRun, MeasurementValue, ProvenanceRecord, StageRecord
from src.process.modalities import ModalityObservation, ModalityType
from src.process.stages import ProcessStage


class NaIonHTEAdapter(NormalizedRunAdapter):
    """Stream source CSV traces; HDF5 copies remain immutable raw evidence."""

    ADAPTER_VERSION = "2"
    ARCHIVE = "Na-upscaling.zip"
    TRACE_FIELDS = {
        "cycle_index": "Cycle_Index", "test_time_s": "Test_Time(s)", "current_a": "Current(A)",
        "voltage_v": "Voltage(V)", "charge_capacity_ah": "Charge_Capacity(Ah)",
        "discharge_capacity_ah": "Discharge_Capacity(Ah)",
    }

    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="naion_hte", display_name="Na-ion High-Throughput Process Chain", chemistry="sodium-ion",
            evidence_kind="PHYSICAL_HISTORICAL", source_doi="10.5281/zenodo.7981011", version="7981011", license="not specified",
            process_stages=(ProcessStage.FORMATION, ProcessStage.FINAL_CHARACTERIZATION), modalities=("FORMATION_CURVE", "CYCLING_CURVE"),
            recommended_splits=("GROUP_BY_BATCH", "LATER_RUN_HOLDOUT"), optimization_capable=False, multimodal_capable=True,
            limitations="Source archive provides cycling traces and protocol/batch identity, not audited synthesis, coating, or assembly settings.")

    @property
    def archive_path(self) -> Path:
        return self.raw_dir / self.ARCHIVE

    def load_runs(self) -> list[BatteryProcessRun]:
        if not self.normalized_runs_path.is_file() and self.archive_path.is_file():
            self.write_processed_cache(self._parse_raw(), raw_hashes=self._raw_hashes())
        return super().load_runs()

    def _raw_hashes(self) -> dict[str, str]:
        if not self.archive_path.is_file():
            raise RawDatasetUnavailableError(f"Missing Na-ion source archive: {self.archive_path}")
        return {self.ARCHIVE: hashlib.sha256(self.archive_path.read_bytes()).hexdigest()}

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

    @staticmethod
    def _number(value: str | None) -> float | None:
        try:
            return float(value) if value not in (None, "") else None
        except ValueError:
            return None

    def _trace(self, archive: zipfile.ZipFile, entry: zipfile.ZipInfo) -> tuple[dict[str, list[float | None]], dict[str, list[float | None]], float | None]:
        head: dict[str, list[float | None]] = {name: [] for name in self.TRACE_FIELDS}
        tail: deque[dict[str, float | None]] = deque(maxlen=256)
        last_discharge: float | None = None
        with archive.open(entry) as source, io.TextIOWrapper(source, encoding="utf-8-sig", newline="") as text:
            for row in csv.DictReader(text):
                point = {name: self._number(row.get(column)) for name, column in self.TRACE_FIELDS.items()}
                if point["voltage_v"] is None:
                    continue
                if len(head["voltage_v"]) < 256:
                    for name, value in point.items():
                        head[name].append(value)
                tail.append(point)
                if point["discharge_capacity_ah"] is not None:
                    last_discharge = point["discharge_capacity_ah"]
        final = {name: [point[name] for point in tail] for name in self.TRACE_FIELDS}
        return head, final, last_discharge

    def _parse_raw(self) -> list[BatteryProcessRun]:
        if not self.archive_path.is_file():
            raise RawDatasetUnavailableError(f"Missing Na-ion source archive: {self.archive_path}")
        provenance = ProvenanceRecord("PHYSICAL_HISTORICAL", "https://zenodo.org/api/records/7981011/files/Na-upscaling.zip/content", "10.5281/zenodo.7981011", "7981011", self._raw_hashes(), self.ADAPTER_VERSION)
        runs: list[BatteryProcessRun] = []
        with zipfile.ZipFile(self.archive_path) as archive:
            entries = sorted((entry for entry in archive.infolist() if entry.filename.lower().endswith(".csv")), key=lambda entry: entry.filename)
            for entry in entries:
                formation, cycling, capacity = self._trace(archive, entry)
                if not formation["voltage_v"] or not cycling["voltage_v"]:
                    continue
                path = Path(entry.filename)
                protocol = path.parts[1] if len(path.parts) > 1 else "unknown"
                match = re.search(r"cel+l?(\d+)", entry.filename, re.IGNORECASE)
                cell_id = match.group(1) if match else None
                run_id = f"naion-{self._slug(path.stem)}-{hashlib.sha256(entry.filename.encode()).hexdigest()[:8]}"
                stages = [
                    StageRecord(f"{run_id}:formation", ProcessStage.FORMATION, 7, {}, {}, [ModalityObservation(f"{run_id}:formation_curve", ModalityType.FORMATION_CURVE, ProcessStage.FORMATION, entry.filename, formation, "mixed", (len(formation["voltage_v"]),), provenance=provenance)], provenance=provenance),
                    StageRecord(f"{run_id}:final", ProcessStage.FINAL_CHARACTERIZATION, 8, {}, {}, [ModalityObservation(f"{run_id}:cycling_curve", ModalityType.CYCLING_CURVE, ProcessStage.FINAL_CHARACTERIZATION, entry.filename, cycling, "mixed", (len(cycling["voltage_v"]),), provenance=provenance)], f"{run_id}:formation", provenance=provenance),
                ]
                kpis = {"last_observed_discharge_capacity_ah": MeasurementValue(capacity, "Ah")} if capacity is not None else {}
                runs.append(BatteryProcessRun(run_id, cell_id, self._slug(protocol), "sodium-ion", {"source_protocol": protocol}, {}, stages, kpis, provenance))
        if not runs:
            raise ValueError("No readable Na-ion CSV traces found in source archive")
        return runs
