from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .base import BatteryDatasetMetadata, NormalizedRunAdapter, RawDatasetUnavailableError
from src.datasets.cache import compute_file_sha256
from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.modalities import ModalityObservation, ModalityType
from src.process.stages import ProcessStage


class WarwickUltrasoundAdapter(NormalizedRunAdapter):
    """V4 FFT signal adapter; before/after signals remain distinct source modalities."""

    ADAPTER_VERSION = "2"

    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="warwick_ultrasound", display_name="Warwick Ultrasonic Inline QC", chemistry="graphite anode / NMC622 cathode",
            evidence_kind="PHYSICAL_HISTORICAL", source_doi="10.17632/c62yn37d9h.4", version="4", license="CC BY 4.0",
            process_stages=(ProcessStage.CALENDERING, ProcessStage.FINAL_CHARACTERIZATION), modalities=("ULTRASOUND_SPECTRUM", "PROCESS_TABULAR"),
            recommended_splits=("LEAVE_ONE_PROCESS_SETTING_OUT",), optimization_capable=False, multimodal_capable=True,
            limitations="Post-calender spectra are represented but are unavailable at a CALENDERING decision horizon.")

    def load_runs(self) -> list[BatteryProcessRun]:
        if not self.normalized_runs_path.is_file() and self._source_root() is not None:
            self.write_processed_cache(self._parse_raw(), raw_hashes=self._raw_hashes())
        return super().load_runs()

    def _source_root(self) -> Path | None:
        return next(iter(self.raw_dir.glob("unpacked/*")), None)

    def _raw_hashes(self) -> dict[str, str]:
        archive = next(iter(self.raw_dir.glob("*.zip")), None)
        return {archive.name: compute_file_sha256(archive)} if archive else {}

    @staticmethod
    def _name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    def _parse_raw(self) -> list[BatteryProcessRun]:
        root = self._source_root()
        if root is None:
            raise RawDatasetUnavailableError("Missing extracted Warwick ultrasound V4 archive")
        provenance = ProvenanceRecord("PHYSICAL_HISTORICAL", "https://data.mendeley.com/public-api/zip/c62yn37d9h/download/4", "10.17632/c62yn37d9h.4", "4", self._raw_hashes(), self.ADAPTER_VERSION)
        pairs: dict[tuple[str, str], dict[str, tuple[Path, dict]]] = {}
        for path in root.rglob("*-calendering.json"):
            if path.parent.name == "ToF":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            state = str(payload.get("metadata", {}).get("Calendering_State", ""))
            sample = str(payload.get("metadata", {}).get("Sample_ID", path.parent.name))
            material = path.parents[1].name
            pairs.setdefault((material, sample), {})[state] = (path, payload)
        runs: list[BatteryProcessRun] = []
        for (material, sample), states in sorted(pairs.items()):
            if "before-calendering" not in states or "after-calendering" not in states:
                continue
            before_path, before = states["before-calendering"]
            after_path, after = states["after-calendering"]
            before_meta, after_meta = before["metadata"], after["metadata"]
            thickness = after_meta.get("Thickness")
            if not isinstance(thickness, (int, float)):
                continue
            controls = {
                self._name(name): ParameterValue(float(value))
                for name, value in before_meta.items()
                if name not in {"Sample_ID", "Thickness", "Density", "Calendering_State"} and isinstance(value, (int, float))
            }
            group = "ultrasound-" + hashlib.sha256("|".join(f"{name}={value.value}" for name, value in sorted(controls.items())).encode()).hexdigest()[:12]
            spectra = [
                ModalityObservation(f"{sample}:before_spectrum", ModalityType.ULTRASOUND_SPECTRUM, ProcessStage.CALENDERING, str(before_path.relative_to(root)), {"fft_frequency": before["fft_frequency"], "fft_magnitude": before["fft_magnitude"]}, "MHz/normalized_a.u.", (len(before["fft_magnitude"]),), provenance=provenance),
                ModalityObservation(f"{sample}:after_spectrum", ModalityType.ULTRASOUND_SPECTRUM, ProcessStage.CALENDERING, str(after_path.relative_to(root)), {"fft_frequency": after["fft_frequency"], "fft_magnitude": after["fft_magnitude"]}, "MHz/normalized_a.u.", (len(after["fft_magnitude"]),), provenance=provenance),
            ]
            properties = {"pre_calendering_thickness_um": MeasurementValue(float(before_meta["Thickness"]), "um")} if isinstance(before_meta.get("Thickness"), (int, float)) else {}
            stages = [
                StageRecord(f"{material}:{sample}:calendering", ProcessStage.CALENDERING, 4, controls, properties, spectra, provenance=provenance),
                StageRecord(f"{material}:{sample}:final", ProcessStage.FINAL_CHARACTERIZATION, 8, {}, {}, [], f"{material}:{sample}:calendering", provenance=provenance),
            ]
            final_kpis = {"post_calendering_thickness_um": MeasurementValue(float(thickness), "um")}
            if isinstance(after_meta.get("Density"), (int, float)):
                final_kpis["post_calendering_density_g_cm3"] = MeasurementValue(float(after_meta["Density"]), "g/cm3")
            runs.append(BatteryProcessRun(f"{material}:{sample}", None, group, material.lower(), {}, {}, stages, final_kpis, provenance))
        if not runs:
            raise ValueError("No paired before/after source ultrasound records found")
        return runs
