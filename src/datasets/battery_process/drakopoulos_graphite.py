from __future__ import annotations

import hashlib

import pandas as pd

from .base import BatteryDatasetMetadata, NormalizedRunAdapter, RawDatasetUnavailableError
from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.stages import ProcessStage


class DrakopoulosGraphiteAdapter(NormalizedRunAdapter):
    SOURCE_FILE = "AS-Cell_Data-Azar-Stavros_corrected_FCL-25-01-2021.xlsx"

    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="drakopoulos_graphite", display_name="Graphite Process-15", chemistry="graphite Li-ion electrode",
            evidence_kind="PHYSICAL_HISTORICAL", source_doi="10.17632/4dh2h3tsf4.1", version="1", license="CC BY 4.0",
            process_stages=(ProcessStage.FORMULATION, ProcessStage.MIXING, ProcessStage.COATING, ProcessStage.DRYING, ProcessStage.CALENDERING, ProcessStage.FINAL_CHARACTERIZATION),
            modalities=("PROCESS_TABULAR", "SCALAR_METROLOGY"), recommended_splits=("GROUP_BY_PROCESS_RECIPE", "OOD_FACTOR_EXTREME"),
            optimization_capable=True, multimodal_capable=False, limitations="Parser covers AS cells with numeric 372 mAh/g cell-capacity records; other source workbooks remain inventory-only.")

    def load_runs(self) -> list[BatteryProcessRun]:
        if not self.normalized_runs_path.is_file() and (self.raw_dir / self.SOURCE_FILE).is_file():
            self.write_processed_cache(self._parse_raw(), raw_hashes=self._raw_hashes())
        return super().load_runs()

    def _raw_hashes(self) -> dict[str, str]:
        return {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(self.raw_dir.iterdir()) if file.is_file()}

    def _parse_raw(self) -> list[BatteryProcessRun]:
        source = self.raw_dir / self.SOURCE_FILE
        if not source.is_file():
            raise RawDatasetUnavailableError(f"Missing {source}")
        raw = pd.read_excel(source, header=None, usecols=range(25))
        provenance = ProvenanceRecord("PHYSICAL_HISTORICAL", "https://data.mendeley.com/public-files/datasets/4dh2h3tsf4/files/01237dc5-0ca1-45ad-8367-16c464a1a5fd/file_downloaded", "10.17632/4dh2h3tsf4.1", "1", self._raw_hashes(), self.ADAPTER_VERSION)
        runs: list[BatteryProcessRun] = []
        for _, row in raw.iloc[2:].iterrows():
            run_id = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
            capacity = pd.to_numeric(row.iloc[15], errors="coerce")
            speed, gap, temp = (pd.to_numeric(row.iloc[index], errors="coerce") for index in (18, 19, 17))
            if not run_id.startswith("AS-") or not all(pd.notna(value) for value in (capacity, speed, gap)):
                continue
            controls = {"coating_speed_m_per_min": ParameterValue(float(speed), "m/min"), "coating_gap_um": ParameterValue(float(gap), "um")}
            if pd.notna(temp):
                controls["drying_temperature_c"] = ParameterValue(float(temp), "C")
            fractions = (("active_material_fraction_pct", 20), ("conductive_additive_fraction_pct", 21), ("binder_cmc_fraction_pct", 22), ("binder_sbr_fraction_pct", 23), ("additive_fraction_pct", 24))
            formulation = {name: ParameterValue(float(value), "%") for name, index in fractions if pd.notna(value := pd.to_numeric(row.iloc[index], errors="coerce"))}
            group = "protocol-" + hashlib.sha256("|".join(f"{key}={value.value}" for key, value in sorted({**controls, **formulation}.items())).encode()).hexdigest()[:12]
            stages = [
                StageRecord(f"{run_id}:formulation", ProcessStage.FORMULATION, 0, formulation, {}, [], provenance=provenance),
                StageRecord(f"{run_id}:coating", ProcessStage.COATING, 2, {key: value for key, value in controls.items() if key.startswith("coating_")}, {}, [], f"{run_id}:formulation", provenance=provenance),
                StageRecord(f"{run_id}:drying", ProcessStage.DRYING, 3, {key: value for key, value in controls.items() if key.startswith("drying_")}, {}, [], f"{run_id}:coating", provenance=provenance),
            ]
            runs.append(BatteryProcessRun(run_id, run_id, group, "graphite Li-ion electrode", {}, {}, stages, {"cell_capacity_mah": MeasurementValue(float(capacity), "mAh", source_name="Cell Capacity (372 mAh/g)")}, provenance))
        if not runs:
            raise ValueError("No AS source rows matched the audited graphite schema")
        return runs
