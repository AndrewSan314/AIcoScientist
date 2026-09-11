from __future__ import annotations

import hashlib

import pandas as pd

from .base import BatteryDatasetMetadata, NormalizedRunAdapter, RawDatasetUnavailableError
from src.datasets.cache import compute_file_sha256
from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.stages import ProcessStage


class WarwickNMC622Adapter(NormalizedRunAdapter):
    TABLE_FILE = "Intermediate measurements during calendering.xlsx"

    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="warwick_nmc622", display_name="Warwick NMC622 Pilot Multimodal", chemistry="NMC622 Li-ion electrode/cell",
            evidence_kind="PILOT_LINE_HISTORICAL", source_doi="10.17632/wwhm2frfmy.1", version="1", license="CC0 1.0",
            process_stages=(ProcessStage.CALENDERING, ProcessStage.FINAL_CHARACTERIZATION),
            modalities=("PROCESS_TABULAR", "SCALAR_METROLOGY"), recommended_splits=("GROUP_BY_DOE_CONDITION",),
            optimization_capable=True, multimodal_capable=True, limitations="Parser covers the audited Cathode intermediate-calendering table; electrochemical files remain independently auditable modalities.")

    def _table_path(self):
        return next(iter(self.raw_dir.rglob(self.TABLE_FILE)), None)

    def load_runs(self) -> list[BatteryProcessRun]:
        if not self.normalized_runs_path.is_file() and self._table_path() is not None:
            self.write_processed_cache(self._parse_raw(), raw_hashes=self._raw_hashes())
        return super().load_runs()

    def _raw_hashes(self) -> dict[str, str]:
        archive = next(iter(self.raw_dir.glob("*.zip")), None)
        return {archive.name: compute_file_sha256(archive)} if archive else {}

    def _parse_raw(self) -> list[BatteryProcessRun]:
        source = self._table_path()
        if source is None:
            raise RawDatasetUnavailableError("Missing extracted Warwick NMC622 intermediate-calendering table")
        raw = pd.read_excel(source, sheet_name="Cathode", header=None)
        provenance = ProvenanceRecord("PILOT_LINE_HISTORICAL", "https://data.mendeley.com/public-files/datasets/wwhm2frfmy/files/b88e31b9-068f-40a4-bb22-3e81b5f7b198/file_downloaded", "10.17632/wwhm2frfmy.1", "1", self._raw_hashes(), self.ADAPTER_VERSION)
        runs: list[BatteryProcessRun] = []
        for _, row in raw.iloc[2:].iterrows():
            run_id = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
            density, porosity = (pd.to_numeric(row.iloc[index], errors="coerce") for index in (26, 27))
            control_positions = (("target_coating_weight_gsm", 2, "g/m2"), ("roll_temperature_c", 3, "C"), ("target_density_g_cm3", 4, "g/cm3"), ("target_porosity_pct", 5, "%"), ("roll_gap_um", 21, "um"), ("number_of_passes", 22, None))
            controls = {name: ParameterValue(float(value), units) for name, index, units in control_positions if pd.notna(value := pd.to_numeric(row.iloc[index], errors="coerce"))}
            if not run_id or not pd.notna(density) or not pd.notna(porosity) or len(controls) != len(control_positions):
                continue
            group = "doe-" + hashlib.sha256("|".join(f"{key}={value.value}" for key, value in sorted(controls.items())).encode()).hexdigest()[:12]
            pre = {name: MeasurementValue(float(value), units) for name, index, units in (("pre_calendering_thickness_um", 7, "um"), ("pre_calendering_density_g_cm3", 10, "g/cm3"), ("pre_calendering_porosity_pct", 11, "%")) if pd.notna(value := pd.to_numeric(row.iloc[index], errors="coerce"))}
            post = {name: MeasurementValue(float(value), units) for name, value, units in (("calendered_thickness_um", pd.to_numeric(row.iloc[23], errors="coerce"), "um"), ("calendered_density_g_cm3", density, "g/cm3"), ("calendered_porosity_pct", porosity, "%"), ("calendered_tensile_strength_kpa", pd.to_numeric(row.iloc[28], errors="coerce"), "kPa")) if pd.notna(value)}
            stages = [StageRecord(f"{run_id}:coating", ProcessStage.COATING, 2, {}, pre, [], provenance=provenance), StageRecord(f"{run_id}:calendering", ProcessStage.CALENDERING, 4, controls, post, [], f"{run_id}:coating", provenance=provenance)]
            runs.append(BatteryProcessRun(run_id, None, group, "NMC622 Li-ion cathode", {}, {}, stages, {"calendered_density_g_cm3": MeasurementValue(float(density), "g/cm3"), "calendered_porosity_pct": MeasurementValue(float(porosity), "%")}, provenance))
        if not runs:
            raise ValueError("No Cathode source rows matched the audited NMC622 schema")
        return runs
