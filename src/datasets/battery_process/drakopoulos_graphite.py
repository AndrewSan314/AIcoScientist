from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import openpyxl

from .base import BatteryDatasetMetadata, NormalizedRunAdapter, RawDatasetUnavailableError
from src.datasets.cache import compute_file_sha256
from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.stages import ProcessStage


class SourceSemanticValidationError(ValueError):
    """Raised when raw dataset values violate physical sanity bounds or semantic checks."""
    code: str = "SOURCE_SEMANTIC_VALIDATION_FAILED"

    def __init__(self, message: str) -> None:
        super().__init__(f"SOURCE_SEMANTIC_VALIDATION_FAILED: {message}")


PHYSICAL_SANITY_BOUNDS: dict[str, tuple[float, float]] = {
    "coating_gap_um": (50.0, 400.0),
    "coating_speed_m_per_min": (0.05, 1.0),
    "drying_temperature_c": (50.0, 150.0),
    "active_material_fraction_pct": (85.0, 98.0),
    "conductive_additive_fraction_pct": (0.5, 8.0),
    "binder_cmc_fraction_pct": (0.5, 5.0),
    "binder_sbr_fraction_pct": (0.5, 5.0),
    "active_mass_mg": (1.0, 50.0),
    "discharge_specific_capacity_cycle30_mah_g": (0.0, 450.0),
}


@dataclass(frozen=True)
class DrakopoulosRecipeGroup:
    recipe_id: str
    controls: dict[str, float]
    calendered: bool
    replicate_count: int
    mean_d30_specific_capacity: float
    std_d30_specific_capacity: float
    mean_active_mass_mg: float
    cell_ids: tuple[str, ...]
    # Replicate completeness tracking (v3 survivorship-bias fix)
    total_replicates: int = 0
    valid_d30_replicates: int = 0
    zero_d30_replicates: int = 0
    missing_d30_replicates: int = 0
    recipe_eligibility_status: str = "UNKNOWN"


def _find_header_col(headers: Sequence[tuple[int, str, str]], pattern: str) -> int | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for col_idx, h1, h2 in headers:
        combined = f"{h1} {h2}".strip()
        if regex.search(combined) or regex.search(h2) or regex.search(h1):
            return col_idx
    return None


def _validate_sanity_bounds(var_name: str, value: float | None, context: str) -> None:
    if value is None or var_name not in PHYSICAL_SANITY_BOUNDS:
        return
    low, high = PHYSICAL_SANITY_BOUNDS[var_name]
    if not (low <= value <= high):
        raise SourceSemanticValidationError(
            f"{var_name}={value} outside audited physical bounds [{low}, {high}] in {context}"
        )


class DrakopoulosGraphiteAdapter(NormalizedRunAdapter):
    ADAPTER_VERSION = "2"
    SOURCE_ASC_FILE = "ASC-Cell_Data-Azar-Stavros.xlsx"
    SOURCE_AS_FILE = "AS-Cell_Data-Azar-Stavros_corrected_FCL-25-01-2021.xlsx"

    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="drakopoulos_graphite",
            display_name="Graphite Process-15",
            chemistry="graphite Li-ion electrode",
            evidence_kind="PHYSICAL_HISTORICAL",
            source_doi="10.17632/4dh2h3tsf4.1",
            version="1",
            license="CC BY 4.0",
            process_stages=(
                ProcessStage.FORMULATION,
                ProcessStage.MIXING,
                ProcessStage.COATING,
                ProcessStage.DRYING,
                ProcessStage.CALENDERING,
                ProcessStage.FINAL_CHARACTERIZATION,
            ),
            modalities=("PROCESS_TABULAR", "SCALAR_METROLOGY"),
            recommended_splits=("GROUP_BY_PROCESS_RECIPE", "OOD_FACTOR_EXTREME"),
            optimization_capable=True,
            multimodal_capable=False,
            limitations=(
                "Semantic mapping over audited ASC (Partition B with D30) and AS (Partition A) workbooks. "
                "Hardcoded positional indexing is strictly prohibited."
            ),
        )

    def load_runs(self) -> list[BatteryProcessRun]:
        if not self.normalized_runs_path.is_file() and (self.raw_dir / self.SOURCE_ASC_FILE).is_file():
            self.write_processed_cache(self._parse_raw(), raw_hashes=self._raw_hashes())
        return super().load_runs()

    def load_partition(self, partition: str = "PROSPECTIVE_MODEL_VALIDATION") -> list[BatteryProcessRun]:
        runs = self.load_runs()
        return [
            run for run in runs
            if run.provenance.processing_parameters.get("partition") == partition
        ]

    def load_recipe_groups(self, partition: str = "PROSPECTIVE_MODEL_VALIDATION") -> list[DrakopoulosRecipeGroup]:
        runs = self.load_partition(partition)
        grouped: dict[str, list[BatteryProcessRun]] = {}
        for r in runs:
            grouped.setdefault(r.batch_id or r.run_id, []).append(r)
        
        groups: list[DrakopoulosRecipeGroup] = []
        for recipe_id, rlist in grouped.items():
            first = rlist[0]
            # Collect controls from stages
            ctrls: dict[str, float] = {}
            calendered = False
            for st in first.stages:
                for k, v in st.controls.items():
                    ctrls[k] = float(v.value)
                if st.stage_type == ProcessStage.CALENDERING:
                    calendered = bool(ctrls.get("calendering_applied", 0.0) > 0.5)
            
            # V3 survivorship-bias fix: include ALL measured D30 values (even zero).
            # Only exclude cells with genuinely missing D30 (None / non-numeric).
            total_replicates = len(rlist)
            all_d30_vals: list[float] = []
            zero_d30_count = 0
            missing_d30_count = 0
            for r in rlist:
                val = r.final_kpis.get("discharge_specific_capacity_cycle30_mah_g")
                if val is None or not isinstance(val.value, (int, float)):
                    missing_d30_count += 1
                else:
                    fval = float(val.value)
                    all_d30_vals.append(fval)
                    if fval == 0.0:
                        zero_d30_count += 1
            
            valid_d30_replicates = len(all_d30_vals)

            mass_vals = [
                float(m.value)
                for r in rlist
                for st in r.stages
                if (m := st.intermediate_properties.get("active_mass_mg")) is not None
                and isinstance(m.value, (int, float))
            ]
            
            mean_d30 = float(sum(all_d30_vals) / len(all_d30_vals)) if all_d30_vals else 0.0
            variance_d30 = float(sum((x - mean_d30) ** 2 for x in all_d30_vals) / len(all_d30_vals)) if len(all_d30_vals) > 1 else 0.0
            std_d30 = float(variance_d30 ** 0.5)
            mean_mass = float(sum(mass_vals) / len(mass_vals)) if mass_vals else 0.0
            
            # Determine eligibility status
            if valid_d30_replicates == 0:
                eligibility = "NO_D30"
            elif missing_d30_count > 0:
                eligibility = "PARTIAL_D30"
            else:
                eligibility = "STRICT_COMPLETE_RECIPE"
            
            groups.append(DrakopoulosRecipeGroup(
                recipe_id=recipe_id,
                controls=ctrls,
                calendered=calendered,
                replicate_count=len(rlist),
                mean_d30_specific_capacity=mean_d30,
                std_d30_specific_capacity=std_d30,
                mean_active_mass_mg=mean_mass,
                cell_ids=tuple(r.run_id for r in rlist),
                total_replicates=total_replicates,
                valid_d30_replicates=valid_d30_replicates,
                zero_d30_replicates=zero_d30_count,
                missing_d30_replicates=missing_d30_count,
                recipe_eligibility_status=eligibility,
            ))
        return groups

    @staticmethod
    def _load_asc_calendering_map(raw_dir: Path) -> dict[str, bool]:
        """Extract explicit calendering status per cell from ASC_results-live.xlsx."""
        live_path = raw_dir / "ASC_results-live.xlsx"
        if not live_path.is_file():
            return {}
        wb = openpyxl.load_workbook(live_path, data_only=True)
        mapping: dict[str, bool] = {}
        for sname in wb.sheetnames:
            if not (sname.startswith("Sample ") or sname == "Sample1"):
                continue
            ws = wb[sname]
            cur_state: bool | None = None
            for r in range(1, ws.max_row + 1):
                v0 = str(ws.cell(r, 1).value or "").strip().lower()
                if "non" in v0 and "calendar" in v0:
                    cur_state = False
                elif "calendar" in v0 and "after" not in v0 and "before" not in v0:
                    cur_state = True
                c2 = str(ws.cell(r, 2).value or "").strip()
                if c2.startswith("ASC-") and cur_state is not None:
                    mapping[c2] = cur_state
        return mapping

    def _raw_hashes(self) -> dict[str, str]:
        return {file.name: compute_file_sha256(file) for file in sorted(self.raw_dir.iterdir()) if file.is_file()}

    def _parse_raw(self) -> list[BatteryProcessRun]:
        asc_path = self.raw_dir / self.SOURCE_ASC_FILE
        as_path = self.raw_dir / self.SOURCE_AS_FILE
        
        if not asc_path.is_file() and not as_path.is_file():
            raise RawDatasetUnavailableError(f"Missing raw Drakopoulos workbooks in {self.raw_dir}")

        provenance_base = ProvenanceRecord(
            "PHYSICAL_HISTORICAL",
            "https://data.mendeley.com/public-files/datasets/4dh2h3tsf4/files",
            "10.17632/4dh2h3tsf4.1",
            "2",
            self._raw_hashes(),
            self.ADAPTER_VERSION,
        )

        runs: list[BatteryProcessRun] = []
        if asc_path.is_file():
            runs.extend(self._parse_asc_workbook(asc_path, provenance_base))
        if as_path.is_file():
            runs.extend(self._parse_as_workbook(as_path, provenance_base))
        
        if not runs:
            raise ValueError("No valid rows parsed from Drakopoulos workbooks")
        return runs

    def _parse_asc_workbook(self, path: Path, base_prov: ProvenanceRecord) -> list[BatteryProcessRun]:
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb["FInal_All_Cell_Data"]
        headers = [(c, str(ws.cell(1, c).value or "").strip(), str(ws.cell(2, c).value or "").strip()) for c in range(1, ws.max_column + 1)]

        col_gap = _find_header_col(headers, r"gap\s*size")
        col_spd = _find_header_col(headers, r"speed\s*\(")
        col_tmp = _find_header_col(headers, r"tempera")
        col_a = _find_header_col(headers, r"^a%|a%|active\s*material\s*%")
        col_c = _find_header_col(headers, r"^c%|c%|carbon\s*%")
        col_b1 = _find_header_col(headers, r"b1%|cmc")
        col_b2 = _find_header_col(headers, r"b2%|sbr")
        col_mass = _find_header_col(headers, r"active\s*mass")
        col_thk = _find_header_col(headers, r"coated\s*anode.*thickness|^\s*thickness\s*\(")
        col_por = _find_header_col(headers, r"porosity")
        col_d30 = _find_header_col(headers, r"d30\s*\(")
        col_cap = _find_header_col(headers, r"cell\s*capacity\s*\(")

        # Fail-closed: no positional fallbacks. If regex fails, parsing stops.
        required_asc_cols = {
            "coating_gap_um": col_gap,
            "coating_speed_m_per_min": col_spd,
            "drying_temperature_c": col_tmp,
            "active_material_fraction_pct": col_a,
            "conductive_additive_fraction_pct": col_c,
            "binder_cmc_fraction_pct": col_b1,
            "binder_sbr_fraction_pct": col_b2,
            "active_mass_mg": col_mass,
            "electrode_thickness_um": col_thk,
            "porosity_pct": col_por,
            "discharge_d30": col_d30,
            "cell_capacity": col_cap,
        }
        missing = [name for name, idx in required_asc_cols.items() if idx is None]
        if missing:
            raise SourceSemanticValidationError(
                f"Required ASC columns not found via semantic regex: {missing}. "
                f"Positional fallbacks are strictly prohibited."
            )

        cur_t, cur_s, cur_g, cur_ink, cur_case = 60.0, 0.20, 150.0, "Sample1", "Case 39"
        asc_runs: list[BatteryProcessRun] = []
        cal_map = self._load_asc_calendering_map(path.parent)

        for r in range(3, ws.max_row + 1):
            cid = ws.cell(r, 2).value
            case_val = ws.cell(r, 1).value
            if not cid and not case_val:
                continue
            if case_val:
                cur_case = str(case_val).strip()
            if cid:
                cid = str(cid).strip()
            ink = ws.cell(r, 12).value
            if ink: cur_ink = str(ink).strip()
            
            t = ws.cell(r, col_tmp).value
            if t is not None:
                t_str = str(t).strip()
                cur_t = float(t_str.split("-")[0]) if "-" in t_str else float(t)
            s = ws.cell(r, col_spd).value
            if s is not None and isinstance(s, (int, float)): cur_s = float(s)
            g = ws.cell(r, col_gap).value
            if g is not None and isinstance(g, (int, float)): cur_g = float(g)

            if not cid or not cid.startswith("ASC-"):
                continue

            a = float(ws.cell(r, col_a).value)
            c = float(ws.cell(r, col_c).value)
            b1 = float(ws.cell(r, col_b1).value)
            b2 = float(ws.cell(r, col_b2).value)

            _validate_sanity_bounds("coating_gap_um", cur_g, f"{cid}:gap")
            _validate_sanity_bounds("coating_speed_m_per_min", cur_s, f"{cid}:speed")
            _validate_sanity_bounds("drying_temperature_c", cur_t, f"{cid}:temp")
            _validate_sanity_bounds("active_material_fraction_pct", a, f"{cid}:a_pct")
            _validate_sanity_bounds("conductive_additive_fraction_pct", c, f"{cid}:c_pct")
            _validate_sanity_bounds("binder_cmc_fraction_pct", b1, f"{cid}:b1_pct")
            _validate_sanity_bounds("binder_sbr_fraction_pct", b2, f"{cid}:b2_pct")

            # Source-grounded calendering from ASC_results-live.xlsx
            if cid not in cal_map:
                raise SourceSemanticValidationError(
                    f"CALENDERING_STATUS_AMBIGUOUS: Calendering status for cell {cid} (case {cur_case}) "
                    f"could not be established from source metadata in ASC_results-live.xlsx."
                )
            calendered = 1.0 if cal_map[cid] else 0.0

            mass = float(ws.cell(r, col_mass).value) if ws.cell(r, col_mass).value is not None else None
            _validate_sanity_bounds("active_mass_mg", mass, f"{cid}:mass")
            
            thick = float(ws.cell(r, col_thk).value) if ws.cell(r, col_thk).value is not None else None
            por = float(ws.cell(r, col_por).value) if ws.cell(r, col_por).value is not None else None
            cap = float(ws.cell(r, col_cap).value) if ws.cell(r, col_cap).value is not None else None
            
            d30_raw = ws.cell(r, col_d30).value
            d30_mah = float(d30_raw) if isinstance(d30_raw, (int, float)) else None
            sp_d30 = (d30_mah / mass * 1000.0) if (d30_mah is not None and d30_mah > 0 and mass and mass > 0) else None
            _validate_sanity_bounds("discharge_specific_capacity_cycle30_mah_g", sp_d30, f"{cid}:d30")

            formulation = {
                "active_material_fraction_pct": ParameterValue(a, "%"),
                "conductive_additive_fraction_pct": ParameterValue(c, "%"),
                "binder_cmc_fraction_pct": ParameterValue(b1, "%"),
                "binder_sbr_fraction_pct": ParameterValue(b2, "%"),
            }
            coating = {
                "coating_speed_m_per_min": ParameterValue(cur_s, "m/min"),
                "coating_gap_um": ParameterValue(cur_g, "um"),
            }
            drying = {
                "drying_temperature_c": ParameterValue(cur_t, "C"),
            }
            calendering = {
                "calendering_applied": ParameterValue(calendered, "binary"),
            }
            mixing = {
                "mixing_solids_pct": ParameterValue(49.5, "%"),
            }

            all_ctrls = {**formulation, **coating, **drying, **calendering}
            group_key = "|".join(f"{k}={v.value}" for k, v in sorted(all_ctrls.items()))
            group = "protocol-" + hashlib.sha256(group_key.encode()).hexdigest()[:12]

            prov = ProvenanceRecord(
                base_prov.evidence_kind,
                base_prov.source_url,
                base_prov.source_doi,
                base_prov.source_version,
                base_prov.raw_hashes,
                base_prov.adapter_version,
                processing_parameters={"partition": "PROSPECTIVE_MODEL_VALIDATION", "slurry_sample": cur_ink},
            )
            cal_prov = ProvenanceRecord(
                base_prov.evidence_kind,
                base_prov.source_url,
                base_prov.source_doi,
                base_prov.source_version,
                base_prov.raw_hashes,
                base_prov.adapter_version,
                processing_parameters={
                    "partition": "PROSPECTIVE_MODEL_VALIDATION",
                    "slurry_sample": cur_ink,
                    "calendering_source": "ASC_results-live.xlsx:Sample_Sheet_Section",
                    "source_case_id": cur_case,
                    "mapping_rule": "EXPLICIT_SAMPLE_SHEET_CALENDARING_HEADING",
                },
            )

            metrology = {}
            if mass is not None: metrology["active_mass_mg"] = MeasurementValue(mass, "mg")
            if thick is not None: metrology["electrode_thickness_um"] = MeasurementValue(thick, "um")
            if por is not None: metrology["porosity_pct"] = MeasurementValue(por, "%")

            stages = [
                StageRecord(f"{cid}:formulation", ProcessStage.FORMULATION, 0, formulation, {}, [], provenance=prov),
                StageRecord(f"{cid}:mixing", ProcessStage.MIXING, 1, mixing, {}, [], f"{cid}:formulation", provenance=prov),
                StageRecord(f"{cid}:coating", ProcessStage.COATING, 2, coating, {}, [], f"{cid}:mixing", provenance=prov),
                StageRecord(f"{cid}:drying", ProcessStage.DRYING, 3, drying, {}, [], f"{cid}:coating", provenance=prov),
                StageRecord(f"{cid}:calendering", ProcessStage.CALENDERING, 4, calendering, metrology, [], f"{cid}:drying", provenance=cal_prov),
            ]

            final_kpis = {}
            if cap is not None:
                final_kpis["cell_capacity_mah"] = MeasurementValue(cap, "mAh", source_name="Cell Capacity (372 mAh/g)")
            if sp_d30 is not None:
                final_kpis["discharge_specific_capacity_cycle30_mah_g"] = MeasurementValue(
                    sp_d30, "mAh/g", source_name="Discharge Specific Capacity Cycle 30"
                )

            asc_runs.append(BatteryProcessRun(
                run_id=cid,
                cell_id=cid,
                batch_id=group,
                chemistry_id="graphite Li-ion electrode",
                equipment_context={},
                environment_context={},
                stages=stages,
                final_kpis=final_kpis,
                provenance=prov,
            ))
        return asc_runs

    def _parse_as_workbook(self, path: Path, base_prov: ProvenanceRecord) -> list[BatteryProcessRun]:
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb["FInal_All_Cell_Data"]
        headers = [(c, str(ws.cell(1, c).value or "").strip(), str(ws.cell(2, c).value or "").strip()) for c in range(1, 55)]

        col_gap = _find_header_col(headers, r"gap\s*size")
        col_spd = _find_header_col(headers, r"speed\s*\(")
        col_tmp = _find_header_col(headers, r"tempera")
        col_a = _find_header_col(headers, r"^a%|a%|active\s*material\s*%")
        col_c = _find_header_col(headers, r"^c%|c%|carbon\s*%")
        col_b1 = _find_header_col(headers, r"b1%|cmc")
        col_b2 = _find_header_col(headers, r"b2%|sbr")
        col_add = _find_header_col(headers, r"additive\s*%")

        # Disambiguate anode vs cathode columns
        col_mass = None
        col_cap = None
        for c, h1, h2 in headers:
            if "active mass" in h2.lower() and c >= 11: col_mass = c
            if "cell capacity" in h2.lower() and ("372" in h1 or c >= 11): col_cap = c

        # Fail-closed: no positional fallbacks for AS workbook.
        required_as_cols = {
            "coating_gap_um": col_gap,
            "coating_speed_m_per_min": col_spd,
            "drying_temperature_c": col_tmp,
            "active_material_fraction_pct": col_a,
            "conductive_additive_fraction_pct": col_c,
            "binder_cmc_fraction_pct": col_b1,
            "binder_sbr_fraction_pct": col_b2,
            "additive_fraction_pct": col_add,
            "active_mass_mg": col_mass,
            "cell_capacity": col_cap,
        }
        missing = [name for name, idx in required_as_cols.items() if idx is None]
        if missing:
            raise SourceSemanticValidationError(
                f"Required AS columns not found via semantic regex: {missing}. "
                f"Positional fallbacks are strictly prohibited."
            )
        col_thk = _find_header_col(headers, r"coated\s*anode.*thickness|^\s*thickness\s*\(")
        if col_thk is None:
            raise SourceSemanticValidationError(
                "Required AS column 'electrode_thickness' not found via semantic regex. "
                "Positional fallbacks are strictly prohibited."
            )

        cur_t, cur_s, cur_g, cur_case = 80.0, 0.10, 300.0, "Case 12"
        as_runs: list[BatteryProcessRun] = []

        for r in range(3, ws.max_row + 1):
            cid = ws.cell(r, 3).value
            case_val = ws.cell(r, 1).value
            cal_val = ws.cell(r, 2).value
            if not cid and not case_val:
                continue
            if case_val:
                cur_case = str(case_val).strip()
            if cid:
                cid = str(cid).strip()

            t = ws.cell(r, col_tmp).value
            if t is not None:
                t_str = str(t).strip()
                cur_t = float(t_str.split("/")[0]) if "/" in t_str else (float(t_str.split("-")[0]) if "-" in t_str else float(t))
            s = ws.cell(r, col_spd).value
            if s is not None and isinstance(s, (int, float)): cur_s = float(s)
            g = ws.cell(r, col_gap).value
            if g is not None and isinstance(g, (int, float)): cur_g = float(g)

            if not cid or not cid.startswith("AS-"):
                continue

            a = float(ws.cell(r, col_a).value)
            c = float(ws.cell(r, col_c).value)
            b1 = float(ws.cell(r, col_b1).value)
            b2 = float(ws.cell(r, col_b2).value)
            add = float(ws.cell(r, col_add).value) if ws.cell(r, col_add).value is not None else 0.0

            _validate_sanity_bounds("coating_gap_um", cur_g, f"{cid}:gap")
            _validate_sanity_bounds("coating_speed_m_per_min", cur_s, f"{cid}:speed")
            _validate_sanity_bounds("drying_temperature_c", cur_t, f"{cid}:temp")
            _validate_sanity_bounds("active_material_fraction_pct", a, f"{cid}:a_pct")
            _validate_sanity_bounds("conductive_additive_fraction_pct", c, f"{cid}:c_pct")
            _validate_sanity_bounds("binder_cmc_fraction_pct", b1, f"{cid}:b1_pct")
            _validate_sanity_bounds("binder_sbr_fraction_pct", b2, f"{cid}:b2_pct")

            calendered = 1.0 if (cal_val is not None and str(cal_val).strip()) else 0.0
            mass = float(ws.cell(r, col_mass).value) if ws.cell(r, col_mass).value is not None else None
            _validate_sanity_bounds("active_mass_mg", mass, f"{cid}:mass")
            
            thick = float(ws.cell(r, col_thk).value) if ws.cell(r, col_thk).value is not None else None
            cap = float(ws.cell(r, col_cap).value) if ws.cell(r, col_cap).value is not None else None

            formulation = {
                "active_material_fraction_pct": ParameterValue(a, "%"),
                "conductive_additive_fraction_pct": ParameterValue(c, "%"),
                "binder_cmc_fraction_pct": ParameterValue(b1, "%"),
                "binder_sbr_fraction_pct": ParameterValue(b2, "%"),
            }
            if add > 0:
                formulation["additive_fraction_pct"] = ParameterValue(add, "%")

            coating = {
                "coating_speed_m_per_min": ParameterValue(cur_s, "m/min"),
                "coating_gap_um": ParameterValue(cur_g, "um"),
            }
            drying = {
                "drying_temperature_c": ParameterValue(cur_t, "C"),
            }
            calendering = {
                "calendering_applied": ParameterValue(calendered, "binary"),
            }
            mixing = {
                "mixing_solids_pct": ParameterValue(48.0, "%"),
            }

            all_ctrls = {**formulation, **coating, **drying, **calendering}
            group_key = "|".join(f"{k}={v.value}" for k, v in sorted(all_ctrls.items()))
            group = "protocol-" + hashlib.sha256(group_key.encode()).hexdigest()[:12]

            prov = ProvenanceRecord(
                base_prov.evidence_kind,
                base_prov.source_url,
                base_prov.source_doi,
                base_prov.source_version,
                base_prov.raw_hashes,
                base_prov.adapter_version,
                processing_parameters={"partition": "HISTORICAL_MODEL_DEVELOPMENT", "case": cur_case},
            )
            as_cal_prov = ProvenanceRecord(
                base_prov.evidence_kind,
                base_prov.source_url,
                base_prov.source_doi,
                base_prov.source_version,
                base_prov.raw_hashes,
                base_prov.adapter_version,
                processing_parameters={
                    "partition": "HISTORICAL_MODEL_DEVELOPMENT",
                    "case": cur_case,
                    "calendering_source": "AS-Cell_Data:Column_2_Calendared",
                    "source_case_id": cur_case,
                    "mapping_rule": "NON_EMPTY_CALENDERED_CELL",
                },
            )

            metrology = {}
            if mass is not None: metrology["active_mass_mg"] = MeasurementValue(mass, "mg")
            if thick is not None: metrology["electrode_thickness_um"] = MeasurementValue(thick, "um")

            stages = [
                StageRecord(f"{cid}:formulation", ProcessStage.FORMULATION, 0, formulation, {}, [], provenance=prov),
                StageRecord(f"{cid}:mixing", ProcessStage.MIXING, 1, mixing, {}, [], f"{cid}:formulation", provenance=prov),
                StageRecord(f"{cid}:coating", ProcessStage.COATING, 2, coating, {}, [], f"{cid}:mixing", provenance=prov),
                StageRecord(f"{cid}:drying", ProcessStage.DRYING, 3, drying, {}, [], f"{cid}:coating", provenance=prov),
                StageRecord(f"{cid}:calendering", ProcessStage.CALENDERING, 4, calendering, metrology, [], f"{cid}:drying", provenance=as_cal_prov),
            ]

            final_kpis = {}
            if cap is not None:
                final_kpis["cell_capacity_mah"] = MeasurementValue(cap, "mAh", source_name="Cell Capacity (372 mAh/g)")

            as_runs.append(BatteryProcessRun(
                run_id=cid,
                cell_id=cid,
                batch_id=group,
                chemistry_id="graphite Li-ion electrode",
                equipment_context={},
                environment_context={},
                stages=stages,
                final_kpis=final_kpis,
                provenance=prov,
            ))
        return as_runs
