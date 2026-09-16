from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .base import (
    BatteryDatasetMetadata,
    NormalizedRunAdapter,
    ProcessOptimizationTask,
    ProcessPredictionTask,
    ProcessTrainingFrame,
    RawDatasetUnavailableError,
)
from src.datasets.cache import compute_file_sha256
from src.process.contracts import (
    BatteryProcessRun,
    MeasurementValue,
    ParameterValue,
    ProvenanceRecord,
    StageRecord,
)
from src.process.information_horizon import InformationHorizon
from src.process.optimization.process_space import ProcessSearchSpace
from src.process.stages import ProcessStage


@dataclass(frozen=True)
class WarwickDOEGroup:
    condition_id: str
    condition_no: int
    electrode_id: str
    loading_regime: str
    target_coating_weight_gsm: float
    roll_temperature_c: float
    target_density_g_cm3: float
    target_porosity_pct: float
    roll_gap_um: float
    number_of_passes: int
    cell_ids: tuple[str, ...]
    mean_rate_performance_5c_0_2c: float
    std_rate_performance_5c_0_2c: float
    replicate_rate_values: tuple[float, ...]


class WarwickNMC622CalenderingAdapter(NormalizedRunAdapter):
    """Adapter for Warwick NMC622 Pilot-Plant Calendering dataset.

    Source DOI: 10.17632/wwhm2frfmy.1
    Pilot-plant scale battery manufacturing of NMC622 cathodes across
    18 full factorial calendering conditions and 54 half-cells (3 replicates per condition).
    """

    ADAPTER_VERSION = "1"
    SCHEMA_VERSION = "1"
    ACCEPTED_EVIDENCE_KINDS = ("PILOT_LINE_HISTORICAL", "TEST_FIXTURE")

    def __init__(self, root: str | Path | None = None) -> None:
        if root is not None:
            self.root = Path(root)
        else:
            # Check default directories in order of preference
            candidate_roots = [
                Path("data/external/warwick_nmc622_calendering/raw"),
                Path("data/external/warwick_nmc622_calendering"),
                Path("data/external/Data of Physical and Electrochemical Characteristics of Calendered NMC622 Electrodes and Lithium-ion Cells at Pilot-Plant Scale Battery Manufacturing/Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale"),
                Path("data/external/warwick_nmc622/1"),
            ]
            found = None
            for cand in candidate_roots:
                if (cand / "1- Tables").exists() or (cand / "Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale" / "1- Tables").exists():
                    found = cand
                    break
            self.root = found or Path("data/external/warwick_nmc622_calendering/raw")

    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="warwick_nmc622_calendering",
            display_name="Warwick NMC622 Pilot Calendering",
            chemistry="NMC622 cathode / lithium-metal half-cell",
            evidence_kind="PILOT_LINE_HISTORICAL",
            source_doi="10.17632/wwhm2frfmy.1",
            version="1",
            license="CC0 1.0",
            process_stages=(ProcessStage.COATING, ProcessStage.CALENDERING, ProcessStage.FINAL_CHARACTERIZATION),
            modalities=("PROCESS_TABULAR", "SCALAR_METROLOGY"),
            recommended_splits=("GROUP_BY_DOE_CONDITION",),
            optimization_capable=True,
            multimodal_capable=True,
            limitations="Pilot-plant scale calendering study with 18 full factorial conditions (temperature, density/porosity, mass loading) across 54 half-cells (3 replicates per condition).",
        )

    def _find_tables_dir(self) -> Path:
        candidates = [
            self.root / "1- Tables",
            self.root / "Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale" / "1- Tables",
            Path("data/external/warwick_nmc622_calendering/raw/Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale/1- Tables"),
            Path("data/external/Data of Physical and Electrochemical Characteristics of Calendered NMC622 Electrodes and Lithium-ion Cells at Pilot-Plant Scale Battery Manufacturing/Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale/1- Tables"),
        ]
        for c in candidates:
            if c.exists() and (c / "Half-cell (Cathode) Electrochemical Performance.xlsx").exists():
                return c
        raise RawDatasetUnavailableError(
            f"WARWICK_NMC622_OFFICIAL_DATA_UNAVAILABLE: Could not locate extracted 1- Tables directory for Warwick NMC622 dataset. "
            "Required DOI: 10.17632/wwhm2frfmy.1"
        )

    def _raw_hashes(self) -> dict[str, str]:
        archive_candidates = [
            self.root / "Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale.zip",
            self.root.parent / "Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale.zip",
            Path("data/external/warwick_nmc622_calendering/raw/Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale.zip"),
            Path("data/external/Data of Physical and Electrochemical Characteristics of Calendered NMC622 Electrodes and Lithium-ion Cells at Pilot-Plant Scale Battery Manufacturing/Characteristics of Electrodes and Lithium-ion Cells at Pilot-Plant Manufacturing Scale.zip"),
        ]
        for a in archive_candidates:
            if a.exists():
                return {a.name: compute_file_sha256(a)}
        return {}

    def load_runs(self) -> list[BatteryProcessRun]:
        tables_dir = self._find_tables_dir()
        table_p1 = tables_dir / "Half-cell (Cathode) Electrochemical Performance.xlsx"
        table_p2 = tables_dir / "Intermediate measurements during calendering.xlsx"
        table_p3 = tables_dir / "Half-cell Electrochemical Performance Cycling Performance.xlsx"

        raw_hashes = self._raw_hashes()
        base_prov = ProvenanceRecord(
            evidence_kind="PILOT_LINE_HISTORICAL",
            source_url="https://data.mendeley.com/datasets/wwhm2frfmy/1",
            source_doi="10.17632/wwhm2frfmy.1",
            source_version="1",
            raw_hashes=raw_hashes,
            adapter_version=self.ADAPTER_VERSION,
        )

        xl_cal = pd.ExcelFile(table_p2)
        df_cal = xl_cal.parse("Cathode", header=None)

        xl_el = pd.ExcelFile(table_p1)
        df_tab = xl_el.parse("Table", header=None).iloc[2:20]

        xl_cyc = pd.ExcelFile(table_p3)
        df_cyc = xl_cyc.parse("Sheet1", header=None)

        # Cycling cells map
        cyc_cell_map: dict[str, dict[str, float]] = {}
        for col_idx in range(1, df_cyc.shape[1]):
            c_id = str(df_cyc.iloc[0, col_idx]).strip()
            if not c_id.startswith("DD"):
                continue
            try:
                c0 = float(df_cyc.iloc[3, col_idx])
                c1 = float(df_cyc.iloc[4, col_idx])
                c50 = float(df_cyc.iloc[53, col_idx])
                retention = (c50 / c1 * 100.0) if c1 > 0 else np.nan
                cyc_cell_map[c_id] = {
                    "cycle_0_capacity_c10_mah": c0,
                    "cycle_1_capacity_c2_mah": c1,
                    "cycle_50_capacity_c2_mah": c50,
                    "cycle_50_retention_pct": retention,
                }
            except Exception:
                pass

        runs: list[BatteryProcessRun] = []

        for _, row in df_tab.iterrows():
            cond_no = int(row.iloc[0])
            electrode_id = str(row.iloc[1]).strip()
            exp_id = f"EXP_{cond_no:02d}"
            roll_temp_c = float(row.iloc[2])
            target_density_g_cm3 = float(row.iloc[3])
            target_porosity_pct = float(row.iloc[4])

            cal_row = df_cal.iloc[1 + cond_no]
            target_coating_weight_gsm = float(cal_row.iloc[2])
            loading_regime = "HIGH" if target_coating_weight_gsm > 150 else "LOW"

            pre_thick_um = float(cal_row.iloc[8]) if pd.notna(cal_row.iloc[8]) else float(cal_row.iloc[7])
            pre_weight_gsm = float(cal_row.iloc[9])
            pre_density_g_cm3 = float(cal_row.iloc[10])
            pre_porosity_pct = float(cal_row.iloc[11])
            pre_tensile_kpa = float(cal_row.iloc[12]) if pd.notna(cal_row.iloc[12]) else None

            roll_gap_um = float(cal_row.iloc[21])
            num_passes = int(cal_row.iloc[22])

            post_thick_um = float(cal_row.iloc[24]) if pd.notna(cal_row.iloc[24]) else float(cal_row.iloc[23])
            post_weight_gsm = float(cal_row.iloc[25])
            post_density_g_cm3 = float(cal_row.iloc[26])
            post_porosity_pct = float(cal_row.iloc[27])
            post_tensile_kpa = float(cal_row.iloc[28]) if pd.notna(cal_row.iloc[28]) else None

            df_g = xl_el.parse(f"Group{cond_no}", header=None)

            rate_row_idx = None
            cell_id_row_idx = None
            d_c5_row_idx = None
            d_5c_row_idx = None
            fcl_row_idx = None

            for r in range(len(df_g)):
                txt = " ".join([str(x) for x in df_g.iloc[r].dropna().values])
                if "5C:0.2C" in txt:
                    rate_row_idx = r
                if "CellID" in txt or "Cell ID" in txt:
                    cell_id_row_idx = r
                if "At C/5- 6" in txt:
                    d_c5_row_idx = r
                if "At 5C" in txt and "Volumetric" not in txt and r < 110:
                    d_5c_row_idx = r
                if "First cycle loss" in txt:
                    fcl_row_idx = r

            c_ids = [str(df_g.iloc[cell_id_row_idx, c]).strip() for c in range(2, 5)]
            rate_nums = [float(df_g.iloc[rate_row_idx, c]) for c in range(2, 5)]
            d_c5_nums = [float(df_g.iloc[d_c5_row_idx, c]) for c in range(2, 5)] if d_c5_row_idx else [None, None, None]
            d_5c_nums = [float(df_g.iloc[d_5c_row_idx, c]) for c in range(2, 5)] if d_5c_row_idx else [None, None, None]
            fcl_nums = [float(df_g.iloc[fcl_row_idx, c]) for c in range(2, 5)] if fcl_row_idx else [None, None, None]
            active_mass_nums = [float(df_g.iloc[cell_id_row_idx + 1, c]) for c in range(2, 5)]

            for rep_idx, c_id in enumerate(c_ids):
                # Coating stage
                coating_ctrl = {
                    "target_coating_weight_gsm": ParameterValue(target_coating_weight_gsm, "g/m2"),
                }
                coating_obs = {
                    "pre_calendering_thickness_um": MeasurementValue(pre_thick_um, "um"),
                    "pre_calendering_coating_weight_gsm": MeasurementValue(pre_weight_gsm, "g/m2"),
                    "pre_calendering_density_g_cm3": MeasurementValue(pre_density_g_cm3, "g/cm3"),
                    "pre_calendering_porosity_pct": MeasurementValue(pre_porosity_pct, "%"),
                }
                if pre_tensile_kpa is not None and not np.isnan(pre_tensile_kpa):
                    coating_obs["pre_calendering_tensile_strength_kpa"] = MeasurementValue(pre_tensile_kpa, "kPa")

                # Calendering stage
                cal_ctrl = {
                    "roll_temperature_c": ParameterValue(roll_temp_c, "degC"),
                    "roll_gap_um": ParameterValue(roll_gap_um, "um"),
                    "number_of_passes": ParameterValue(float(num_passes), "count"),
                }
                cal_obs = {
                    "target_density_g_cm3": MeasurementValue(target_density_g_cm3, "g/cm3"),
                    "target_porosity_pct": MeasurementValue(target_porosity_pct, "%"),
                    "calendered_thickness_um": MeasurementValue(post_thick_um, "um"),
                    "calendered_coating_weight_gsm": MeasurementValue(post_weight_gsm, "g/m2"),
                    "calendered_density_g_cm3": MeasurementValue(post_density_g_cm3, "g/cm3"),
                    "calendered_porosity_pct": MeasurementValue(post_porosity_pct, "%"),
                }
                if post_tensile_kpa is not None and not np.isnan(post_tensile_kpa):
                    cal_obs["calendered_tensile_strength_kpa"] = MeasurementValue(post_tensile_kpa, "kPa")

                stages = [
                    StageRecord(f"{c_id}:coating", ProcessStage.COATING, 2, coating_ctrl, coating_obs, [], provenance=base_prov),
                    StageRecord(f"{c_id}:calendering", ProcessStage.CALENDERING, 4, cal_ctrl, cal_obs, [], upstream_stage_id=f"{c_id}:coating", provenance=base_prov),
                    StageRecord(f"{c_id}:final", ProcessStage.FINAL_CHARACTERIZATION, 8, {}, {}, [], upstream_stage_id=f"{c_id}:calendering", provenance=base_prov),
                ]

                kpis: dict[str, MeasurementValue] = {
                    "rate_performance_5c_over_0_2c": MeasurementValue(rate_nums[rep_idx], "dimensionless_ratio"),
                }
                if d_c5_nums[rep_idx] is not None and not np.isnan(d_c5_nums[rep_idx]):
                    kpis["discharge_capacity_0_2c_mah_g"] = MeasurementValue(d_c5_nums[rep_idx], "mAh/g")
                if d_5c_nums[rep_idx] is not None and not np.isnan(d_5c_nums[rep_idx]):
                    kpis["discharge_capacity_5c_mah_g"] = MeasurementValue(d_5c_nums[rep_idx], "mAh/g")
                if fcl_nums[rep_idx] is not None and not np.isnan(fcl_nums[rep_idx]):
                    kpis["first_cycle_loss_pct"] = MeasurementValue(fcl_nums[rep_idx], "%")

                cyc = cyc_cell_map.get(c_id, {})
                if "cycle_50_capacity_c2_mah" in cyc:
                    kpis["cycle_50_capacity_c2_mah"] = MeasurementValue(cyc["cycle_50_capacity_c2_mah"], "mAh")
                if "cycle_50_retention_pct" in cyc:
                    kpis["cycle_50_retention_pct"] = MeasurementValue(cyc["cycle_50_retention_pct"], "%")

                runs.append(
                    BatteryProcessRun(
                        run_id=c_id,
                        cell_id=c_id,
                        batch_id=exp_id,
                        chemistry_id="NMC622 cathode / lithium-metal half-cell",
                        equipment_context={"coater": "Megtec", "dryer": "Megtec", "calender": "pilot_scale"},
                        environment_context={"loading_regime": loading_regime},
                        stages=stages,
                        final_kpis=kpis,
                        provenance=base_prov,
                    )
                )

        if len(runs) != 54:
            raise ValueError(f"Expected exactly 54 NMC622 runs, found {len(runs)}")
        return runs

    def get_candidate_pool(self) -> pd.DataFrame:
        """Derive the 18 unique experimental/DOE conditions as candidates for rediscovery."""
        runs = self.load_runs()
        runs_by_exp: dict[str, list[BatteryProcessRun]] = {}
        for run in runs:
            runs_by_exp.setdefault(run.batch_id, []).append(run)

        rows: list[dict[str, Any]] = []
        for exp_id, exp_runs in sorted(runs_by_exp.items()):
            first_run = exp_runs[0]
            cal_stage = next(s for s in first_run.stages if s.stage_type == ProcessStage.CALENDERING)
            coat_stage = next(s for s in first_run.stages if s.stage_type == ProcessStage.COATING)

            rates = [r.final_kpis["rate_performance_5c_over_0_2c"].value for r in exp_runs]
            d5c = [r.final_kpis["discharge_capacity_5c_mah_g"].value for r in exp_runs if "discharge_capacity_5c_mah_g" in r.final_kpis]
            d02c = [r.final_kpis["discharge_capacity_0_2c_mah_g"].value for r in exp_runs if "discharge_capacity_0_2c_mah_g" in r.final_kpis]

            rows.append({
                "recipe_id": exp_id,
                "experiment_id": exp_id,
                "target_coating_weight_gsm": coat_stage.controls["target_coating_weight_gsm"].value,
                "roll_temperature_c": cal_stage.controls["roll_temperature_c"].value,
                "roll_gap_um": cal_stage.controls["roll_gap_um"].value,
                "number_of_passes": cal_stage.controls["number_of_passes"].value,
                "target_density_g_cm3": cal_stage.intermediate_properties["target_density_g_cm3"].value,
                "target_porosity_pct": cal_stage.intermediate_properties["target_porosity_pct"].value,
                "rate_performance_5c_over_0_2c": float(np.mean(rates)),
                "rate_performance_5c_over_0_2c_std": float(np.std(rates, ddof=1)),
                "discharge_capacity_5c_mah_g": float(np.mean(d5c)) if d5c else np.nan,
                "discharge_capacity_0_2c_mah_g": float(np.mean(d02c)) if d02c else np.nan,
                "num_replicates": len(exp_runs),
            })

        pool = pd.DataFrame(rows)
        if len(pool) != 18:
            raise ValueError(f"Candidate pool must have exactly 18 DOE conditions, got {len(pool)}")
        return pool

    def get_runs_by_recipe(self) -> dict[str, list[BatteryProcessRun]]:
        runs = self.load_runs()
        mapping: dict[str, list[BatteryProcessRun]] = {}
        for r in runs:
            mapping.setdefault(r.batch_id, []).append(r)
        return mapping

    def build_optimization_space(self, task: ProcessOptimizationTask) -> ProcessSearchSpace:
        pool = self.get_candidate_pool()
        return ProcessSearchSpace.from_finite_pool(pool, id_column="recipe_id")
