from __future__ import annotations

import hashlib
import json
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
from src.process.modalities import ModalityObservation, ModalityType
from src.process.stages import ProcessStage


class WarwickUltrasonicAdapter(NormalizedRunAdapter):
    """Adapter for Warwick Frequency-Domain Ultrasonic Electrode Dataset (Version 4).

    Source DOI: 10.17632/c62yn37d9h.4
    Includes:
    - 18 NMC622 Cathode samples
    - 30 Graphite Anode samples
    - Paired before-calendering and after-calendering ultrasonic frequency spectra and metrology
    """

    ADAPTER_VERSION = "2"
    SCHEMA_VERSION = "1"
    ACCEPTED_EVIDENCE_KINDS = ("PHYSICAL_HISTORICAL", "TEST_FIXTURE")

    def __init__(self, root: str | Path | None = None) -> None:
        if root is not None:
            self.root = Path(root)
        else:
            candidates = [
                Path("data/external/warwick_ultrasonic/raw/Frequency-Domain Ultrasonic Signal Dataset for Bat"),
                Path("data/external/warwick_ultrasonic/raw"),
                Path("data/external/Frequency-Domain Ultrasonic Signal Dataset for Bat"),
                Path("data/external/warwick_ultrasound/4"),
            ]
            found = None
            for c in candidates:
                if (c / "Cathode").exists() and (c / "Anode").exists():
                    found = c
                    break
            self.root = found or Path("data/external/warwick_ultrasonic/raw")

    def metadata(self) -> BatteryDatasetMetadata:
        return BatteryDatasetMetadata(
            dataset_id="warwick_ultrasonic",
            display_name="Warwick Ultrasonic Inline Metrology",
            chemistry="Graphite anode / NMC622 cathode",
            evidence_kind="PHYSICAL_HISTORICAL",
            source_doi="10.17632/c62yn37d9h.4",
            version="4",
            license="CC BY 4.0",
            process_stages=(ProcessStage.COATING, ProcessStage.CALENDERING, ProcessStage.FINAL_CHARACTERIZATION),
            modalities=("ULTRASOUND_SPECTRUM", "PROCESS_TABULAR", "SCALAR_METROLOGY"),
            recommended_splits=("GROUP_BY_SAMPLE_ID",),
            optimization_capable=False,
            multimodal_capable=True,
            limitations="Inline non-destructive ultrasonic inspection before and after calendering; strictly evaluated for state-transition prediction z_t + u_{t+1} -> z_{t+1}.",
        )

    def _find_data_root(self) -> Path:
        candidates = [
            self.root,
            self.root / "Frequency-Domain Ultrasonic Signal Dataset for Bat",
            Path("data/external/warwick_ultrasonic/raw/Frequency-Domain Ultrasonic Signal Dataset for Bat"),
            Path("data/external/warwick_ultrasonic/raw"),
            Path("data/external/Frequency-Domain Ultrasonic Signal Dataset for Bat"),
        ]
        for c in candidates:
            if (c / "Cathode").exists() and (c / "Anode").exists():
                return c
        raise RawDatasetUnavailableError(
            "WARWICK_ULTRASONIC_OFFICIAL_DATA_UNAVAILABLE: Could not locate Cathode and Anode folders. "
            "Required DOI: 10.17632/c62yn37d9h.4"
        )

    def _raw_hashes(self) -> dict[str, str]:
        archive_candidates = [
            self.root / "frequency_domain_ultrasound_v4.zip",
            self.root.parent / "frequency_domain_ultrasound_v4.zip",
            Path("data/external/warwick_ultrasonic/raw/frequency_domain_ultrasound_v4.zip"),
        ]
        for a in archive_candidates:
            if a.exists():
                return {a.name: compute_file_sha256(a)}
        return {}

    def load_runs(self, material: str | None = None) -> list[BatteryProcessRun]:
        data_root = self._find_data_root()
        raw_hashes = self._raw_hashes()
        base_prov = ProvenanceRecord(
            evidence_kind="PHYSICAL_HISTORICAL",
            source_url="https://data.mendeley.com/datasets/c62yn37d9h/4",
            source_doi="10.17632/c62yn37d9h.4",
            source_version="4",
            raw_hashes=raw_hashes,
            adapter_version=self.ADAPTER_VERSION,
        )

        materials_to_load = [material] if material else ["Cathode", "Anode"]
        runs: list[BatteryProcessRun] = []

        for mat in materials_to_load:
            mat_dir = data_root / mat
            if not mat_dir.exists():
                continue
            sample_dirs = sorted([d for d in mat_dir.iterdir() if d.is_dir() and d.name not in ["ToF", "ToF (ms)"]])
            for s_dir in sample_dirs:
                sample_id = s_dir.name
                p_before = s_dir / "before-calendering.json"
                p_after = s_dir / "after-calendering.json"

                if not p_before.is_file() or not p_after.is_file():
                    continue

                with open(p_before, "r", encoding="utf-8") as f:
                    data_b = json.load(f)
                with open(p_after, "r", encoding="utf-8") as f:
                    data_a = json.load(f)

                meta_b = data_b.get("metadata", {})
                meta_a = data_a.get("metadata", {})

                # Frequencies and magnitudes
                freq_b = data_b.get("fft_frequency", [])
                mag_b = data_b.get("fft_magnitude", [])
                freq_a = data_a.get("fft_frequency", [])
                mag_a = data_a.get("fft_magnitude", [])

                # Coating / Pre-calendering state
                pre_thick = float(meta_b.get("Thickness", np.nan))
                pre_density = float(meta_b.get("Density", np.nan))

                coating_controls: dict[str, ParameterValue] = {}
                if "Coat_Weight" in meta_b and pd.notna(meta_b["Coat_Weight"]):
                    coating_controls["coat_weight_gsm"] = ParameterValue(float(meta_b["Coat_Weight"]), "g/m2")

                coating_obs: dict[str, MeasurementValue] = {}
                if pd.notna(pre_thick):
                    coating_obs["pre_calendering_thickness_um"] = MeasurementValue(pre_thick, "um")
                if pd.notna(pre_density):
                    coating_obs["pre_calendering_density_g_cm3"] = MeasurementValue(pre_density, "g/cm3")

                coating_modalities = [
                    ModalityObservation(
                        modality_id=f"{sample_id}:before_spectrum",
                        modality_type=ModalityType.ULTRASOUND_SPECTRUM,
                        observed_at_stage=ProcessStage.COATING,
                        source_path=str(p_before.relative_to(data_root)),
                        values={"fft_frequency": freq_b, "fft_magnitude": mag_b},
                        units="MHz/a.u.",
                        shape=(len(mag_b),),
                        provenance=base_prov,
                    )
                ]

                # Calendering actuation
                cal_controls: dict[str, ParameterValue] = {}
                if "Roll_Gap" in meta_b and pd.notna(meta_b["Roll_Gap"]):
                    cal_controls["roll_gap_um"] = ParameterValue(float(meta_b["Roll_Gap"]), "um")
                if "Web_speed" in meta_b and pd.notna(meta_b["Web_speed"]):
                    cal_controls["web_speed_m_min"] = ParameterValue(float(meta_b["Web_speed"]), "m/min")
                if "Calendering_Speed" in meta_b and pd.notna(meta_b["Calendering_Speed"]):
                    cal_controls["calendering_speed_m_min"] = ParameterValue(float(meta_b["Calendering_Speed"]), "m/min")

                # Post-calendering state
                post_thick = float(meta_a.get("Thickness", np.nan))
                post_density = float(meta_a.get("Density", np.nan))

                cal_obs: dict[str, MeasurementValue] = {}
                if pd.notna(post_thick):
                    cal_obs["calendered_thickness_um"] = MeasurementValue(post_thick, "um")
                if pd.notna(post_density):
                    cal_obs["calendered_density_g_cm3"] = MeasurementValue(post_density, "g/cm3")

                cal_modalities = [
                    ModalityObservation(
                        modality_id=f"{sample_id}:after_spectrum",
                        modality_type=ModalityType.ULTRASOUND_SPECTRUM,
                        observed_at_stage=ProcessStage.CALENDERING,
                        source_path=str(p_after.relative_to(data_root)),
                        values={"fft_frequency": freq_a, "fft_magnitude": mag_a},
                        units="MHz/a.u.",
                        shape=(len(mag_a),),
                        provenance=base_prov,
                    )
                ]

                stages = [
                    StageRecord(f"{mat}:{sample_id}:coating", ProcessStage.COATING, 2, coating_controls, coating_obs, coating_modalities, provenance=base_prov),
                    StageRecord(f"{mat}:{sample_id}:calendering", ProcessStage.CALENDERING, 4, cal_controls, cal_obs, cal_modalities, upstream_stage_id=f"{mat}:{sample_id}:coating", provenance=base_prov),
                    StageRecord(f"{mat}:{sample_id}:final", ProcessStage.FINAL_CHARACTERIZATION, 8, {}, {}, [], upstream_stage_id=f"{mat}:{sample_id}:calendering", provenance=base_prov),
                ]

                kpis: dict[str, MeasurementValue] = {}
                if pd.notna(post_thick):
                    kpis["post_calendering_thickness_um"] = MeasurementValue(post_thick, "um")
                if pd.notna(post_density):
                    kpis["post_calendering_density_g_cm3"] = MeasurementValue(post_density, "g/cm3")

                runs.append(
                    BatteryProcessRun(
                        run_id=f"{mat}:{sample_id}",
                        cell_id=sample_id,
                        batch_id=sample_id,
                        chemistry_id=f"{mat.lower()} electrode",
                        equipment_context={"ultrasonic_sensor": "frequency_domain_transducer"},
                        environment_context={"material": mat},
                        stages=stages,
                        final_kpis=kpis,
                        provenance=base_prov,
                    )
                )

        return runs

    def load_cathode_runs(self) -> list[BatteryProcessRun]:
        return self.load_runs(material="Cathode")

    def load_anode_runs(self) -> list[BatteryProcessRun]:
        return self.load_runs(material="Anode")
