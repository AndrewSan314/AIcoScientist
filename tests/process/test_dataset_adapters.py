from __future__ import annotations

import json
import zipfile
from copy import deepcopy
from dataclasses import replace

import pytest

from src.datasets.battery_process.base import ProcessPredictionTask, RawDatasetUnavailableError
from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosGraphiteAdapter
from src.datasets.battery_process.naion_hte import NaIonHTEAdapter
from src.datasets.battery_process.warwick_ultrasound import WarwickUltrasoundAdapter
from src.process.modalities import ModalityType
from src.process.stages import ProcessStage
from .conftest import process_run


def test_adapter_fails_closed_before_a_source_audit(tmp_path) -> None:
    adapter = DrakopoulosGraphiteAdapter(tmp_path)
    with pytest.raises(RawDatasetUnavailableError):
        adapter.load_runs()


def test_adapter_loads_a_hashed_normalized_cache(tmp_path) -> None:
    adapter = DrakopoulosGraphiteAdapter(tmp_path)
    adapter.write_processed_cache([process_run()], raw_hashes={"raw.csv": "a" * 64})
    assert adapter.load_runs()[0].run_id == "run-1"


def test_training_view_marks_absent_numeric_fields_instead_of_treating_them_as_zero(tmp_path) -> None:
    adapter = DrakopoulosGraphiteAdapter(tmp_path)
    partial = deepcopy(process_run())
    partial = replace(partial, run_id="run-2", cell_id="cell-2", batch_id="batch-b")
    partial.stages[0].controls.pop("solids")
    adapter.write_processed_cache([process_run(), partial], raw_hashes={"raw.csv": "a" * 64})

    frame = adapter.build_training_view(ProcessPredictionTask("capacity", ProcessStage.FINAL_CHARACTERIZATION))

    assert frame.features["formulation.solids"].tolist() == [0.6, 0.0]
    assert frame.features["formulation.solids__observed"].tolist() == [1.0, 0.0]


def test_ultrasound_adapter_parses_paired_fft_records(tmp_path) -> None:
    source_dir = tmp_path / "raw" / "unpacked" / "source" / "Anode" / "GA_1"
    source_dir.mkdir(parents=True)
    (tmp_path / "raw" / "source.zip").write_bytes(b"source")
    for state, thickness in (("before-calendering", 46.0), ("after-calendering", 34.0)):
        (source_dir / f"{state}.json").write_text(json.dumps({
            "metadata": {"Sample_ID": "GA_1", "Calendering_State": state, "Calendering_Speed": 1.0, "Roll_Gap": 3.0, "Thickness": thickness, "Density": 1.2},
            "fft_frequency": [0.8, 1.2], "fft_magnitude": [0.1, 1.0],
        }), encoding="utf-8")

    run = WarwickUltrasoundAdapter(tmp_path).load_runs()[0]

    assert run.final_kpis["post_calendering_thickness_um"].value == 34.0
    assert [item.modality_type for item in run.stages[0].modalities] == [ModalityType.ULTRASOUND_SPECTRUM] * 2


def test_naion_adapter_streams_source_csv_traces(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = "Data_Point,Cycle_Index,Test_Time(s),Current(A),Voltage(V),Charge_Capacity(Ah),Discharge_Capacity(Ah)\n1,1,1,0.1,3.0,0.01,0\n2,1,2,-0.1,2.9,0.01,0.009\n"
    with zipfile.ZipFile(raw / "Na-upscaling.zip", "w") as archive:
        archive.writestr("Na-upscaling/C20Form/test_cell35_Channel_12_Wb_1.CSV", rows)
    run = NaIonHTEAdapter(tmp_path).load_runs()[0]
    assert run.cell_id == "35"
    assert run.final_kpis["last_observed_discharge_capacity_ah"].value == 0.009
    assert run.stages[1].modalities[0].modality_type == ModalityType.CYCLING_CURVE
