from __future__ import annotations

import json

import pytest

from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosGraphiteAdapter
from src.datasets.battery_process.base import RawDatasetUnavailableError
from src.datasets.battery_process.warwick_ultrasound import WarwickUltrasoundAdapter
from src.process.modalities import ModalityType
from .conftest import process_run


def test_adapter_fails_closed_before_a_source_audit(tmp_path) -> None:
    adapter = DrakopoulosGraphiteAdapter(tmp_path)
    with pytest.raises(RawDatasetUnavailableError):
        adapter.load_runs()


def test_adapter_loads_a_hashed_normalized_cache(tmp_path) -> None:
    adapter = DrakopoulosGraphiteAdapter(tmp_path)
    adapter.write_processed_cache([process_run()], raw_hashes={"raw.csv": "a" * 64})
    assert adapter.load_runs()[0].run_id == "run-1"


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
