from __future__ import annotations

import pytest

from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosGraphiteAdapter
from src.datasets.battery_process.base import RawDatasetUnavailableError
from .conftest import process_run


def test_adapter_fails_closed_before_a_source_audit(tmp_path) -> None:
    adapter = DrakopoulosGraphiteAdapter(tmp_path)
    with pytest.raises(RawDatasetUnavailableError):
        adapter.load_runs()


def test_adapter_loads_a_hashed_normalized_cache(tmp_path) -> None:
    adapter = DrakopoulosGraphiteAdapter(tmp_path)
    adapter.write_processed_cache([process_run()], raw_hashes={"raw.csv": "a" * 64})
    assert adapter.load_runs()[0].run_id == "run-1"
