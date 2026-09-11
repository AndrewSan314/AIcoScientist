from __future__ import annotations

import pytest

from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosGraphiteAdapter
from .conftest import process_run


def test_processed_cache_requires_raw_hashes(tmp_path) -> None:
    with pytest.raises(ValueError, match="raw_hashes"):
        DrakopoulosGraphiteAdapter(tmp_path).write_processed_cache([process_run()], raw_hashes={})
