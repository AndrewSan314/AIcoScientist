from __future__ import annotations

import pytest

from src.process.contracts import BatteryProcessRun
from .conftest import process_run


def test_stages_must_follow_canonical_order() -> None:
    run = process_run()
    with pytest.raises(ValueError, match="sequence order"):
        BatteryProcessRun(run.run_id, run.cell_id, run.batch_id, run.chemistry_id, {}, {}, list(reversed(run.stages)), run.final_kpis, run.provenance)
