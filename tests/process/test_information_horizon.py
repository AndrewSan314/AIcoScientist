from __future__ import annotations

import pytest

from src.process.information_horizon import InformationHorizon, InformationHorizonError
from src.process.stages import ProcessStage
from .conftest import process_run


def test_calendering_horizon_excludes_current_and_final_outcomes() -> None:
    horizon = InformationHorizon(ProcessStage.CALENDERING)
    view = horizon.project(process_run())
    assert "drying.temperature" in view.controls
    assert "calendering.pressure" not in view.controls
    with pytest.raises(InformationHorizonError):
        horizon.assert_visible(process_run(), ["calendering.porosity", "capacity"])
