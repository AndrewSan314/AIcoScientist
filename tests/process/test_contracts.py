from __future__ import annotations

import pytest

from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.stages import ProcessStage
from .conftest import process_run


def test_process_identity_is_serialization_stable() -> None:
    run = process_run()
    assert BatteryProcessRun.from_dict(run.to_dict()).identity_fingerprint == run.identity_fingerprint


def test_controls_and_observations_cannot_share_a_name() -> None:
    with pytest.raises(ValueError, match="distinct"):
        StageRecord("x", ProcessStage.MIXING, 1, {"speed": ParameterValue(1)}, {"speed": MeasurementValue(1)}, [])
