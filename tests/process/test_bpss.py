from __future__ import annotations

import numpy as np

from src.datasets.battery_process import ArtisticSimulationAdapter, DrakopoulosGraphiteAdapter, NaIonHTEAdapter, WarwickNMC622Adapter, WarwickUltrasoundAdapter
from scripts.run_battery_process_benchmark import _metrics


def test_bpss_has_independent_source_adapters_and_simulation_label() -> None:
    adapters = [DrakopoulosGraphiteAdapter(), WarwickNMC622Adapter(), WarwickUltrasoundAdapter(), NaIonHTEAdapter(), ArtisticSimulationAdapter()]
    assert len({adapter.metadata().dataset_id for adapter in adapters}) == 5
    assert ArtisticSimulationAdapter().metadata().evidence_kind == "SIMULATED_PHYSICS"


def test_bpss_metrics_report_zero_error_for_exact_predictions() -> None:
    report = _metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert report == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}
