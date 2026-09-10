from __future__ import annotations

from src.datasets.battery_process import ArtisticSimulationAdapter, DrakopoulosGraphiteAdapter, NaIonHTEAdapter, WarwickNMC622Adapter, WarwickUltrasoundAdapter


def test_bpss_has_independent_source_adapters_and_simulation_label() -> None:
    adapters = [DrakopoulosGraphiteAdapter(), WarwickNMC622Adapter(), WarwickUltrasoundAdapter(), NaIonHTEAdapter(), ArtisticSimulationAdapter()]
    assert len({adapter.metadata().dataset_id for adapter in adapters}) == 5
    assert ArtisticSimulationAdapter().metadata().evidence_kind == "SIMULATED_PHYSICS"
