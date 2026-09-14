from __future__ import annotations

import numpy as np

from src.datasets.battery_process import ArtisticSimulationAdapter, DrakopoulosGraphiteAdapter, NaIonHTEAdapter, WarwickNMC622Adapter, WarwickUltrasoundAdapter
from scripts.run_battery_process_benchmark import _latency_report, _metrics, _ultrasound_ablation


def test_bpss_has_independent_source_adapters_and_simulation_label() -> None:
    adapters = [DrakopoulosGraphiteAdapter(), WarwickNMC622Adapter(), WarwickUltrasoundAdapter(), NaIonHTEAdapter(), ArtisticSimulationAdapter()]
    assert len({adapter.metadata().dataset_id for adapter in adapters}) == 5
    assert ArtisticSimulationAdapter().metadata().evidence_kind == "SIMULATED_PHYSICS"


def test_bpss_metrics_report_zero_error_for_exact_predictions() -> None:
    report = _metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert report == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_ultrasound_gated_fusion_is_source_backed_and_reports_dropout_stress() -> None:
    report = _ultrasound_ablation(WarwickUltrasoundAdapter(), seed=42)
    modes = {item["mode"]: item for item in report["reports"]}
    assert report["gated_fusion"]["status"] == "EVALUATED"
    assert report["cross_attention"]["status"] == "EVALUATED"
    assert modes["gated_missing_aware_fusion"]["metrics"]["rmse"] >= 0
    assert modes["cross_attention_set_fusion"]["experimental"] is True
    assert modes["gated_fusion_signal_dropout"]["derived_stress"] is True
    partial = [item for item in report["reports"] if item["mode"] == "gated_fusion_partial_signal_dropout"]
    assert [item["requested_dropout_rate"] for item in partial] == [0.10, 0.25, 0.50, 0.75]
    assert all(item["missing_modality"] == "ultrasound" and item["derived_stress"] for item in partial)
    assert all(item["evidence_kind"] == "SIMULATED_STRESS" and item["dropped_source_run_ids"] and item["stress_transform_fingerprint"] for item in partial)


def test_latency_report_records_warmup_and_tail_quantiles() -> None:
    report = _latency_report([{"status": "EVALUATED", "candidate_pool_size": 5, "warmup_decision_latency_seconds": 0.01, "trajectory": [{"decision_latency_seconds": 0.02}, {"decision_latency_seconds": 0.03}]}])
    assert report["status"] == "EVALUATED"
    assert report["warmup"]["count"] == 1
    assert report["optimizer_proposal"]["p99_ms"] >= report["optimizer_proposal"]["p95_ms"]
