from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.datasets.battery_process import ArtisticSimulationAdapter, DrakopoulosGraphiteAdapter, NaIonHTEAdapter, WarwickNMC622Adapter, WarwickUltrasoundAdapter
from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.modalities import ModalityObservation, ModalityType
from src.process.stages import ProcessStage, STAGE_ORDER
from scripts.run_battery_process_benchmark import _latency_report, _metrics, _ultrasound_ablation


def test_bpss_has_independent_source_adapters_and_simulation_label() -> None:
    adapters = [DrakopoulosGraphiteAdapter(), WarwickNMC622Adapter(), WarwickUltrasoundAdapter(), NaIonHTEAdapter(), ArtisticSimulationAdapter()]
    assert len({adapter.metadata().dataset_id for adapter in adapters}) == 5
    assert ArtisticSimulationAdapter().metadata().evidence_kind == "SIMULATED_PHYSICS"


def test_bpss_metrics_report_zero_error_for_exact_predictions() -> None:
    report = _metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert report == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


@pytest.mark.external_data
def test_ultrasound_gated_fusion_is_source_backed_and_reports_dropout_stress() -> None:
    adapter = WarwickUltrasoundAdapter()
    if not adapter.normalized_runs_path.is_file():
        pytest.skip("Warwick ultrasound normalized data not present in local workspace")
    report = _ultrasound_ablation(adapter, seed=42)
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


def test_ultrasound_gated_fusion_software_invariants_with_fixture(tmp_path: Path) -> None:
    prov = ProvenanceRecord(evidence_kind="TEST_FIXTURE", raw_hashes={"raw_fixture.csv": "a" * 64})
    runs: list[BatteryProcessRun] = []
    for i in range(8):
        stage = StageRecord(
            stage_id=f"stage-cal-{i}",
            stage_type=ProcessStage.CALENDERING,
            sequence_index=STAGE_ORDER[ProcessStage.CALENDERING],
            controls={"calendering_gap_um": ParameterValue(20.0 + i * 2.5), "calendering_pressure_mpa": ParameterValue(100.0 + i * 5.0)},
            intermediate_properties={},
            modalities=[
                ModalityObservation(
                    modality_id="calendering_before_spectrum",
                    modality_type=ModalityType.ULTRASOUND_SPECTRUM,
                    observed_at_stage=ProcessStage.CALENDERING,
                    values={"fft_magnitude": [1.0 + i * 0.5, 2.0, 3.0, 4.0 + i * 0.1]},
                    provenance=prov,
                )
            ],
            provenance=prov,
        )
        run = BatteryProcessRun(
            run_id=f"run-{i}",
            cell_id=f"cell-{i}",
            batch_id=f"batch-{i % 4}",
            chemistry_id="NMC622",
            equipment_context={},
            environment_context={},
            stages=[stage],
            final_kpis={"post_calendering_thickness_um": MeasurementValue(60.0 + i * 1.5)},
            provenance=prov,
        )
        runs.append(run)

    adapter = WarwickUltrasoundAdapter(root=tmp_path)
    adapter.write_processed_cache(runs, raw_hashes={"raw_fixture.csv": "a" * 64})

    report = _ultrasound_ablation(adapter, seed=42)
    assert report["status"] == "EVALUATED"
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
