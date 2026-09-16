"""Tests validating Warwick Ultrasonic Adapter, Grouped CV Split, and Leakage Firewall."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.datasets.battery_process.warwick_ultrasonic import WarwickUltrasonicAdapter
from src.process.contracts import BatteryProcessRun, ModalityType
from src.process.stages import ProcessStage


@pytest.fixture(scope="module")
def ultrasonic_adapter() -> WarwickUltrasonicAdapter:
    repo_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = repo_root / "data" / "external" / "warwick_ultrasonic" / "raw"
    if not raw_dir.exists():
        pytest.skip(f"Raw directory not found at {raw_dir}")
    return WarwickUltrasonicAdapter(raw_dir)


def test_ultrasonic_adapter_load_runs(ultrasonic_adapter: WarwickUltrasonicAdapter) -> None:
    """Verify adapter loads exactly 18 Cathode and 30 Anode runs (48 total)."""
    runs = ultrasonic_adapter.load_runs()
    assert len(runs) == 48

    cathode_runs = ultrasonic_adapter.load_cathode_runs()
    anode_runs = ultrasonic_adapter.load_anode_runs()

    assert len(cathode_runs) == 18
    assert len(anode_runs) == 30

    sample_run = runs[0]
    assert sample_run.provenance.evidence_kind == "PHYSICAL_HISTORICAL"
    assert "10.17632/c62yn37d9h.4" in sample_run.provenance.source_doi


def test_ultrasonic_spectra_integrity(ultrasonic_adapter: WarwickUltrasonicAdapter) -> None:
    """Verify each run contains valid pre- and post-calendering ultrasonic spectra and physical measurements."""
    runs = ultrasonic_adapter.load_runs()
    for run in runs:
        # Pre-calendering stage (COATING)
        coating_stage = [s for s in run.stages if s.stage_type == ProcessStage.COATING][0]
        assert len(coating_stage.modalities) == 1
        mod_before = coating_stage.modalities[0]
        assert mod_before.modality_type == ModalityType.ULTRASOUND_SPECTRUM
        assert len(mod_before.values["fft_frequency"]) > 0
        assert len(mod_before.values["fft_magnitude"]) == len(mod_before.values["fft_frequency"])

        # Physical measurements pre-calendering
        assert "pre_calendering_thickness_um" in coating_stage.intermediate_properties
        assert "pre_calendering_density_g_cm3" in coating_stage.intermediate_properties
        t_before = coating_stage.intermediate_properties["pre_calendering_thickness_um"].value
        d_before = coating_stage.intermediate_properties["pre_calendering_density_g_cm3"].value
        assert t_before > 0
        assert d_before > 0

        # Calendering stage
        cal_stage = [s for s in run.stages if s.stage_type == ProcessStage.CALENDERING][0]
        assert "roll_gap_um" in cal_stage.controls
        assert len(cal_stage.modalities) == 1
        mod_after = cal_stage.modalities[0]
        assert mod_after.modality_type == ModalityType.ULTRASOUND_SPECTRUM
        assert len(mod_after.values["fft_frequency"]) > 0

        # Final KPIs (post-calendering state)
        assert "post_calendering_thickness_um" in run.final_kpis
        assert "post_calendering_density_g_cm3" in run.final_kpis
        t_after = run.final_kpis["post_calendering_thickness_um"].value
        d_after = run.final_kpis["post_calendering_density_g_cm3"].value
        assert t_after > 0
        assert d_after > 0
        # Post-calendering density is higher due to compaction
        assert d_after >= d_before


def test_ultrasonic_leakage_firewall_and_grouped_cv() -> None:
    """Verify benchmark split_manifest.json strictly isolates Sample_IDs without cross-fold leakage."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    split_path = repo_root / "outputs" / "warwick_ultrasonic" / "split_manifest.json"
    if not split_path.exists():
        pytest.skip(f"Split manifest not found at {split_path}")

    with open(split_path) as f:
        splits_data = json.load(f)

    for material, sample_fold_map in splits_data.items():
        expected_count = 18 if material == "Cathode" else 30
        assert len(sample_fold_map) == expected_count

        # Fold indices must span exactly 1 to 5
        fold_indices = set(sample_fold_map.values())
        assert fold_indices == {1, 2, 3, 4, 5}

        # Check disjointness across folds
        samples_per_fold: dict[int, set[str]] = {k: set() for k in range(1, 6)}
        for sample_id, fold_idx in sample_fold_map.items():
            assert 1 <= fold_idx <= 5
            samples_per_fold[fold_idx].add(sample_id)

        for f1 in range(1, 6):
            for f2 in range(f1 + 1, 6):
                overlap = samples_per_fold[f1].intersection(samples_per_fold[f2])
                assert len(overlap) == 0, f"Sample overlap between fold {f1} and {f2}: {overlap}"


def test_ultrasonic_pre_decision_horizon_isolation(ultrasonic_adapter: WarwickUltrasonicAdapter) -> None:
    """Verify that stage transition pre-decision context (z_t + u_{t+1}) hides all post-calendering information."""
    runs = ultrasonic_adapter.load_runs()
    for run in runs[:5]:
        coating_stage = [s for s in run.stages if s.stage_type == ProcessStage.COATING][0]
        cal_stage = [s for s in run.stages if s.stage_type == ProcessStage.CALENDERING][0]

        # Stage t (COATING) intermediate state z_t is known
        z_t_thickness = coating_stage.intermediate_properties["pre_calendering_thickness_um"].value
        z_t_density = coating_stage.intermediate_properties["pre_calendering_density_g_cm3"].value
        assert z_t_thickness > 0
        assert z_t_density > 0

        # Stage t+1 actuation control u_{t+1} is known
        u_next_roll_gap = cal_stage.controls["roll_gap_um"].value
        assert u_next_roll_gap > 0

        # But outcome z_{t+1} MUST NOT be in cal_stage controls
        assert "post_calendering_thickness_um" not in cal_stage.controls
        assert "post_calendering_density_g_cm3" not in cal_stage.controls
        assert "calendered_thickness_um" not in cal_stage.controls
        assert "calendered_density_g_cm3" not in cal_stage.controls
