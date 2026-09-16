"""Tests validating Warwick Ultrasonic Adapter, Grouped CV Split, Leakage Firewall, and Production Transitions."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import pytest
import numpy as np
import pandas as pd
import torch

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.audit_multi_dataset_result_consistency import run_consistency_audit
from src.datasets.battery_process.warwick_ultrasonic import WarwickUltrasonicAdapter
from src.process.contracts import BatteryProcessRun, MeasurementValue, ModalityType, StageRecord
from src.process.information_horizon import InformationHorizon
from src.process.modalities import ModalitySlotSpec, SourceBoundModalityInput, source_values_fingerprint
from src.process.models.maspo import MASPOProcessStateModel
from src.process.models.transitions import LegalStageTransition, StageFeatureEncoder
from src.process.stages import ProcessStage


@pytest.fixture(scope="module")
def ultrasonic_adapter() -> WarwickUltrasonicAdapter:
    repo_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = repo_root / "data" / "external" / "warwick_ultrasonic" / "raw"
    if not raw_dir.exists():
        pytest.skip(f"Raw directory not found at {raw_dir}")
    return WarwickUltrasonicAdapter(raw_dir)


# -----------------------------------------------------------------------------
# Test 1: Adapter Loading and Provenance
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Test 2: Spectra Integrity
# -----------------------------------------------------------------------------
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
        assert d_after >= d_before


# -----------------------------------------------------------------------------
# Test 3: Leakage Firewall & Grouped CV Split
# -----------------------------------------------------------------------------
def test_ultrasonic_leakage_firewall_and_grouped_cv() -> None:
    """Verify benchmark split_manifest.json strictly isolates Sample_IDs without cross-fold leakage."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    split_path = repo_root / "outputs" / "warwick_ultrasonic" / "split_manifest.json"
    assert split_path.exists(), f"Split manifest not found at {split_path}"

    with open(split_path) as f:
        splits_data = json.load(f)

    for material, sample_fold_map in splits_data.items():
        expected_count = 18 if material == "Cathode" else 30
        assert len(sample_fold_map) == expected_count

        fold_indices = set(sample_fold_map.values())
        assert fold_indices == {1, 2, 3, 4, 5}

        samples_per_fold: dict[int, set[str]] = {k: set() for k in range(1, 6)}
        for sample_id, fold_idx in sample_fold_map.items():
            assert 1 <= fold_idx <= 5
            samples_per_fold[fold_idx].add(sample_id)

        for f1 in range(1, 6):
            for f2 in range(f1 + 1, 6):
                overlap = samples_per_fold[f1].intersection(samples_per_fold[f2])
                assert len(overlap) == 0, f"Sample overlap between fold {f1} and {f2}: {overlap}"


# -----------------------------------------------------------------------------
# Test 4: Pre-Decision Horizon Isolation
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Test 5: Frequency Grid Audit
# -----------------------------------------------------------------------------
def test_ultrasonic_frequency_grid_audit() -> None:
    """Verify frequency_grid_audit.json records native alignment and no interpolation."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    audit_path = repo_root / "outputs" / "warwick_ultrasonic" / "frequency_grid_audit.json"
    assert audit_path.exists(), f"Audit file not found at {audit_path}"
    with open(audit_path) as f:
        data = json.load(f)

    assert data["materials"]["Cathode"]["num_samples"] == 18
    assert data["materials"]["Cathode"]["num_points_per_spectrum"] == 29
    assert data["materials"]["Cathode"]["frequency_grid_aligned"] is True
    assert data["materials"]["Cathode"]["interpolation_required"] is False

    assert data["materials"]["Anode"]["num_samples"] == 30
    assert data["materials"]["Anode"]["num_points_per_spectrum"] == 36
    assert data["materials"]["Anode"]["frequency_grid_aligned"] is True
    assert data["materials"]["Anode"]["interpolation_required"] is False


# -----------------------------------------------------------------------------
# Test 6: Production Execution Trace Audit
# -----------------------------------------------------------------------------
def test_ultrasonic_production_execution_trace() -> None:
    """Verify ultrasonic benchmark records truthful non-zero execution counts and test_only == 0."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    trace_path = repo_root / "outputs" / "warwick_ultrasonic" / "execution_trace_audit.json"
    assert trace_path.exists(), f"Execution trace audit not found at {trace_path}"
    with open(trace_path) as f:
        data = json.load(f)

    trace = data["execution_trace"]
    assert trace["battery_process_runs_seen"] >= 48
    assert trace["horizon_projection_count"] >= 48
    assert trace["stage_feature_encoder_fit_count"] > 0
    assert trace["encoded_source_transition_count"] > 0
    assert trace["source_bound_modality_count"] > 0
    assert trace["maspo_public_forward_count"] > 0
    assert trace["stage_aware_public_transition_count"] > 0
    assert trace["final_prediction_count"] > 0
    assert trace["test_only_transition_count"] == 0


# -----------------------------------------------------------------------------
# Test 7: Zero Lookahead Perturbation Test
# -----------------------------------------------------------------------------
def test_no_lookahead_perturbation(ultrasonic_adapter: WarwickUltrasonicAdapter) -> None:
    """Perturbing post-calendering properties must NOT affect the pre-decision transition."""
    runs = ultrasonic_adapter.load_cathode_runs()
    r = runs[0]
    horizon = InformationHorizon(ProcessStage.CALENDERING, include_decision_stage_controls=True)
    v_orig = horizon.project(r)

    # Perturb the post-calendering stage measurements in a clone
    cal_stage = [s for s in r.stages if s.stage_type == ProcessStage.CALENDERING][0]
    cal_perturbed = replace(
        cal_stage,
        intermediate_properties={
            **cal_stage.intermediate_properties,
            "calendered_thickness_um": MeasurementValue(999999.0, "um"),
            "calendered_density_g_cm3": MeasurementValue(999999.0, "g/cm3"),
        },
    )
    r_perturbed = replace(
        r,
        stages=tuple(cal_perturbed if s.stage_type == ProcessStage.CALENDERING else s for s in r.stages),
        final_kpis={"post_calendering_thickness_um": MeasurementValue(999999.0, "um")},
    )
    v_perturbed = horizon.project(r_perturbed)

    # Pre-decision controls, intermediate properties, and modalities must be identical
    assert v_orig.controls == v_perturbed.controls
    assert v_orig.intermediate_properties == v_perturbed.intermediate_properties
    assert len(v_orig.modalities) == len(v_perturbed.modalities)
    assert np.allclose(v_orig.modalities[0].values["fft_magnitude"], v_perturbed.modalities[0].values["fft_magnitude"])


# -----------------------------------------------------------------------------
# Test 8: Source-Bound Transition Provenance
# -----------------------------------------------------------------------------
def test_source_bound_transition_provenance(ultrasonic_adapter: WarwickUltrasonicAdapter) -> None:
    """Verify LegalStageTransition.from_encoded_source_stage produces unsafe_test_only is None and valid fingerprint."""
    runs = ultrasonic_adapter.load_cathode_runs()[:3]
    horizon = InformationHorizon(ProcessStage.CALENDERING, include_decision_stage_controls=True)
    st_records = []
    for r in runs:
        v = horizon.project(r)
        st_records.append([s for s in v.source_stages if s.stage_type == ProcessStage.COATING][0])
    encoder = StageFeatureEncoder.fit(st_records)
    slot = ModalitySlotSpec("pre_calendering_ultrasound", ProcessStage.COATING, ModalityType.ULTRASOUND_SPECTRUM, "ultrasound", 29, False, True)

    t = LegalStageTransition.from_encoded_source_stage(st_records[0], encoder=encoder, modality_slots=[slot])
    assert getattr(t, "unsafe_test_only", None) is None
    assert t.provenance.get("encoder_fingerprint") == encoder.fingerprint
    assert "source_stage_fingerprint" in t.provenance
    assert t.controls.shape == (encoder.control_dim,)
    assert t.scalar_observations.shape == (encoder.observation_dim,)


# -----------------------------------------------------------------------------
# Test 9: Train-Only StageFeatureEncoder Isolation
# -----------------------------------------------------------------------------
def test_encoder_fit_isolation(ultrasonic_adapter: WarwickUltrasonicAdapter) -> None:
    """Verify StageFeatureEncoder.fit on train folds does not see test fold records."""
    runs = ultrasonic_adapter.load_anode_runs()
    repo_root = Path(__file__).resolve().parent.parent.parent
    with open(repo_root / "outputs" / "warwick_ultrasonic" / "split_manifest.json") as f:
        splits = json.load(f)["Anode"]

    horizon = InformationHorizon(ProcessStage.CALENDERING, include_decision_stage_controls=True)
    fold1_train = [r for r in runs if splits[r.cell_id] != 1]
    fold1_test = [r for r in runs if splits[r.cell_id] == 1]

    train_records = []
    for r in fold1_train:
        v = horizon.project(r)
        train_records.extend(v.source_stages)

    encoder = StageFeatureEncoder.fit(train_records)
    assert encoder.control_dim > 0
    assert encoder.observation_dim > 0
    # Confirm encoder was built strictly from train records
    test_ids = {r.cell_id for r in fold1_test}
    for rec in train_records:
        assert not any(rec.stage_id.startswith(tid) for tid in test_ids)


# -----------------------------------------------------------------------------
# Test 10: MASPO forward_batch Output Tensor Shape
# -----------------------------------------------------------------------------
def test_maspo_forward_batch_output_shape(ultrasonic_adapter: WarwickUltrasonicAdapter) -> None:
    """Verify MASPOProcessStateModel.forward_batch outputs a 1D tensor of shape (batch_size,)."""
    runs = ultrasonic_adapter.load_cathode_runs()[:4]
    horizon = InformationHorizon(ProcessStage.CALENDERING, include_decision_stage_controls=True)
    pairs = []
    for r in runs:
        v = horizon.project(r)
        s1 = [s for s in v.source_stages if s.stage_type == ProcessStage.COATING][0]
        s2 = [s for s in v.source_stages if s.stage_type == ProcessStage.CALENDERING][0]
        s2_m = StageRecord(s2.stage_id, s2.stage_type, s2.sequence_index, s2.controls, {}, [], s2.upstream_stage_id, s2.provenance)
        pairs.append((s1, s2_m))

    encoder = StageFeatureEncoder.fit([s for p in pairs for s in p])
    slot = ModalitySlotSpec("pre_calendering_ultrasound", ProcessStage.COATING, ModalityType.ULTRASOUND_SPECTRUM, "ultrasound", 29, False, True)

    model = MASPOProcessStateModel(16, encoder.control_dim, encoder.observation_dim, {"ultrasound": 29}, 16)
    model.add_final_head("target_metric")

    batch = []
    for s1, s2 in pairs:
        t1 = LegalStageTransition.from_encoded_source_stage(s1, encoder=encoder, modality_slots=[slot])
        t2 = LegalStageTransition.from_encoded_source_stage(s2, encoder=encoder, modality_slots=[slot])
        batch.append([t1, t2])

    preds = model.forward_batch(batch, target="target_metric")
    assert preds.shape == (len(runs),)
    assert torch.isfinite(preds).all()


# -----------------------------------------------------------------------------
# Test 11: Modality Fingerprint Verification and Anti-Tamper
# -----------------------------------------------------------------------------
def test_modality_slot_source_values_fingerprint(ultrasonic_adapter: WarwickUltrasonicAdapter) -> None:
    """Verify SourceBoundModalityInput validates against decoded values fingerprint."""
    runs = ultrasonic_adapter.load_cathode_runs()[:1]
    s1 = [s for s in runs[0].stages if s.stage_type == ProcessStage.COATING][0]
    mod = s1.modalities[0]
    slot = ModalitySlotSpec("pre_calendering_ultrasound", ProcessStage.COATING, ModalityType.ULTRASOUND_SPECTRUM, "ultrasound", 29, False, True)

    # Valid binding
    bound = SourceBoundModalityInput.from_observation(mod, slot)
    assert bound.model_input_name == "ultrasound"
    assert bound.tensor.shape == (29,)

    # Tampered values fingerprint must raise ValueError
    tampered_prov = replace(mod.provenance, decoded_values_fingerprint="invalid_hash_12345")
    tampered_mod = replace(mod, provenance=tampered_prov)
    with pytest.raises(ValueError, match="trusted decoded_values_fingerprint"):
        SourceBoundModalityInput.from_observation(tampered_mod, slot)


# -----------------------------------------------------------------------------
# Test 12: Source-Level Ablation Masking (No Post-Binding Tensor Mutation)
# -----------------------------------------------------------------------------
def test_source_level_ablation_masking(ultrasonic_adapter: WarwickUltrasonicAdapter) -> None:
    """Verify make_ablation_stage_records masks features at source record level without post-binding tensor mutation."""
    from scripts.run_warwick_ultrasonic_benchmark import make_ablation_stage_records

    runs = ultrasonic_adapter.load_cathode_runs()[:2]
    horizon = InformationHorizon(ProcessStage.CALENDERING, include_decision_stage_controls=True)
    v = horizon.project(runs[0])
    s1 = [s for s in v.source_stages if s.stage_type == ProcessStage.COATING][0]
    s2 = [s for s in v.source_stages if s.stage_type == ProcessStage.CALENDERING][0]

    base_encoder = StageFeatureEncoder.fit([s1, s2])
    base_cdim = max(base_encoder.control_dim, 1)
    base_odim = max(base_encoder.observation_dim, 1)
    slot = ModalitySlotSpec("pre_calendering_ultrasound", ProcessStage.COATING, ModalityType.ULTRASOUND_SPECTRUM, "ultrasound", 29, False, True)

    # 1. PROCESS_ONLY
    s1_proc, s2_proc = make_ablation_stage_records(s1, s2, "PROCESS_ONLY")
    assert len(s1_proc.modalities) == 0
    enc_proc = StageFeatureEncoder.fit([s1_proc, s2_proc], control_dim=base_cdim, observation_dim=base_odim)
    t1_proc = LegalStageTransition.from_encoded_source_stage(s1_proc, encoder=enc_proc, modality_slots=[])
    enc_proc.validate_transition(s1_proc, t1_proc)
    assert len(t1_proc.modality_inputs) == 0

    # 2. ULTRASOUND_ONLY: source records have empty controls & observations
    s1_ultra, s2_ultra = make_ablation_stage_records(s1, s2, "ULTRASOUND_ONLY")
    assert len(s1_ultra.controls) == 0
    assert len(s1_ultra.intermediate_properties) == 0
    assert len(s1_ultra.modalities) == 1
    assert len(s2_ultra.controls) == 0
    assert len(s2_ultra.intermediate_properties) == 0
    assert len(s2_ultra.modalities) == 0

    enc_ultra = StageFeatureEncoder.fit([s1_ultra, s2_ultra], control_dim=base_cdim, observation_dim=base_odim)
    t1_ultra = LegalStageTransition.from_encoded_source_stage(s1_ultra, encoder=enc_ultra, modality_slots=[slot])
    t2_ultra = LegalStageTransition.from_encoded_source_stage(s2_ultra, encoder=enc_ultra, modality_slots=[slot])

    # Validate transitions against source records WITHOUT any post-binding tensor mutation
    enc_ultra.validate_transition(s1_ultra, t1_ultra)
    enc_ultra.validate_transition(s2_ultra, t2_ultra)

    assert t1_ultra.controls.sum().item() == 0.0
    assert t1_ultra.scalar_observations.sum().item() == 0.0
    assert "ultrasound" in t1_ultra.modality_inputs
    assert t1_ultra.availability["ultrasound"] is True
    assert getattr(t1_ultra, "unsafe_test_only", None) is None

    # 3. PROCESS_PLUS_ULTRASOUND
    s1_fused, s2_fused = make_ablation_stage_records(s1, s2, "PROCESS_PLUS_ULTRASOUND")
    assert len(s1_fused.modalities) == 1
    enc_fused = StageFeatureEncoder.fit([s1_fused, s2_fused], control_dim=base_cdim, observation_dim=base_odim)
    t1_fused = LegalStageTransition.from_encoded_source_stage(s1_fused, encoder=enc_fused, modality_slots=[slot])
    enc_fused.validate_transition(s1_fused, t1_fused)
    assert "ultrasound" in t1_fused.modality_inputs
    assert t1_fused.controls.shape == (base_cdim,)


# -----------------------------------------------------------------------------
# Test 13: Process-Only Mode Empty Modality State
# -----------------------------------------------------------------------------
def test_process_only_mode_empty_modality(ultrasonic_adapter: WarwickUltrasonicAdapter) -> None:
    """Verify process-only mode yields empty modality inputs and false availability."""
    runs = ultrasonic_adapter.load_cathode_runs()[:2]
    horizon = InformationHorizon(ProcessStage.CALENDERING, include_decision_stage_controls=True)
    v = horizon.project(runs[0])
    s1 = [s for s in v.source_stages if s.stage_type == ProcessStage.COATING][0]
    encoder = StageFeatureEncoder.fit([s1])

    s1_no_mod = replace(s1, modalities=[])
    t_proc = LegalStageTransition.from_encoded_source_stage(s1_no_mod, encoder=encoder, modality_slots=[])

    assert len(t_proc.modality_inputs) == 0
    assert t_proc.controls.shape == (encoder.control_dim,)
    assert t_proc.scalar_observations.shape == (encoder.observation_dim,)
    assert getattr(t_proc, "unsafe_test_only", None) is None


# -----------------------------------------------------------------------------
# Test 14: Model Comparison Summary Schema and Dynamic Separation
# -----------------------------------------------------------------------------
def test_model_comparison_summary_schema_and_deltas() -> None:
    """Verify model_comparison_summary.json enforces strict separation between Ridge and StageAware deltas."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    summary_path = repo_root / "outputs" / "warwick_ultrasonic" / "model_comparison_summary.json"
    assert summary_path.exists(), f"Summary path not found at {summary_path}"
    with open(summary_path) as f:
        data = json.load(f)

    # Verify Ridge and StageAware branches exist separately
    assert "Ridge" in data["models"]
    assert "StageAwareProcessModel" in data["models"]

    # Verify deltas are computed strictly intra-model
    for mat in ["Cathode", "Anode"]:
        for tgt in ["thickness_after_um", "density_after_g_cm3"]:
            r_res = data["models"]["Ridge"]["results"][mat][tgt]
            sa_res = data["models"]["StageAwareProcessModel"]["results"][mat][tgt]

            assert math.isclose(r_res["fusion_delta_r2"], r_res["tabular_plus_ultrasound_r2"] - r_res["tabular_state_process_r2"], rel_tol=1e-5)
            assert math.isclose(sa_res["fusion_delta_r2"], sa_res["tabular_plus_ultrasound_r2"] - sa_res["tabular_state_process_r2"], rel_tol=1e-5)

    # Check for stale literals in text
    supported_text = data["claim_boundaries"]["supported"]
    for stale in ["0.9715", "6.25", "0.849", "0.874", "0.895", "+0.033"]:
        assert stale not in supported_text


# -----------------------------------------------------------------------------
# Test 15: Consistency Audit Fail-Closed Mutation Test
# -----------------------------------------------------------------------------
def test_consistency_audit_catches_mutation() -> None:
    """Verify that run_consistency_audit passes normally and raises AssertionError on intentional mutation."""
    # 1. Base audit must pass
    base_res = run_consistency_audit()
    assert base_res["status"] == "PASS"

    # 2. Mutate an artifact and verify audit catches it
    repo_root = Path(__file__).resolve().parent.parent.parent
    matrix_path = repo_root / "outputs" / "multi_dataset_validation" / "benchmark_matrix.csv"
    orig_content = matrix_path.read_text(encoding="utf-8")
    try:
        mutated_content = orig_content.replace("Hit@5 = 100.0%", "Hit@5 = 85.0%")
        matrix_path.write_text(mutated_content, encoding="utf-8")
        failed = False
        try:
            run_consistency_audit()
        except AssertionError:
            failed = True
        assert failed, "Consistency audit must fail on mutated benchmark matrix!"
    finally:
        matrix_path.write_text(orig_content, encoding="utf-8")


# -----------------------------------------------------------------------------
# Test 16: StageAware Deterministic Initialization & Training
# -----------------------------------------------------------------------------
def test_stage_aware_deterministic_initialization_and_training(ultrasonic_adapter: WarwickUltrasonicAdapter) -> None:
    """Verify PyTorch seed set before model construction ensures bit-exact reproducible predictions."""
    from scripts.run_warwick_ultrasonic_benchmark import train_stage_aware_model, make_ablation_stage_records

    runs = ultrasonic_adapter.load_cathode_runs()[:4]
    horizon = InformationHorizon(ProcessStage.CALENDERING, include_decision_stage_controls=True)
    slot = ModalitySlotSpec("pre_calendering_ultrasound", ProcessStage.COATING, ModalityType.ULTRASOUND_SPECTRUM, "ultrasound", 29, False, True)

    pairs = []
    for r in runs:
        v = horizon.project(r)
        s1 = [s for s in v.source_stages if s.stage_type == ProcessStage.COATING][0]
        s2_raw = [s for s in v.source_stages if s.stage_type == ProcessStage.CALENDERING][0]
        s2_cal = StageRecord(
            stage_id=s2_raw.stage_id,
            stage_type=s2_raw.stage_type,
            sequence_index=s2_raw.sequence_index,
            controls=s2_raw.controls,
            intermediate_properties={},
            modalities=[],
            upstream_stage_id=s2_raw.upstream_stage_id,
            provenance=s2_raw.provenance,
        )
        s1_m, s2_m = make_ablation_stage_records(s1, s2_cal, "PROCESS_PLUS_ULTRASOUND")
        pairs.append((s1_m, s2_m))

    encoder = StageFeatureEncoder.fit([s for p in pairs for s in p], control_dim=4, observation_dim=4)

    transitions = []
    for s1, s2 in pairs:
        t1 = LegalStageTransition.from_encoded_source_stage(s1, encoder=encoder, modality_slots=[slot])
        t2 = LegalStageTransition.from_encoded_source_stage(s2, encoder=encoder, modality_slots=[])
        transitions.append([t1, t2])

    train_trans = transitions[:3]
    test_trans = transitions[3:]
    y_tr = np.array([50.0, 52.0, 48.0])

    def _init_and_train(seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MASPOProcessStateModel(
            state_dim=16,
            control_dim=4,
            observation_dim=4,
            modality_input_dims={"ultrasound": 29},
            embedding_dim=16,
        )
        model.add_final_head("target")
        return train_stage_aware_model(
            model=model,
            train_transitions=train_trans,
            y_tr=y_tr,
            test_transitions=test_trans,
            target_col="target",
            epochs=20,
            seed=seed,
        )

    pred1 = _init_and_train(seed=123)
    pred2 = _init_and_train(seed=123)
    pred_diff_seed = _init_and_train(seed=999)

    assert np.allclose(pred1, pred2, rtol=1e-7, atol=1e-7), "Identical seed must produce identical predictions!"
    assert not np.allclose(pred1, pred_diff_seed, rtol=1e-3, atol=1e-3), "Different seeds should produce different outputs"


# -----------------------------------------------------------------------------
# Test 17: Unsafe Test-Only Detector Catches Synthetic Transition
# -----------------------------------------------------------------------------
def test_unsafe_test_only_detector(ultrasonic_adapter: WarwickUltrasonicAdapter) -> None:
    """Verify benchmark detection logic catches unsafe_test_only transitions in provenance."""
    runs = ultrasonic_adapter.load_cathode_runs()[:1]
    s1 = [s for s in runs[0].stages if s.stage_type == ProcessStage.COATING][0]
    s1_no_mod = replace(s1, modalities=[])

    # 1. Unvalidated transition constructed via test-only factory
    t_unsafe = LegalStageTransition.from_source_stage(
        s1_no_mod,
        controls=torch.zeros(4),
        scalar_observations=torch.zeros(4),
        modality_inputs={},
        modality_bindings={},
        test_only=True,
    )
    # Benchmark detection logic
    is_unsafe = bool(t_unsafe.provenance.get("unsafe_test_only", False)) or bool(getattr(t_unsafe, "unsafe_test_only", False))
    assert is_unsafe is True, "Detector must flag test_only transitions!"

    # 2. Production transition
    encoder = StageFeatureEncoder.fit([s1_no_mod], control_dim=4, observation_dim=4)
    t_safe = LegalStageTransition.from_encoded_source_stage(s1_no_mod, encoder=encoder, modality_slots=[])
    is_unsafe_safe = bool(t_safe.provenance.get("unsafe_test_only", False)) or bool(getattr(t_safe, "unsafe_test_only", False))
    assert is_unsafe_safe is False


# -----------------------------------------------------------------------------
# Test 18: Report Context Parity with Source Artifacts and Markdown
# -----------------------------------------------------------------------------
def test_report_context_parity_and_markdown() -> None:
    """Verify report_context.json matches source artifacts and markdown matches context."""
    from scripts.audit_multi_dataset_result_consistency import (
        verify_report_context_against_sources,
        verify_markdown_report_matches_context,
        verify_nmc622_report_contains_hit5,
        verify_no_unsupported_wording,
    )

    repo_root = Path(__file__).resolve().parent.parent.parent
    drak_dir = repo_root / "outputs" / "drakopoulos_rediscovery_v4"
    nmc_dir = repo_root / "outputs" / "warwick_nmc622_calendering"
    ultra_dir = repo_root / "outputs" / "warwick_ultrasonic"
    multi_dir = repo_root / "outputs" / "multi_dataset_validation"

    with open(multi_dir / "report_context.json") as f:
        report_context = json.load(f)

    drak_df = pd.read_csv(drak_dir / "policy_summary.csv")
    with open(drak_dir / "slide_summary.json") as f:
        drak_slide = json.load(f)
    nmc_df = pd.read_csv(nmc_dir / "policy_summary.csv")
    with open(nmc_dir / "slide_summary.json") as f:
        nmc_slide = json.load(f)
    ultra_df = pd.read_csv(ultra_dir / "ablation_summary.csv")
    with open(ultra_dir / "model_comparison_summary.json") as f:
        ultra_comp = json.load(f)

    verify_report_context_against_sources(
        report_context, drak_df, nmc_df, ultra_df, ultra_comp,
        drak_slide_dict=drak_slide, nmc_slide_dict=nmc_slide,
    )

    multi_md = (multi_dir / "MULTI_DATASET_VALIDATION_REPORT.md").read_text(encoding="utf-8")
    verify_markdown_report_matches_context(multi_md, report_context)

    nmc_md = (nmc_dir / "WARWICK_NMC622_PROCESS_BENCHMARK_REPORT.md").read_text(encoding="utf-8")
    verify_nmc622_report_contains_hit5(nmc_md)

    ultra_md = (ultra_dir / "WARWICK_ULTRASONIC_MULTIMODAL_REPORT.md").read_text(encoding="utf-8")
    verify_no_unsupported_wording({
        "multi": multi_md,
        "nmc": nmc_md,
        "ultra": ultra_md,
        "ultra_comp": json.dumps(ultra_comp),
    })


# -----------------------------------------------------------------------------
# Test 19: Deliberate Corruption Causes Audit and Verification Failures
# -----------------------------------------------------------------------------
def test_deliberate_corruption_fails_verification() -> None:
    """Verify deliberate corruptions in metrics, tables, and wording fail immediately."""
    from scripts.audit_multi_dataset_result_consistency import (
        verify_report_context_against_sources,
        verify_nmc622_report_contains_hit5,
        verify_no_unsupported_wording,
    )

    repo_root = Path(__file__).resolve().parent.parent.parent
    multi_dir = repo_root / "outputs" / "multi_dataset_validation"
    with open(multi_dir / "report_context.json") as f:
        ctx = json.load(f)

    drak_df = pd.read_csv(repo_root / "outputs" / "drakopoulos_rediscovery_v4" / "policy_summary.csv")
    with open(repo_root / "outputs" / "drakopoulos_rediscovery_v4" / "slide_summary.json") as f:
        drak_slide = json.load(f)
    nmc_df = pd.read_csv(repo_root / "outputs" / "warwick_nmc622_calendering" / "policy_summary.csv")
    with open(repo_root / "outputs" / "warwick_nmc622_calendering" / "slide_summary.json") as f:
        nmc_slide = json.load(f)
    ultra_df = pd.read_csv(repo_root / "outputs" / "warwick_ultrasonic" / "ablation_summary.csv")
    with open(repo_root / "outputs" / "warwick_ultrasonic" / "model_comparison_summary.json") as f:
        ultra_comp = json.load(f)

    # 1. Corrupt report_context ai_hit5
    ctx_corrupted = json.loads(json.dumps(ctx))
    ctx_corrupted["drakopoulos"]["ai_hit5"] = 0.50
    with pytest.raises(AssertionError):
        verify_report_context_against_sources(ctx_corrupted, drak_df, nmc_df, ultra_df, ultra_comp, drak_slide, nmc_slide)

    # 1b. Corrupt best_recipe
    ctx_corrupted_recipe = json.loads(json.dumps(ctx))
    ctx_corrupted_recipe["drakopoulos"]["best_recipe"] = "protocol-wrong"
    with pytest.raises(AssertionError):
        verify_report_context_against_sources(ctx_corrupted_recipe, drak_df, nmc_df, ultra_df, ultra_comp, drak_slide, nmc_slide)

    # 1c. Corrupt nmc best_condition
    ctx_corrupted_nmc = json.loads(json.dumps(ctx))
    ctx_corrupted_nmc["nmc622"]["best_condition"] = "EXP_99"
    with pytest.raises(AssertionError):
        verify_report_context_against_sources(ctx_corrupted_nmc, drak_df, nmc_df, ultra_df, ultra_comp, drak_slide, nmc_slide)

    # 2. Corrupt NMC622 report (remove Hit@5 row)
    nmc_text = (repo_root / "outputs" / "warwick_nmc622_calendering" / "WARWICK_NMC622_PROCESS_BENCHMARK_REPORT.md").read_text(encoding="utf-8")
    nmc_corrupted = "\n".join(line for line in nmc_text.splitlines() if "Hit@5" not in line)
    with pytest.raises(AssertionError):
        verify_nmc622_report_contains_hit5(nmc_corrupted)

    # 3. Insert unsupported wording
    with pytest.raises(AssertionError, match="Prohibited phrase"):
        verify_no_unsupported_wording({"test_report": "The performance drop is due to finite sample size."})

    with pytest.raises(AssertionError, match="Prohibited phrase"):
        verify_no_unsupported_wording({"test_report": "Identifies the optimal pilot-scale condition."})

    with pytest.raises(AssertionError, match="Prohibited phrase"):
        verify_no_unsupported_wording({"test_report": "Roll gap mechanically dictates thickness."})

    with pytest.raises(AssertionError, match="Prohibited phrase"):
        verify_no_unsupported_wording({"test_report": "All evaluations adhere strictly to grouped cross-validation."})

    with pytest.raises(AssertionError, match="Prohibited phrase"):
        verify_no_unsupported_wording({"test_report": "Rediscovering the source-observed optimal condition."})

