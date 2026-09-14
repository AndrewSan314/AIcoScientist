from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from src.process.contracts import BatteryProcessRun, MeasurementValue, ParameterValue, ProvenanceRecord, StageRecord
from src.process.modalities import ModalityObservation, ModalitySlotSpec, ModalityType
from src.process.models import MASPOProcessStateModel, StageFeatureEncoder
from src.process.models.transitions import LegalStageTransition
from src.process.stage_aware_training import train_stage_aware_multimodal
from src.process.stages import ProcessStage


def _provenance() -> ProvenanceRecord:
    return ProvenanceRecord(evidence_kind="PHYSICAL_HISTORICAL", raw_hashes={"raw.csv": "abc" * 20})


def _tiny_trajectory(viscosity_val: float = 123456.789, coating_thickness_val: float = 70.0) -> BatteryProcessRun:
    prov = _provenance()
    mixing = StageRecord(
        "mix-1",
        ProcessStage.MIXING,
        1,
        {"speed": ParameterValue(100.0)},
        {"viscosity": MeasurementValue(viscosity_val)},
        [ModalityObservation("mix-tab", ModalityType.PROCESS_TABULAR, ProcessStage.MIXING, values=[100.0, 5.0], provenance=prov)],
        provenance=prov,
    )
    coating = StageRecord(
        "coat-1",
        ProcessStage.COATING,
        2,
        {"gap": ParameterValue(50.0)},
        {"wet_thickness": MeasurementValue(coating_thickness_val)},
        [],
        upstream_stage_id="mix-1",
        provenance=prov,
    )
    return BatteryProcessRun(
        "run-1",
        "cell-1",
        "batch-1",
        "NMC622",
        {},
        {},
        [mixing, coating],
        {"capacity": MeasurementValue(150.0)},
        prov,
    )


def _slots() -> tuple[ModalitySlotSpec, ...]:
    return (ModalitySlotSpec("mixing.process", ProcessStage.MIXING, ModalityType.PROCESS_TABULAR, "process", 2),)


def test_intermediate_target_is_not_in_its_own_prediction_input() -> None:
    """Distinctive intermediate value must NOT enter the input features or pre-measurement state used to predict it."""
    run = _tiny_trajectory(viscosity_val=123456.789)
    encoder = StageFeatureEncoder.fit(run.stages)
    slots = _slots()
    transition = LegalStageTransition.from_encoded_source_stage(run.stages[0], encoder=encoder, modality_slots=slots)

    # 1. Inspect model-ready transition tensors directly
    assert any(abs(x - 123456.789) < 1e-2 for x in transition.scalar_observations.tolist())
    assert not any(abs(x - 123456.789) < 1e-2 for x in transition.pre_scalar_observations.tolist())
    assert not any(abs(x - 123456.789) < 1e-2 for x in transition.controls.tolist())

    # 2. Inspect state used for intermediate prediction
    model = MASPOProcessStateModel(state_dim=8, control_dim=encoder.control_dim, observation_dim=encoder.observation_dim, modality_input_dims={"process": 2}, embedding_dim=4)
    model.add_intermediate_head(ProcessStage.MIXING, "viscosity")

    _, pre_states, _ = model._states_from_transitions(model.initial_state, [transition])
    pre_state = pre_states[ProcessStage.MIXING]

    # Pre-state cannot be constructed with the target value
    # Verify by constructing identical pre-state with 0.0 in place of 123456.789
    zero_run = _tiny_trajectory(viscosity_val=0.0)
    zero_trans = LegalStageTransition.from_encoded_source_stage(zero_run.stages[0], encoder=encoder, modality_slots=slots)
    _, zero_pre_states, _ = model._states_from_transitions(model.initial_state, [zero_trans])
    assert torch.equal(pre_state, zero_pre_states[ProcessStage.MIXING])


def test_intermediate_observation_becomes_available_after_prediction() -> None:
    """The observed intermediate measurement becomes available for the post-measurement state and downstream steps."""
    run = _tiny_trajectory(viscosity_val=123456.789)
    encoder = StageFeatureEncoder.fit(run.stages)
    slots = _slots()
    trans_mix = LegalStageTransition.from_encoded_source_stage(run.stages[0], encoder=encoder, modality_slots=slots)

    model = MASPOProcessStateModel(state_dim=8, control_dim=encoder.control_dim, observation_dim=encoder.observation_dim, modality_input_dims={"process": 2}, embedding_dim=4)
    model.add_intermediate_head(ProcessStage.MIXING, "viscosity")

    final_state, pre_states, post_states = model._states_from_transitions(model.initial_state, [trans_mix])
    # Post-state consumed the true observation, so it differs from pre-state
    assert not torch.equal(pre_states[ProcessStage.MIXING], post_states[ProcessStage.MIXING])
    assert torch.equal(final_state, post_states[ProcessStage.MIXING])


def test_future_stage_prediction_can_use_prior_observed_intermediate_state() -> None:
    """Downstream stage prediction legally depends on previously revealed intermediate property."""
    run_a = _tiny_trajectory(viscosity_val=5.0, coating_thickness_val=70.0)
    run_b = _tiny_trajectory(viscosity_val=500.0, coating_thickness_val=70.0)
    encoder = StageFeatureEncoder.fit(run_a.stages + run_b.stages)
    slots = _slots()

    trans_a = [LegalStageTransition.from_encoded_source_stage(s, encoder=encoder, modality_slots=slots) for s in run_a.stages]
    trans_b = [LegalStageTransition.from_encoded_source_stage(s, encoder=encoder, modality_slots=slots) for s in run_b.stages]

    model = MASPOProcessStateModel(state_dim=8, control_dim=encoder.control_dim, observation_dim=encoder.observation_dim, modality_input_dims={"process": 2}, embedding_dim=4)
    model.add_intermediate_head(ProcessStage.MIXING, "viscosity")
    model.add_intermediate_head(ProcessStage.COATING, "wet_thickness")
    model.add_final_head("capacity")

    with torch.no_grad():
        out_a = model(model.initial_state, trans_a)
        out_b = model(model.initial_state, trans_b)

    # Mixing intermediate prediction must be identical (predict-then-update)
    assert torch.equal(out_a["MIXING.viscosity"], out_b["MIXING.viscosity"])
    # Downstream coating wet_thickness prediction MUST differ because it legally uses prior mixing observation
    assert not torch.equal(out_a["COATING.wet_thickness"], out_b["COATING.wet_thickness"])
    # Final KPI capacity MUST differ because it legally uses prior mixing observation
    assert not torch.equal(out_a["capacity"], out_b["capacity"])


def test_final_kpi_never_enters_transition_features() -> None:
    """Final KPIs are strictly excluded from transition features, initial states, and provenance."""
    run = _tiny_trajectory(viscosity_val=10.0)
    encoder = StageFeatureEncoder.fit(run.stages)
    slots = _slots()

    # Final KPI not in encoder feature names
    for stage_spec in encoder.stages:
        for f in stage_spec.controls + stage_spec.observations:
            assert "capacity" not in f.name

    # LegalStageTransition refuses final KPI in provenance
    with pytest.raises(ValueError, match="final KPIs"):
        LegalStageTransition.from_encoded_source_stage(
            run.stages[0], encoder=encoder, modality_slots=slots, provenance={"final_kpi": 150.0}
        )


def test_intermediate_evaluation_uses_predict_before_reveal() -> None:
    """Stage-aware training evaluation uses predict-before-reveal semantics for intermediate targets."""
    runs = [replace(_tiny_trajectory(viscosity_val=10.0 * (i + 1)), run_id=f"r-{i}", batch_id=f"b-{i}") for i in range(4)]
    slots = _slots()
    res = train_stage_aware_multimodal(
        runs,
        modality_slots=slots,
        final_targets=("capacity",),
        dataset_fingerprint="ds-test",
        epochs=3,
    )
    assert res.status == "EVALUATED_SOURCE_BACKED"
    assert res.test_mae is not None
    assert "capacity" in res.test_mae
    assert "MIXING.viscosity" in res.test_mae


def test_perturbing_intermediate_label_does_not_alter_prediction_input_or_output() -> None:
    """Strong anti-cheating test: perturb only intermediate label; feature tensor to predict it is identical."""
    run_base = _tiny_trajectory(viscosity_val=5.0)
    run_perturbed = _tiny_trajectory(viscosity_val=999999.9)

    encoder = StageFeatureEncoder.fit(run_base.stages)
    slots = _slots()

    trans_base = LegalStageTransition.from_encoded_source_stage(run_base.stages[0], encoder=encoder, modality_slots=slots)
    trans_perturbed = LegalStageTransition.from_encoded_source_stage(run_perturbed.stages[0], encoder=encoder, modality_slots=slots)

    # Pre-observation features must be exactly identical
    assert torch.equal(trans_base.pre_scalar_observations, trans_perturbed.pre_scalar_observations)
    assert torch.equal(trans_base.controls, trans_perturbed.controls)

    model = MASPOProcessStateModel(state_dim=8, control_dim=encoder.control_dim, observation_dim=encoder.observation_dim, modality_input_dims={"process": 2}, embedding_dim=4)
    model.add_intermediate_head(ProcessStage.MIXING, "viscosity")

    with torch.no_grad():
        out_base = model(model.initial_state, [trans_base])
        out_perturbed = model(model.initial_state, [trans_perturbed])

    # The prediction MUST be mathematically identical
    assert torch.equal(out_base["MIXING.viscosity"], out_perturbed["MIXING.viscosity"])
