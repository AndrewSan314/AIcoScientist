from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from src.process.contracts import MeasurementValue, ParameterValue
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.information_horizon import HorizonView, InformationHorizon
from src.process.modalities import ModalityObservation, ModalityType
from src.process.maspo import MASPOProcessOptimizationCoordinator
from src.process.models import MASPOProcessStateModel, SourceBackedInitialState, StageFeatureEncoder, build_legal_stage_transitions, encode_legal_multimodal_state
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.optimization.process_space import ProcessSearchSpace
from src.process.stages import ProcessStage
from .conftest import process_run


class FirstCandidateBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def propose(self, **kwargs: object):
        self.calls.append(kwargs)
        row = kwargs["candidate_pool"].iloc[0]
        return [SimpleNamespace(
            candidate_id=row["candidate_instance_id"], predicted_mean=155.0, predicted_std=1.0,
            acquisition_value=1.0, backend_name="fake", acquisition_class="test", metadata={},
        )]


def _multimodal_horizon(*, signal: ModalityObservation | None = None) -> HorizonView:
    run = process_run()
    tabular = run.stages[1].modalities[0]
    modalities = (tabular,) if signal is None else (tabular, signal)
    mixing = replace(run.stages[1], modalities=list(modalities))
    return HorizonView(
        ProcessStage.COATING, {"mixing.speed": mixing.controls["speed"]}, {"mixing.viscosity": mixing.intermediate_properties["viscosity"]},
        modalities, ("mix",), (mixing,),
    )


def _model() -> MASPOProcessStateModel:
    torch.manual_seed(7)
    return MASPOProcessStateModel(3, 2, 2, {"tabular": 2, "signal": 2}, embedding_dim=4)


def _encode(horizon: HorizonView, model: MASPOProcessStateModel, *, signal: torch.Tensor | None = None, control: float = 100.0):
    if control != 100.0:
        record = horizon.source_stages[-1]
        controls = dict(record.controls)
        controls["speed"] = ParameterValue(control)
        record = replace(record, controls=controls)
        horizon = replace(horizon, source_stages=(record,), controls={"mixing.speed": controls["speed"]})
    encoder = StageFeatureEncoder.from_horizon(horizon)
    inputs = {"mix-tab": torch.tensor([100.0, 5.0])}
    bindings = {"mix-tab": "tabular"}
    if any(item.modality_id == "mix-signal" for item in horizon.modalities):
        bindings["mix-signal"] = "signal"
    if signal is not None:
        inputs["mix-signal"] = signal
    transitions = build_legal_stage_transitions(
        horizon,
        feature_encoder=encoder,
        modality_inputs=inputs, modality_bindings=bindings,
    )
    return encode_legal_multimodal_state(
        horizon, model, feature_encoder=encoder, stage_history=transitions,
        initial_state=SourceBackedInitialState.from_tensor(torch.zeros(3), source_stage_ids=horizon.source_stage_ids, model_fingerprint="model-sha256", encoder_fingerprint=encoder.fingerprint),
        model_version="maspo-test-v1", model_fingerprint="model-sha256",
    )


def test_legal_modality_changes_reach_the_multimodal_optimizer_state() -> None:
    signal = ModalityObservation("mix-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.MIXING, values=[0.0, 1.0])
    horizon = _multimodal_horizon(signal=signal)
    model = _model()

    first = _encode(horizon, model, signal=torch.tensor([0.0, 1.0]))
    second = _encode(horizon, model, signal=torch.tensor([4.0, 5.0]))

    assert first.representation_kind == "multimodal_stage_state"
    assert first.provenance["validation_status"] == "TRAINED_UNVALIDATED"
    assert first.provenance_fingerprint != second.provenance_fingerprint
    assert first.feature_values != second.feature_values


def test_canonical_multimodal_path_invokes_stage_aware_transition() -> None:
    model = _model()
    with patch.object(model.stage_model, "transition_stage", wraps=model.stage_model.transition_stage) as transition:
        state = _encode(_multimodal_horizon(), model)

    assert transition.call_count == 1
    assert state.representation_kind == "multimodal_stage_state"


def test_raw_anonymous_stage_tuple_is_not_a_public_history_input() -> None:
    with pytest.raises(TypeError, match="LegalStageTransition"):
        encode_legal_multimodal_state(
            _multimodal_horizon(), _model(), stage_history=[("mix", torch.zeros(2), torch.zeros(1))],
            initial_state=torch.zeros(3), model_version="v1", model_fingerprint="model",
        )


def test_production_rejects_injected_stage_tensors() -> None:
    horizon = _multimodal_horizon()
    with pytest.raises(ValueError, match="test-only"):
        build_legal_stage_transitions(
            horizon, stage_inputs={"mix": (torch.tensor([999.0, 999.0]), torch.tensor([999.0]))},
            modality_inputs={"mix-tab": torch.tensor([100.0, 5.0])}, modality_bindings={"mix-tab": "tabular"},
        )


def test_stage_encoder_is_deterministic_and_source_values_are_not_injectable() -> None:
    horizon = _multimodal_horizon()
    encoder = StageFeatureEncoder.from_horizon(horizon)
    changed_record = replace(horizon.source_stages[0], controls={"speed": ParameterValue(120.0)})
    changed = replace(horizon, source_stages=(changed_record,), controls={"mixing.speed": changed_record.controls["speed"]})
    changed_encoder = StageFeatureEncoder.from_horizon(changed)
    assert encoder.fingerprint == changed_encoder.fingerprint
    assert not torch.equal(encoder.encode_controls(horizon.source_stages[0]).tensor, changed_encoder.encode_controls(changed_record).tensor)
    renamed = replace(horizon.source_stages[0], stage_id="different-source-id")
    renamed_horizon = replace(horizon, source_stage_ids=("different-source-id",), source_stages=(renamed,))
    assert encoder.fingerprint == StageFeatureEncoder.from_horizon(renamed_horizon).fingerprint


def test_stage_encoder_handles_categorical_missing_values_with_a_traceable_schema() -> None:
    run = process_run()
    record = replace(run.stages[1], controls={"protocol": ParameterValue(None)}, intermediate_properties={})
    horizon = HorizonView(ProcessStage.COATING, {"mixing.protocol": record.controls["protocol"]}, {}, tuple(), ("mix",), (record,))
    encoder = StageFeatureEncoder.from_horizon(horizon, category_vocabularies={"mixing.control.protocol": ("fast", "slow", None)})
    encoded = encoder.encode_controls(record)
    assert encoded.tensor.tolist() == [1.0, 0.0, 0.0]
    assert "mixing.control.protocol.category" in encoded.feature_names[0]
    assert encoded.fingerprint and encoder.fingerprint


def test_same_modalities_and_controls_at_different_stages_change_state() -> None:
    values = [100.0, 5.0]
    formulation_modality = ModalityObservation("sensor", ModalityType.PROCESS_TABULAR, ProcessStage.FORMULATION, values=values)
    mixing_modality = ModalityObservation("sensor", ModalityType.PROCESS_TABULAR, ProcessStage.MIXING, values=values)
    run = process_run()
    formulation = replace(run.stages[0], modalities=[formulation_modality])
    mixing = replace(run.stages[1], modalities=[mixing_modality])
    formulation_horizon = HorizonView(ProcessStage.COATING, {"formulation.solids": formulation.controls["solids"]}, {}, (formulation_modality,), ("form",), (formulation,))
    mixing_horizon = HorizonView(ProcessStage.COATING, {"mixing.speed": mixing.controls["speed"]}, {"mixing.viscosity": mixing.intermediate_properties["viscosity"]}, (mixing_modality,), ("mix",), (mixing,))
    model = _model()

    first_encoder = StageFeatureEncoder.from_horizon(formulation_horizon, observation_dim=2)
    second_encoder = StageFeatureEncoder.from_horizon(mixing_horizon, observation_dim=2)
    first = encode_legal_multimodal_state(
        formulation_horizon, model, feature_encoder=first_encoder,
        modality_inputs={"sensor": torch.tensor(values)}, modality_bindings={"sensor": "tabular"},
        initial_state=SourceBackedInitialState.from_tensor(torch.zeros(3), source_stage_ids=("form",), model_fingerprint="model", encoder_fingerprint=first_encoder.fingerprint),
        model_version="v1", model_fingerprint="model",
    )
    second = encode_legal_multimodal_state(
        mixing_horizon, model, feature_encoder=second_encoder,
        modality_inputs={"sensor": torch.tensor(values)}, modality_bindings={"sensor": "tabular"},
        initial_state=SourceBackedInitialState.from_tensor(torch.zeros(3), source_stage_ids=("mix",), model_fingerprint="model", encoder_fingerprint=second_encoder.fingerprint),
        model_version="v1", model_fingerprint="model",
    )

    assert first.feature_values != second.feature_values


def test_same_modality_and_stage_with_different_control_changes_state() -> None:
    model = _model()
    first = _encode(_multimodal_horizon(), model, control=100.0)
    second = _encode(_multimodal_horizon(), model, control=120.0)

    assert first.feature_values != second.feature_values


def test_missing_modality_is_masked_without_a_zero_token() -> None:
    missing = ModalityObservation(
        "mix-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.MIXING, missing_reason="sensor offline",
    )
    model = _model()
    with patch.object(model, "fuse_observations", wraps=model.fuse_observations) as fused:
        state = _encode(_multimodal_horizon(signal=missing), model)

    tokens, available = fused.call_args.args
    assert set(tokens) == {"tabular"}
    assert available == {"tabular": True, "signal": False}
    assert state.provenance["modality_availability"] == {"mix-tab": True, "mix-signal": False}


def test_future_modality_and_final_kpi_cannot_change_a_coating_state() -> None:
    run = process_run()
    future_modality = ModalityObservation(
        "dry-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.DRYING, values=[99.0, 100.0],
    )
    future_run = replace(
        run,
        stages=[*run.stages[:3], replace(run.stages[3], modalities=[future_modality]), *run.stages[4:]],
        final_kpis={"capacity": MeasurementValue(999.0), "impedance": MeasurementValue(0.1)},
    )
    model = _model()
    first_horizon = InformationHorizon(ProcessStage.COATING).project(run)
    second_horizon = InformationHorizon(ProcessStage.COATING).project(future_run)
    first_encoder = StageFeatureEncoder.from_horizon(first_horizon)
    second_encoder = StageFeatureEncoder.from_horizon(second_horizon)
    first = encode_legal_multimodal_state(
        first_horizon, model, feature_encoder=first_encoder, modality_inputs={"mix-tab": torch.tensor([100.0, 5.0])},
        modality_bindings={"mix-tab": "tabular"}, initial_state=SourceBackedInitialState.from_tensor(torch.zeros(3), source_stage_ids=first_horizon.source_stage_ids, model_fingerprint="model-sha256", encoder_fingerprint=first_encoder.fingerprint),
        model_version="maspo-test-v1", model_fingerprint="model-sha256",
    )
    second = encode_legal_multimodal_state(
        second_horizon, model, feature_encoder=second_encoder, modality_inputs={"mix-tab": torch.tensor([100.0, 5.0])},
        modality_bindings={"mix-tab": "tabular"}, initial_state=SourceBackedInitialState.from_tensor(torch.zeros(3), source_stage_ids=second_horizon.source_stage_ids, model_fingerprint="model-sha256", encoder_fingerprint=second_encoder.fingerprint),
        model_version="maspo-test-v1", model_fingerprint="model-sha256",
    )

    assert first.feature_values == second.feature_values
    assert "dry-signal" not in second.provenance["modality_ids"]


def test_multimodal_state_features_reach_backend_while_controls_stay_controls() -> None:
    run = process_run()
    horizon = InformationHorizon(ProcessStage.COATING).project(run)
    model = _model()
    state = _encode(horizon, model)
    backend = FirstCandidateBackend()
    coordinator = MASPOProcessOptimizationCoordinator(
        optimizer=ProcessOptimizationCoordinator(scalar_backend=backend),
        require_validated_multimodal_state=False,
    )
    space = ProcessSearchSpace.from_finite_pool(pd.DataFrame([
        {"recipe_id": "dry-low", "temperature": 100}, {"recipe_id": "dry-high", "temperature": 120},
    ]))
    observations = pd.DataFrame([
        {"recipe_id": "dry-low", "temperature": 100, **state.feature_values, "capacity": 150.0},
        {"recipe_id": "dry-high", "temperature": 120, **state.feature_values, "capacity": 155.0},
    ])

    plan = coordinator.optimize_remaining_process(
        current_state=run, current_stage=ProcessStage.COATING,
        remaining_control_spaces={ProcessStage.DRYING: space}, observations=observations,
        objective=ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
        optimization_state=state,
    )

    call = backend.calls[0]
    assert call["feature_columns"] == [*state.feature_names, "temperature"]
    assert plan.next_control.controls == {"temperature": 100}
    assert plan.next_control.source_recipe_id == "dry-low"
    assert plan.next_control.candidate_instance_id
    assert plan.next_control.provenance["context_provenance_fingerprint"] == state.provenance_fingerprint


def test_multimodal_state_rejects_a_future_bound_modality() -> None:
    run = process_run()
    future = ModalityObservation("dry-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.DRYING, values=[1.0, 2.0])
    drying = replace(run.stages[3], modalities=[future])
    horizon = HorizonView(ProcessStage.COATING, {"drying.temperature": drying.controls["temperature"]}, {"drying.dry_thickness": drying.intermediate_properties["dry_thickness"]}, (future,), ("dry",), (drying,))
    with pytest.raises(ValueError, match="source stage"):
        encoder = StageFeatureEncoder.from_horizon(horizon)
        encode_legal_multimodal_state(
            horizon, _model(), feature_encoder=encoder,
            modality_inputs={"dry-signal": torch.tensor([1.0, 2.0])}, modality_bindings={"dry-signal": "signal"},
            initial_state=SourceBackedInitialState.from_tensor(torch.zeros(3), source_stage_ids=horizon.source_stage_ids, model_fingerprint="model-sha256", encoder_fingerprint=encoder.fingerprint),
            model_version="maspo-test-v1", model_fingerprint="model-sha256",
        )


def test_production_maspo_rejects_unvalidated_multimodal_state() -> None:
    run = process_run()
    model = _model()
    horizon = InformationHorizon(ProcessStage.COATING).project(run)
    encoder = StageFeatureEncoder.from_horizon(horizon)
    state = encode_legal_multimodal_state(
        horizon, model, feature_encoder=encoder, modality_inputs={"mix-tab": torch.tensor([100.0, 5.0])}, modality_bindings={"mix-tab": "tabular"},
        initial_state=SourceBackedInitialState.from_tensor(torch.zeros(3), source_stage_ids=horizon.source_stage_ids, model_fingerprint="model", encoder_fingerprint=encoder.fingerprint),
        model_version="v1", model_fingerprint="model",
    )
    with pytest.raises(ValueError, match="SOURCE_BACKED_VALIDATED"):
        MASPOProcessOptimizationCoordinator().optimize_remaining_process(
            current_state=run, current_stage=ProcessStage.COATING,
            remaining_control_spaces={ProcessStage.DRYING: ProcessSearchSpace.from_finite_pool(pd.DataFrame([
                {"recipe_id": "dry-low", "temperature": 100},
            ]))},
            observations=pd.DataFrame([{
                "recipe_id": "dry-low", "temperature": 100, **state.feature_values, "capacity": 150.0,
            }]), objective=ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
            optimization_state=state,
        )


def test_legal_history_rejects_future_stage_id_and_changed_source_fingerprint() -> None:
    run = process_run()
    horizon = InformationHorizon(ProcessStage.COATING).project(run)
    encoder = StageFeatureEncoder.from_horizon(horizon)
    transitions = build_legal_stage_transitions(
        horizon, feature_encoder=encoder, modality_inputs={"mix-tab": torch.tensor([100.0, 5.0])}, modality_bindings={"mix-tab": "tabular"},
    )
    wrong_id = replace(
        transitions[0], source_stage_id="dry",
        provenance={**transitions[0].provenance, "source_stage_id": "dry"},
    )
    with pytest.raises(ValueError, match="legal source stages"):
        encode_legal_multimodal_state(
            horizon, _model(), stage_history=(wrong_id, transitions[1]),
            feature_encoder=encoder, initial_state=SourceBackedInitialState.from_tensor(torch.zeros(3), source_stage_ids=horizon.source_stage_ids, model_fingerprint="model", encoder_fingerprint=encoder.fingerprint), model_version="v1", model_fingerprint="model",
        )
    altered = replace(transitions[0], provenance={**transitions[0].provenance, "source_stage_fingerprint": "changed"})
    with pytest.raises(ValueError, match="source stage fingerprint"):
        encode_legal_multimodal_state(
            horizon, _model(), stage_history=(altered, transitions[1]),
            feature_encoder=encoder, initial_state=SourceBackedInitialState.from_tensor(torch.zeros(3), source_stage_ids=horizon.source_stage_ids, model_fingerprint="model", encoder_fingerprint=encoder.fingerprint), model_version="v1", model_fingerprint="model",
        )
    future_provenance = replace(
        transitions[1],
        provenance={**transitions[1].provenance, "modality_bindings": {"mix-tab": "tabular", "dry-signal": "signal"}},
    )
    with pytest.raises(ValueError, match="modality provenance"):
        encode_legal_multimodal_state(
            horizon, _model(), stage_history=(transitions[0], future_provenance),
            feature_encoder=encoder, initial_state=SourceBackedInitialState.from_tensor(torch.zeros(3), source_stage_ids=horizon.source_stage_ids, model_fingerprint="model", encoder_fingerprint=encoder.fingerprint), model_version="v1", model_fingerprint="model",
        )
