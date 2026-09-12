from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from src.process.contracts import MeasurementValue, ParameterValue, ProvenanceRecord
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.information_horizon import HorizonView, InformationHorizon
from src.process.modalities import ModalityObservation, ModalitySlotSpec, ModalityType, SourceBoundModalityInput, source_modality_fingerprint
from src.process.maspo import MASPOProcessOptimizationCoordinator
from src.process.models import MASPOModelArtifact, MASPOProcessStateModel, SourceBackedInitialState, StageFeatureEncoder, build_legal_stage_transitions, encode_legal_multimodal_state, model_weight_fingerprint
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


def _slots(bindings: dict[str, str], *, stage: ProcessStage = ProcessStage.MIXING) -> tuple[ModalitySlotSpec, ...]:
    types = {"tabular": ModalityType.PROCESS_TABULAR, "signal": ModalityType.MACHINE_TIME_SERIES}
    return tuple(
        ModalitySlotSpec(f"{stage.value.lower()}.{model_name}", stage, types.get(model_name, ModalityType.PROCESS_TABULAR), model_name, 2)
        for model_name in bindings.values()
    )


def _artifact(
    model: MASPOProcessStateModel, encoder: StageFeatureEncoder, bindings: dict[str, str], *, stage: ProcessStage = ProcessStage.MIXING,
) -> MASPOModelArtifact:
    return MASPOModelArtifact.from_training(
        model, feature_encoder=encoder, model_version="maspo-test-v1", semantic_modality_slots=_slots(bindings, stage=stage),
    )


def _encode(horizon: HorizonView, model: MASPOProcessStateModel, *, signal: torch.Tensor | None = None, control: float = 100.0):
    if signal is not None:
        record = horizon.source_stages[-1]
        modalities = [
            replace(modality, values=signal.detach().cpu().tolist()) if modality.modality_id == "mix-signal" else modality
            for modality in record.modalities
        ]
        record = replace(record, modalities=modalities)
        horizon = replace(horizon, source_stages=(record,), modalities=tuple(modalities))
    if control != 100.0:
        record = horizon.source_stages[-1]
        controls = dict(record.controls)
        controls["speed"] = ParameterValue(control)
        record = replace(record, controls=controls)
        horizon = replace(horizon, source_stages=(record,), controls={"mixing.speed": controls["speed"]})
    encoder = StageFeatureEncoder.from_training_data(
        horizon, control_dim=model.stage_model.control_dim,
        observation_dim=model.stage_model.observation_dim - model.embedding_dim,
    )
    bindings = {"mix-tab": "tabular"}
    if any(item.modality_id == "mix-signal" for item in horizon.modalities):
        bindings["mix-signal"] = "signal"
    transitions = build_legal_stage_transitions(
        horizon,
        feature_encoder=encoder, modality_slots=_slots(bindings),
    )
    artifact = _artifact(model, encoder, bindings)
    return encode_legal_multimodal_state(
        horizon, artifact, stage_history=transitions,
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


def test_production_rejects_arbitrary_modality_tensor_injection() -> None:
    horizon = _multimodal_horizon(signal=ModalityObservation("mix-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.MIXING, values=[0.0, 1.0]))
    model = _model()
    encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
    artifact = _artifact(model, encoder, {"mix-tab": "tabular", "mix-signal": "signal"})
    with pytest.raises(ValueError, match="SourceBoundModalityInput"):
        encode_legal_multimodal_state(
            horizon, artifact,
            modality_inputs={"mix-tab": torch.tensor([100.0, 5.0]), "mix-signal": torch.tensor([4.0, 5.0])},
        )


def test_source_bound_input_changes_with_source_values_and_rejects_tampering() -> None:
    signal = ModalityObservation("mix-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.MIXING, values=[0.0, 1.0])
    horizon = _multimodal_horizon(signal=signal)
    model = _model()
    first = _encode(horizon, model, signal=torch.tensor([0.0, 1.0]))
    second = _encode(horizon, model, signal=torch.tensor([4.0, 5.0]))
    assert first.provenance_fingerprint != second.provenance_fingerprint

    encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
    slots = _slots({"mix-tab": "tabular", "mix-signal": "signal"})
    transitions = build_legal_stage_transitions(horizon, feature_encoder=encoder, modality_slots=slots)
    transitions[0].modality_inputs["signal"][0] = 999.0
    artifact = _artifact(model, encoder, {"mix-tab": "tabular", "mix-signal": "signal"})
    with pytest.raises(ValueError, match="tampered"):
        encode_legal_multimodal_state(horizon, artifact, stage_history=transitions)


def test_source_modality_fingerprint_requires_file_content_identity() -> None:
    observation = ModalityObservation(
        "mix-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.MIXING,
        source_path="future.csv", values=[0.0, 1.0],
    )
    with pytest.raises(ValueError, match="content hash"):
        source_modality_fingerprint(observation)


def test_source_modality_fingerprint_changes_with_file_content_hash() -> None:
    first = ModalityObservation(
        "mix-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.MIXING,
        source_path="signal.csv", values=[0.0, 1.0],
        provenance=ProvenanceRecord("instrument", raw_hashes={"signal.csv": "a"}),
    )
    second = replace(first, provenance=ProvenanceRecord("instrument", raw_hashes={"signal.csv": "b"}))
    assert source_modality_fingerprint(first) != source_modality_fingerprint(second)


def test_source_bound_preprocessing_fingerprint_mismatch_is_rejected() -> None:
    horizon = _multimodal_horizon()
    record = horizon.source_stages[0]
    slot = _slots({"mix-tab": "tabular"})[0]
    bound = SourceBoundModalityInput.from_observation(record.modalities[0], slot)
    bad = replace(bound, preprocessing_fingerprint="wrong-preprocessor-v1")
    encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
    with pytest.raises(ValueError, match="preprocessing fingerprint"):
        build_legal_stage_transitions(
            horizon, feature_encoder=encoder, modality_inputs={record.modalities[0].modality_id: bad}, modality_slots=(slot,),
        )


def test_test_only_source_bound_tensor_escape_hatch_remains_explicit() -> None:
    slot = _slots({"mix-tab": "tabular"})[0]
    bound = SourceBoundModalityInput.from_tensor(
        torch.tensor([999.0, 5.0]), source_modality_id="mix-tab", slot=slot, test_only=True,
    )
    assert bound.source_kind == "TEST_ONLY"
    with pytest.raises(ValueError, match="test-only"):
        SourceBoundModalityInput.from_tensor(
            torch.tensor([999.0, 5.0]), source_modality_id="mix-tab", slot=slot,
        )


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
            initial_state=torch.zeros(3), model_version="v1", model_fingerprint="model", validation_status="TEST_ONLY",
        )


def test_production_rejects_injected_stage_tensors() -> None:
    horizon = _multimodal_horizon()
    with pytest.raises(ValueError, match="test-only"):
        build_legal_stage_transitions(
            horizon, stage_inputs={"mix": (torch.tensor([999.0, 999.0]), torch.tensor([999.0]))},
            modality_inputs={"mix-tab": torch.tensor([100.0, 5.0])}, modality_slots=_slots({"mix-tab": "tabular"}),
        )


def test_stage_encoder_is_deterministic_and_source_values_are_not_injectable() -> None:
    horizon = _multimodal_horizon()
    encoder = StageFeatureEncoder.from_training_data(horizon)
    changed_record = replace(horizon.source_stages[0], controls={"speed": ParameterValue(120.0)})
    changed = replace(horizon, source_stages=(changed_record,), controls={"mixing.speed": changed_record.controls["speed"]})
    changed_encoder = StageFeatureEncoder.from_training_data(changed)
    assert encoder.fingerprint == changed_encoder.fingerprint
    assert not torch.equal(encoder.encode_controls(horizon.source_stages[0]).tensor, changed_encoder.encode_controls(changed_record).tensor)
    renamed = replace(horizon.source_stages[0], stage_id="different-source-id")
    renamed_horizon = replace(horizon, source_stage_ids=("different-source-id",), source_stages=(renamed,))
    assert encoder.fingerprint == StageFeatureEncoder.from_training_data(renamed_horizon).fingerprint


def test_stage_encoder_handles_categorical_missing_values_with_a_traceable_schema() -> None:
    run = process_run()
    record = replace(run.stages[1], controls={"protocol": ParameterValue(None)}, intermediate_properties={})
    horizon = HorizonView(ProcessStage.COATING, {"mixing.protocol": record.controls["protocol"]}, {}, tuple(), ("mix",), (record,))
    encoder = StageFeatureEncoder.from_training_data(horizon, category_vocabularies={"mixing.control.protocol": ("fast", "slow", None)})
    encoded = encoder.encode_controls(record)
    assert encoded.tensor.tolist() == [1.0, 0.0, 0.0]
    assert "mixing.control.protocol.category" in encoded.feature_names[0]
    assert encoded.fingerprint and encoder.fingerprint


def test_stage_encoder_fits_one_frozen_schema_from_multiple_training_records() -> None:
    run = process_run()
    first = replace(run.stages[1], stage_id="mix-1", controls={"speed": ParameterValue(100.0)})
    second = replace(run.stages[1], stage_id="mix-2", controls={"speed": ParameterValue(120.0)})
    encoder = StageFeatureEncoder.fit([first, second])
    assert encoder.feature_schema[0][0] == ProcessStage.MIXING.value
    assert encoder.encode_controls(first).feature_names == encoder.encode_controls(second).feature_names


def test_stage_encoder_unions_training_categories_and_rejects_unseen_inference_values() -> None:
    run = process_run()
    first = replace(run.stages[1], stage_id="mix-1", controls={"protocol": ParameterValue("fast")}, intermediate_properties={})
    second = replace(run.stages[1], stage_id="mix-2", controls={"protocol": ParameterValue("slow")}, intermediate_properties={})
    encoder = StageFeatureEncoder.fit([first, second], observation_dim=1)
    assert encoder.category_vocabularies["mixing.control.protocol"] == (("value", "fast"), ("value", "slow"))
    unseen = replace(first, controls={"protocol": ParameterValue("unseen")})
    with pytest.raises(ValueError, match="unknown to its vocabulary"):
        encoder.encode_controls(unseen)


def test_stage_encoder_freezes_a_categorical_missing_token_only_when_allowed() -> None:
    run = process_run()
    observed = replace(run.stages[1], stage_id="mix-observed", controls={"protocol": ParameterValue("fast")}, intermediate_properties={})
    missing = replace(run.stages[1], stage_id="mix-missing", controls={"protocol": ParameterValue(None)}, intermediate_properties={})
    encoder = StageFeatureEncoder.fit([observed, missing], observation_dim=1)
    assert encoder.category_vocabularies["mixing.control.protocol"] == (("missing", ""), ("value", "fast"))
    assert encoder.encode_controls(missing).tensor.tolist()[:2] == [1.0, 0.0]
    with pytest.raises(ValueError, match="unknown to its vocabulary"):
        encoder.encode_controls(replace(observed, controls={"protocol": ParameterValue("slow")}))

    disallowed = StageFeatureEncoder.fit([observed], observation_dim=1)
    with pytest.raises(ValueError, match="missing but not optional"):
        disallowed.encode_controls(missing)


def test_stage_encoder_rejects_incompatible_kinds_and_allows_declared_optional_fields() -> None:
    run = process_run()
    numeric = replace(run.stages[1], stage_id="mix-1", controls={"setting": ParameterValue(1.0)})
    categorical = replace(run.stages[1], stage_id="mix-2", controls={"setting": ParameterValue("fast")})
    with pytest.raises(ValueError, match="numeric and categorical"):
        StageFeatureEncoder.fit([numeric, categorical])
    present = replace(run.stages[1], stage_id="mix-1", controls={"speed": ParameterValue(100.0)})
    absent = replace(run.stages[1], stage_id="mix-2", controls={})
    with pytest.raises(ValueError, match="declare it optional"):
        StageFeatureEncoder.fit([present, absent])
    encoder = StageFeatureEncoder.fit([present, absent], optional_fields={"mixing.control.speed": True})
    assert encoder.encode_controls(absent).tensor.tolist()[:2] == [0.0, 0.0]


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

    first_encoder = StageFeatureEncoder.from_training_data(formulation_horizon, control_dim=2, observation_dim=2)
    second_encoder = StageFeatureEncoder.from_training_data(mixing_horizon, control_dim=2, observation_dim=2)
    first_artifact = _artifact(model, first_encoder, {"sensor": "tabular"}, stage=ProcessStage.FORMULATION)
    second_artifact = _artifact(model, second_encoder, {"sensor": "tabular"}, stage=ProcessStage.MIXING)
    first = encode_legal_multimodal_state(
        formulation_horizon, first_artifact,
    )
    second = encode_legal_multimodal_state(
        mixing_horizon, second_artifact,
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
    first_encoder = StageFeatureEncoder.from_training_data(first_horizon, control_dim=2, observation_dim=2)
    second_encoder = StageFeatureEncoder.from_training_data(second_horizon, control_dim=2, observation_dim=2)
    first_artifact = _artifact(model, first_encoder, {"mix-tab": "tabular"})
    second_artifact = _artifact(model, second_encoder, {"mix-tab": "tabular"})
    first = encode_legal_multimodal_state(
        first_horizon, first_artifact,
    )
    second = encode_legal_multimodal_state(
        second_horizon, second_artifact,
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

    metadata = {
        "representation_kind": state.representation_kind,
        "model_artifact_fingerprint": state.provenance["model_artifact_fingerprint"],
        "model_fingerprint": state.provenance["model_fingerprint"],
        "encoder_fingerprint": state.provenance["semantic_fingerprint_inputs"]["encoder_fingerprint"],
        "model_version": state.provenance["model_version"],
        "latent_feature_schema": list(state.feature_names),
        "dataset_fingerprint": "dataset-sha256",
    }
    plan = coordinator.optimize_remaining_process(
        current_state=run, current_stage=ProcessStage.COATING,
        remaining_control_spaces={ProcessStage.DRYING: space}, observations=observations,
        objective=ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
        optimization_state=state,
        historical_latent_metadata=metadata,
    )

    call = backend.calls[0]
    assert call["feature_columns"] == [*state.feature_names, "temperature"]
    assert plan.next_control.controls == {"temperature": 100}
    assert plan.next_control.source_recipe_id == "dry-low"
    assert plan.next_control.candidate_instance_id
    assert plan.next_control.provenance["context_provenance_fingerprint"] == state.provenance_fingerprint
    with pytest.raises(ValueError, match="artifact provenance manifest"):
        coordinator._contextual_inputs(
            horizon, space, observations, ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
            optimization_state=state,
        )
    with pytest.raises(ValueError, match="model_fingerprint"):
        coordinator._contextual_inputs(
            horizon, space, observations, ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
            optimization_state=state, historical_latent_metadata={**metadata, "model_fingerprint": "other"},
        )
    with pytest.raises(ValueError, match="model_artifact_fingerprint"):
        coordinator._contextual_inputs(
            horizon, space, observations, ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
            optimization_state=state, historical_latent_metadata={**metadata, "model_artifact_fingerprint": "other"},
        )


def test_multimodal_state_rejects_a_future_bound_modality() -> None:
    run = process_run()
    future = ModalityObservation("dry-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.DRYING, values=[1.0, 2.0])
    drying = replace(run.stages[3], modalities=[future])
    horizon = HorizonView(ProcessStage.COATING, {"drying.temperature": drying.controls["temperature"]}, {"drying.dry_thickness": drying.intermediate_properties["dry_thickness"]}, (future,), ("dry",), (drying,))
    with pytest.raises(ValueError, match="source stage"):
        encoder = StageFeatureEncoder.from_training_data(horizon)
        artifact = MASPOModelArtifact.from_training(
            MASPOProcessStateModel(3, 2, 2, {"signal": 2}, embedding_dim=4), feature_encoder=encoder,
            model_version="maspo-test-v1", semantic_modality_slots=_slots({"dry-signal": "signal"}, stage=ProcessStage.DRYING),
        )
        encode_legal_multimodal_state(
            horizon, artifact,
        )


def test_production_maspo_rejects_unvalidated_multimodal_state() -> None:
    run = process_run()
    model = _model()
    horizon = InformationHorizon(ProcessStage.COATING).project(run)
    encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
    artifact = _artifact(model, encoder, {"mix-tab": "tabular"})
    state = encode_legal_multimodal_state(
        horizon, artifact,
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
    encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
    transitions = build_legal_stage_transitions(
        horizon, feature_encoder=encoder, modality_slots=_slots({"mix-tab": "tabular"}),
    )
    artifact = _artifact(_model(), encoder, {"mix-tab": "tabular"})
    wrong_id = replace(
        transitions[0], source_stage_id="dry",
        provenance={**transitions[0].provenance, "source_stage_id": "dry"},
    )
    with pytest.raises(ValueError, match="legal source stages"):
        encode_legal_multimodal_state(
            horizon, artifact, stage_history=(wrong_id, transitions[1]),
        )
    altered = replace(transitions[0], provenance={**transitions[0].provenance, "source_stage_fingerprint": "changed"})
    with pytest.raises(ValueError, match="source stage fingerprint"):
        encode_legal_multimodal_state(
            horizon, artifact, stage_history=(altered, transitions[1]),
        )
    future_provenance = replace(
        transitions[1],
        provenance={**transitions[1].provenance, "modality_bindings": {"mix-tab": "tabular", "dry-signal": "signal"}},
    )
    with pytest.raises(ValueError, match="modality provenance"):
        encode_legal_multimodal_state(
            horizon, artifact, stage_history=(transitions[0], future_provenance),
        )


def test_model_artifact_freezes_encoder_and_owns_initial_state() -> None:
    horizon = _multimodal_horizon()
    model = _model()
    encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
    artifact = _artifact(model, encoder, {"mix-tab": "tabular"})
    state = encode_legal_multimodal_state(horizon, artifact)
    assert state.provenance["model_artifact_fingerprint"] == artifact.artifact_fingerprint
    assert state.provenance["audit_provenance"]["initial_state_fingerprint"] == artifact.initial_state_fingerprint

    changed_record = replace(horizon.source_stages[0], controls={"flow": ParameterValue(100.0)})
    changed_horizon = replace(horizon, source_stages=(changed_record,), controls={"mixing.flow": changed_record.controls["flow"]})
    changed_encoder = StageFeatureEncoder.from_training_data(changed_horizon, control_dim=2, observation_dim=2)
    assert changed_encoder.fingerprint != artifact.encoder_fingerprint
    with pytest.raises(ValueError, match="encoder fingerprint"):
        encode_legal_multimodal_state(
            horizon, artifact, feature_encoder=changed_encoder,
        )

    changed_model = _model()
    with torch.no_grad():
        changed_model.initial_state[0] = 1.0
    changed_artifact = _artifact(changed_model, encoder, {"mix-tab": "tabular"})
    assert changed_artifact.artifact_fingerprint != artifact.artifact_fingerprint


def test_model_weight_fingerprint_is_state_dict_deterministic_and_ignores_train_mode() -> None:
    model = _model()
    fingerprint = model_weight_fingerprint(model)
    model.train()
    assert model_weight_fingerprint(model) == fingerprint
    model.eval()
    assert model_weight_fingerprint(model) == fingerprint
    reloaded = _model()
    reloaded.load_state_dict(model.state_dict())
    assert model_weight_fingerprint(reloaded) == fingerprint
    with torch.no_grad():
        reloaded.stage_model.transition[0].weight[0, 0] += 1.0
    assert model_weight_fingerprint(reloaded) != fingerprint


def test_model_artifact_rejects_transition_weight_mutation_before_inference() -> None:
    horizon = _multimodal_horizon()
    model = _model()
    encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
    artifact = _artifact(model, encoder, {"mix-tab": "tabular"})
    encode_legal_multimodal_state(horizon, artifact)
    with torch.no_grad():
        model.stage_model.transition[0].weight[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="integrity"):
        encode_legal_multimodal_state(horizon, artifact)


def test_model_artifact_rejects_modality_encoder_and_initial_state_mutation() -> None:
    horizon = _multimodal_horizon()
    for mutate in ("modality", "initial"):
        model = _model()
        encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
        artifact = _artifact(model, encoder, {"mix-tab": "tabular"})
        if mutate == "modality":
            with torch.no_grad():
                model.modality_encoders["tabular"][0].weight[0, 0] += 1.0
        else:
            with torch.no_grad():
                model.initial_state[0] += 1.0
        with pytest.raises(RuntimeError, match="integrity"):
            encode_legal_multimodal_state(horizon, artifact)


def test_model_artifact_requires_computed_expected_weight_fingerprint() -> None:
    horizon = _multimodal_horizon()
    model = _model()
    encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
    with pytest.raises(ValueError, match="expected_model_fingerprint"):
        MASPOModelArtifact.from_training(
            model, feature_encoder=encoder, model_version="maspo-test-v1",
            expected_model_fingerprint="caller-defined", semantic_modality_slots=_slots({"mix-tab": "tabular"}),
        )


def test_semantic_modality_slot_accepts_a_new_source_instance_id() -> None:
    original = _multimodal_horizon(signal=ModalityObservation("mix-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.MIXING, values=[0.0, 1.0]))
    model = _model()
    encoder = StageFeatureEncoder.from_training_data(original, control_dim=2, observation_dim=2)
    artifact = _artifact(model, encoder, {"mix-tab": "tabular", "mix-signal": "signal"})
    new_signal = replace(original.source_stages[0].modalities[1], modality_id="mix-signal-run-847")
    new_record = replace(original.source_stages[0], modalities=[original.source_stages[0].modalities[0], new_signal])
    new_horizon = replace(original, source_stages=(new_record,), modalities=tuple(new_record.modalities))
    state = encode_legal_multimodal_state(
        new_horizon, artifact,
    )
    assert state.provenance["modality_bindings"]["mix-signal-run-847"] == "signal"
    assert artifact.artifact_fingerprint == _artifact(_model(), encoder, {"different-run-id": "tabular", "other-run-id": "signal"}).artifact_fingerprint


def test_semantic_modality_slot_rejects_wrong_type_stage_and_dimension() -> None:
    horizon = _multimodal_horizon()
    model = _model()
    encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
    artifact = _artifact(model, encoder, {"mix-tab": "tabular"})
    for change, message in (({"modality_type": ModalityType.SEM_IMAGE}, "no semantic slot"),):
        bad = replace(horizon.source_stages[0].modalities[0], **change)
        bad_record = replace(horizon.source_stages[0], modalities=[bad])
        bad_horizon = replace(horizon, source_stages=(bad_record,), modalities=(bad,))
        with pytest.raises(ValueError, match=message):
            encode_legal_multimodal_state(bad_horizon, artifact)
    wrong_stage_artifact = _artifact(model, encoder, {"mix-tab": "tabular"}, stage=ProcessStage.FORMULATION)
    assert wrong_stage_artifact.artifact_fingerprint != artifact.artifact_fingerprint
    with pytest.raises(ValueError, match="no semantic slot"):
        encode_legal_multimodal_state(horizon, wrong_stage_artifact)
    with pytest.raises(ValueError, match="SourceBoundModalityInput"):
        encode_legal_multimodal_state(horizon, artifact, modality_inputs={"mix-tab": torch.tensor([100.0, 5.0, 7.0])})


def test_semantic_modality_slot_missing_and_ambiguous_policies_fail_closed() -> None:
    missing = ModalityObservation("mix-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.MIXING, missing_reason="offline")
    horizon = _multimodal_horizon(signal=missing)
    model = _model()
    encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
    artifact = _artifact(model, encoder, {"mix-tab": "tabular", "mix-signal": "signal"})
    with patch.object(model, "fuse_observations", wraps=model.fuse_observations) as fused:
        encode_legal_multimodal_state(horizon, artifact)
    tokens, availability = fused.call_args.args
    assert "signal" not in tokens and availability["signal"] is False

    empty_record = replace(horizon.source_stages[0], modalities=[])
    empty_horizon = replace(horizon, source_stages=(empty_record,), modalities=())
    required_encoder = StageFeatureEncoder.from_training_data(empty_horizon, control_dim=2, observation_dim=2)
    required_slot = ModalitySlotSpec("mixing.tabular", ProcessStage.MIXING, ModalityType.PROCESS_TABULAR, "tabular", 2, required=True, allowed_missing=False)
    required_artifact = MASPOModelArtifact.from_training(model, feature_encoder=required_encoder, model_version="v1", semantic_modality_slots=(required_slot,))
    with pytest.raises(ValueError, match="required semantic modality slot"):
        encode_legal_multimodal_state(empty_horizon, required_artifact, modality_inputs={})

    ambiguous_artifact = MASPOModelArtifact.from_training(
        _model(), feature_encoder=encoder, model_version="v1",
        semantic_modality_slots=(
            ModalitySlotSpec("mixing.tabular-a", ProcessStage.MIXING, ModalityType.PROCESS_TABULAR, "tabular", 2),
            ModalitySlotSpec("mixing.tabular-b", ProcessStage.MIXING, ModalityType.PROCESS_TABULAR, "signal", 2),
        ),
    )
    with pytest.raises(ValueError, match="ambiguous semantic slots"):
        encode_legal_multimodal_state(horizon, ambiguous_artifact)


def test_required_and_allowed_missing_are_independent() -> None:
    run = process_run()
    record = replace(run.stages[1], modalities=[])
    horizon = HorizonView(
        ProcessStage.COATING,
        {"mixing.speed": record.controls["speed"]},
        {"mixing.viscosity": record.intermediate_properties["viscosity"]},
        (), ("mix",), (record,),
    )
    encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
    optional_slot = ModalitySlotSpec("mixing.optional", ProcessStage.MIXING, ModalityType.PROCESS_TABULAR, "tabular", 2, required=False, allowed_missing=False)
    artifact = MASPOModelArtifact.from_training(
        _model(), feature_encoder=encoder, model_version="v1", semantic_modality_slots=(optional_slot,),
    )
    encode_legal_multimodal_state(horizon, artifact)

    missing = ModalityObservation("mix-tab", ModalityType.PROCESS_TABULAR, ProcessStage.MIXING, missing_reason="offline")
    missing_record = replace(record, modalities=[missing])
    missing_horizon = replace(horizon, source_stages=(missing_record,), modalities=(missing,))
    strict_artifact = MASPOModelArtifact.from_training(
        _model(), feature_encoder=StageFeatureEncoder.from_training_data(missing_horizon, control_dim=2, observation_dim=2),
        model_version="v1", semantic_modality_slots=(ModalitySlotSpec("mixing.tabular", ProcessStage.MIXING, ModalityType.PROCESS_TABULAR, "tabular", 2, required=True, allowed_missing=False),),
    )
    with pytest.raises(ValueError, match="explicitly missing"):
        encode_legal_multimodal_state(missing_horizon, strict_artifact)


def test_production_multimodal_inference_requires_artifact_and_unknown_categories_fail_closed() -> None:
    horizon = _multimodal_horizon()
    encoder = StageFeatureEncoder.from_training_data(horizon, control_dim=2, observation_dim=2)
    with pytest.raises(ValueError, match="MASPOModelArtifact"):
        encode_legal_multimodal_state(
            horizon, _model(), feature_encoder=encoder, modality_inputs={"mix-tab": torch.tensor([100.0, 5.0])},
            modality_bindings={"mix-tab": "tabular"}, validation_status="TRAINED_UNVALIDATED",
        )

    record = replace(horizon.source_stages[0], controls={"protocol": ParameterValue("fast")}, intermediate_properties={})
    categorical = replace(horizon, source_stages=(record,), controls={"mixing.protocol": record.controls["protocol"]}, intermediate_properties={})
    categorical_encoder = StageFeatureEncoder.from_training_data(
        categorical, category_vocabularies={"mixing.control.protocol": ("fast", "slow")}, control_dim=3, observation_dim=2,
    )
    unknown = replace(record, controls={"protocol": ParameterValue("unseen")})
    with pytest.raises(ValueError, match="unknown to its vocabulary"):
        categorical_encoder.encode_controls(unknown)

    with pytest.raises(ValueError, match="test-only"):
        SourceBackedInitialState.from_tensor(
            torch.zeros(3), source_stage_ids=("mix",), model_fingerprint="model", encoder_fingerprint=encoder.fingerprint,
        )
