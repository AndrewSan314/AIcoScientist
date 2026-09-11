from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from src.process.contracts import MeasurementValue
from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.information_horizon import HorizonView, InformationHorizon
from src.process.modalities import ModalityObservation, ModalityType
from src.process.maspo import MASPOProcessOptimizationCoordinator
from src.process.models import MASPOProcessStateModel, encode_legal_multimodal_state
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
    tabular = ModalityObservation(
        "mix-tab", ModalityType.PROCESS_TABULAR, ProcessStage.MIXING, values=[100.0, 5.0],
    )
    modalities = (tabular,) if signal is None else (tabular, signal)
    return HorizonView(ProcessStage.COATING, {}, {}, modalities, ("mix",))


def _model() -> MASPOProcessStateModel:
    torch.manual_seed(7)
    return MASPOProcessStateModel(3, 2, 1, {"tabular": 2, "signal": 2}, embedding_dim=4)


def _encode(horizon: HorizonView, model: MASPOProcessStateModel, *, signal: torch.Tensor | None = None):
    inputs = {"mix-tab": torch.tensor([100.0, 5.0])}
    bindings = {"mix-tab": "tabular"}
    if any(item.modality_id == "mix-signal" for item in horizon.modalities):
        bindings["mix-signal"] = "signal"
    if signal is not None:
        inputs["mix-signal"] = signal
    return encode_legal_multimodal_state(
        horizon, model, modality_inputs=inputs, modality_bindings=bindings,
        model_version="maspo-test-v1", model_fingerprint="model-sha256",
    )


def test_legal_modality_changes_reach_the_multimodal_optimizer_state() -> None:
    signal = ModalityObservation("mix-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.MIXING, values=[0.0, 1.0])
    horizon = _multimodal_horizon(signal=signal)
    model = _model()

    first = _encode(horizon, model, signal=torch.tensor([0.0, 1.0]))
    second = _encode(horizon, model, signal=torch.tensor([4.0, 5.0]))

    assert first.representation_kind == "multimodal_latent"
    assert first.provenance["validation_status"] == "IMPLEMENTED_NOT_VALIDATED"
    assert first.provenance_fingerprint != second.provenance_fingerprint
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
    first = _encode(InformationHorizon(ProcessStage.COATING).project(run), model)
    second = _encode(InformationHorizon(ProcessStage.COATING).project(future_run), model)

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
    future = ModalityObservation("dry-signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.DRYING, values=[1.0, 2.0])
    with pytest.raises(ValueError, match="unavailable"):
        _encode(_multimodal_horizon(signal=future), _model())
