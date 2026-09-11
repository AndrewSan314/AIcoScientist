from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.contracts import MeasurementValue, ParameterValue
from src.process.information_horizon import HorizonView
from src.process.maspo import MASPOProcessOptimizationCoordinator
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.optimization.process_space import ProcessSearchSpace
from src.process.stages import ProcessStage
from .conftest import process_run


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def propose(self, **kwargs: object):
        self.calls.append(kwargs)
        pool = kwargs["candidate_pool"]
        context = float(pool["context.control.formulation.solids"].iloc[0])
        source_id = "dry-high" if context >= 0.8 else "dry-low"
        candidate_id = pool.loc[pool["source_recipe_id"] == source_id, "candidate_instance_id"].iloc[0]
        return [SimpleNamespace(candidate_id=candidate_id, predicted_mean=155.0, predicted_std=1.0, acquisition_value=1.0, backend_name="fake", acquisition_class="test", metadata={})]


class FirstCandidateBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def propose(self, **kwargs: object):
        self.calls.append(kwargs)
        row = kwargs["candidate_pool"].iloc[0]
        return [SimpleNamespace(candidate_id=row["candidate_instance_id"], predicted_mean=155.0, predicted_std=1.0, acquisition_value=1.0, backend_name="fake", acquisition_class="test", metadata={})]


def _space() -> ProcessSearchSpace:
    return ProcessSearchSpace.from_finite_pool(pd.DataFrame([
        {"recipe_id": "dry-low", "temperature": 100},
        {"recipe_id": "dry-high", "temperature": 120},
    ]))


def _observations() -> pd.DataFrame:
    return pd.DataFrame([
        {"recipe_id": "dry-low", "temperature": 100, "context.control.formulation.solids": 0.6, "context.control.mixing.speed": 100.0, "context.intermediate.mixing.viscosity": 5.0, "capacity": 150.0, "future.measurement": 999.0},
        {"recipe_id": "dry-high", "temperature": 120, "context.control.formulation.solids": 0.8, "context.control.mixing.speed": 100.0, "context.intermediate.mixing.viscosity": 5.0, "capacity": 155.0, "future.measurement": 1000.0},
    ])


def _horizon(solids: float) -> HorizonView:
    return HorizonView(
        ProcessStage.COATING,
        {"formulation.solids": ParameterValue(solids), "mixing.speed": ParameterValue(100.0)},
        {"mixing.viscosity": MeasurementValue(5.0)},
        (),
        ("form", "mix"),
    )


def _categorical_horizon(mode: object) -> HorizonView:
    return HorizonView(ProcessStage.COATING, {"formulation.mode": ParameterValue(mode)}, {}, (), ("form",))


def test_maspo_conditions_next_stage_on_legal_context_and_preserves_source_controls() -> None:
    backend = RecordingBackend()
    coordinator = MASPOProcessOptimizationCoordinator(ProcessOptimizationCoordinator(scalar_backend=backend))
    plan = coordinator.optimize_remaining_process(
        current_state=process_run(),
        current_stage=ProcessStage.COATING,
        remaining_control_spaces={ProcessStage.DRYING: _space(), ProcessStage.CALENDERING: _space()},
        observations=_observations(),
        objective=ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
    )

    call = backend.calls[0]
    assert plan.next_stage == ProcessStage.DRYING
    assert plan.next_control.source_recipe_id == "dry-low"
    assert plan.next_control.controls == {"temperature": 100}
    assert call["feature_columns"] == ["context.control.formulation.solids", "context.control.mixing.speed", "context.intermediate.mixing.viscosity", "temperature"]
    assert call["observations"].columns.tolist() == ["candidate_instance_id", "source_recipe_id", "control_action_id", "context_provenance_fingerprint", "context.control.formulation.solids", "context.control.mixing.speed", "context.intermediate.mixing.viscosity", "temperature", "capacity"]
    assert call["observations"]["candidate_instance_id"].nunique() == 2
    assert "future.measurement" not in call["candidate_pool"]
    assert call["candidate_pool"]["context.control.formulation.solids"].tolist() == [0.6, 0.6]
    assert plan.legal_state.source_stage_ids == ("form", "mix")
    assert "capacity" not in plan.legal_state.intermediate_properties


def test_maspo_state_change_changes_action_with_same_historical_pool() -> None:
    backend = RecordingBackend()
    coordinator = MASPOProcessOptimizationCoordinator(ProcessOptimizationCoordinator(scalar_backend=backend))
    objective = ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")])
    low = coordinator.optimize_remaining_process(
        current_state=_horizon(0.6), current_stage=ProcessStage.COATING,
        remaining_control_spaces={ProcessStage.DRYING: _space()}, observations=_observations(), objective=objective,
    )
    high = coordinator.optimize_remaining_process(
        current_state=_horizon(0.8), current_stage=ProcessStage.COATING,
        remaining_control_spaces={ProcessStage.DRYING: _space()}, observations=_observations(), objective=objective,
    )
    assert low.next_control.source_recipe_id == "dry-low"
    assert high.next_control.source_recipe_id == "dry-high"
    assert backend.calls[1]["candidate_pool"]["context.control.formulation.solids"].tolist() == [0.8, 0.8]


def test_contextual_candidate_availability_reuses_only_same_context_action() -> None:
    objective = ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")])
    state, clean, current_space = MASPOProcessOptimizationCoordinator._contextual_inputs(
        _horizon(0.8), _space(), _observations(), objective,
    )
    current_ids = set(current_space.candidates["candidate_instance_id"])
    historical_ids = set(clean["candidate_instance_id"])
    assert len(current_ids - historical_ids) == 1
    assert len(current_ids & historical_ids) == 1

    same_context = _observations().iloc[[0]].copy()
    same_context["context.control.formulation.solids"] = 0.8
    _, observed_clean, observed_space = MASPOProcessOptimizationCoordinator._contextual_inputs(
        _horizon(0.8), _space(), same_context, objective,
    )
    assert observed_clean.loc[observed_clean["source_recipe_id"] == "dry-low", "candidate_instance_id"].iloc[0] in set(observed_space.candidates["candidate_instance_id"])
    assert len(observed_space.candidates) == 2
    assert state.feature_values["context.control.formulation.solids"] == 0.8


def test_maspo_fails_closed_when_historical_context_is_missing() -> None:
    observations = _observations().drop(columns="context.intermediate.mixing.viscosity")
    coordinator = MASPOProcessOptimizationCoordinator(ProcessOptimizationCoordinator(scalar_backend=RecordingBackend()))
    with pytest.raises(ValueError, match="historical context"):
        coordinator.optimize_remaining_process(
            current_state=_horizon(0.6), current_stage=ProcessStage.COATING,
            remaining_control_spaces={ProcessStage.DRYING: _space()}, observations=observations,
            objective=ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
        )


def test_maspo_rejects_a_horizon_stage_mismatch() -> None:
    coordinator = MASPOProcessOptimizationCoordinator(ProcessOptimizationCoordinator(scalar_backend=RecordingBackend()))
    with pytest.raises(ValueError, match="decision_stage"):
        coordinator.optimize_remaining_process(
            current_state=_horizon(0.6), current_stage=ProcessStage.DRYING,
            remaining_control_spaces={ProcessStage.CALENDERING: _space()}, observations=_observations(),
            objective=ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
        )


def test_maspo_one_hot_encodes_legal_categorical_context_from_source_vocabulary() -> None:
    backend = FirstCandidateBackend()
    observations = pd.DataFrame([
        {"recipe_id": "dry-fast", "temperature": 100, "context.control.formulation.mode": "fast", "capacity": 150.0},
        {"recipe_id": "dry-slow", "temperature": 120, "context.control.formulation.mode": "slow", "capacity": 155.0},
    ])
    plan = MASPOProcessOptimizationCoordinator(ProcessOptimizationCoordinator(scalar_backend=backend)).optimize_remaining_process(
        current_state=_categorical_horizon("fast"), current_stage=ProcessStage.COATING,
        remaining_control_spaces={ProcessStage.DRYING: ProcessSearchSpace.from_finite_pool(observations[["recipe_id", "temperature"]])},
        observations=observations, objective=ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
    )

    assert plan.optimization_state.feature_names == (
        "context.control.formulation.mode.category.0", "context.control.formulation.mode.category.1",
    )
    assert backend.calls[0]["feature_columns"] == [*plan.optimization_state.feature_names, "temperature"]
    assert plan.optimization_state.provenance["category_vocabularies"]


def test_maspo_rejects_unknown_categorical_context_instead_of_dropping_it() -> None:
    observations = pd.DataFrame([
        {"recipe_id": "dry-fast", "temperature": 100, "context.control.formulation.mode": "fast", "capacity": 150.0},
        {"recipe_id": "dry-slow", "temperature": 120, "context.control.formulation.mode": "slow", "capacity": 155.0},
    ])
    with pytest.raises(ValueError, match="unknown"):
        MASPOProcessOptimizationCoordinator(ProcessOptimizationCoordinator(scalar_backend=FirstCandidateBackend())).optimize_remaining_process(
            current_state=_categorical_horizon("turbo"), current_stage=ProcessStage.COATING,
            remaining_control_spaces={ProcessStage.DRYING: ProcessSearchSpace.from_finite_pool(observations[["recipe_id", "temperature"]])},
            observations=observations, objective=ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
        )


def test_maspo_represents_missing_categorical_context_explicitly() -> None:
    backend = FirstCandidateBackend()
    observations = pd.DataFrame([
        {"recipe_id": "dry-missing", "temperature": 100, "context.control.formulation.mode": None, "capacity": 150.0},
        {"recipe_id": "dry-fast", "temperature": 120, "context.control.formulation.mode": "fast", "capacity": 155.0},
    ])
    plan = MASPOProcessOptimizationCoordinator(ProcessOptimizationCoordinator(scalar_backend=backend)).optimize_remaining_process(
        current_state=_categorical_horizon(None), current_stage=ProcessStage.COATING,
        remaining_control_spaces={ProcessStage.DRYING: ProcessSearchSpace.from_finite_pool(observations[["recipe_id", "temperature"]])},
        observations=observations, objective=ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
    )

    assert len(plan.optimization_state.feature_names) == 2
    assert plan.optimization_state.feature_values[plan.optimization_state.feature_names[0]] == 1.0
