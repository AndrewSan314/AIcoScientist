from __future__ import annotations

import pandas as pd
from types import SimpleNamespace

from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.maspo import MASPOProcessOptimizationCoordinator
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.optimization.process_space import ProcessSearchSpace
from src.process.stages import ProcessStage
from .conftest import process_run


class FakeBackend:
    def propose(self, **kwargs):
        return [SimpleNamespace(candidate_id="mix-b", predicted_mean=155.0, predicted_std=1.0, acquisition_value=1.0, backend_name="fake", acquisition_class="test", metadata={})]


def test_maspo_commits_only_the_next_stage_action() -> None:
    space = ProcessSearchSpace.from_finite_pool(pd.DataFrame([{"recipe_id": "mix-a", "speed": 100}, {"recipe_id": "mix-b", "speed": 120}]))
    coordinator = MASPOProcessOptimizationCoordinator(ProcessOptimizationCoordinator(scalar_backend=FakeBackend()))
    plan = coordinator.optimize_remaining_process(
        current_state=process_run(),
        current_stage=ProcessStage.FORMULATION,
        remaining_control_spaces={ProcessStage.MIXING: space, ProcessStage.DRYING: space},
        observations=pd.DataFrame([{"recipe_id": "old", "speed": 100, "capacity": 150}]),
        objective=ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")]),
    )
    assert plan.next_stage == ProcessStage.MIXING
    assert plan.next_control.stage == ProcessStage.MIXING
    assert plan.downstream_stages == (ProcessStage.DRYING,)
    assert "capacity" not in plan.legal_state.intermediate_properties
