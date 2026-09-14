from types import SimpleNamespace

import pandas as pd

from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.optimization.process_objective import ObjectiveSpec, ProcessOptimizationObjective
from src.process.optimization.process_space import ProcessSearchSpace


class _RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def propose(self, **kwargs: object):
        self.calls.append(kwargs)
        recipe_id = kwargs["candidate_pool"].iloc[0]["recipe_id"]
        return [SimpleNamespace(candidate_id=recipe_id, predicted_mean=1.0, predicted_std=0.1, acquisition_value=1.0, backend_name="test", acquisition_class="test", metadata={})]


def test_scalar_strategy_is_forwarded_without_changing_default() -> None:
    observations = pd.DataFrame({"recipe_id": ["a"], "speed": [1.0], "capacity": [1.0]})
    backend = _RecordingBackend()
    coordinator = ProcessOptimizationCoordinator(scalar_backend=backend)
    space = ProcessSearchSpace.from_finite_pool(pd.DataFrame({"recipe_id": ["a", "b"], "speed": [1.0, 2.0]}))
    objective = ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")])
    coordinator.propose_recipes(observations, space, objective, strategy="random")
    assert backend.calls[-1]["strategy"] == "random"
    coordinator.propose_recipes(observations, space, objective)
    assert backend.calls[-1]["strategy"] == "noisy_expected_improvement"
