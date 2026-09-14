from types import SimpleNamespace

import pandas as pd
import pytest

from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.optimization.botorch import UnsupportedProcessOptimizationError
from src.process.optimization.process_objective import ConstraintSpec, ObjectiveSpec, ProcessOptimizationObjective
from src.process.optimization.process_space import ProcessSearchSpace


class _FirstFeasibleBackend:
    def propose(self, **kwargs):
        row = kwargs["candidate_pool"].iloc[0]
        return [SimpleNamespace(candidate_id=row["recipe_id"], predicted_mean=1.0, predicted_std=0.1, acquisition_value=1.0, backend_name="fixture", acquisition_class="fixture", metadata={})]


def _observations():
    return pd.DataFrame({"recipe_id": ["a", "b"], "temperature": [100.0, 120.0], "capacity": [1.0, 2.0]})


def test_scalar_hard_control_constraints_filter_the_candidate_pool():
    result = ProcessOptimizationCoordinator(scalar_backend=_FirstFeasibleBackend()).propose_recipes(
        _observations(), ProcessSearchSpace.from_finite_pool(_observations().drop(columns="capacity")),
        ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")], [ConstraintSpec("temperature", "lower", 110.0)]),
    )
    assert result[0].controls == {"temperature": 120.0}
    assert result[0].provenance["hard_control_constraints"] == [{"name": "temperature", "type": "lower", "threshold": 110.0, "hard": True}]


@pytest.mark.parametrize("constraint", [ConstraintSpec("porosity", "range", (0.3, 0.4)), ConstraintSpec("manufacturing_success", "feasibility", 0.9)])
def test_scalar_unsupported_constraints_fail_closed(constraint):
    with pytest.raises(UnsupportedProcessOptimizationError):
        ProcessOptimizationCoordinator(scalar_backend=_FirstFeasibleBackend()).propose_recipes(
            _observations(), ProcessSearchSpace.from_finite_pool(_observations().drop(columns="capacity")),
            ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")], [constraint]),
        )
