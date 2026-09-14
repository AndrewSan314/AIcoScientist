import pandas as pd
import pytest

from src.process.coordinator import ProcessOptimizationCoordinator
from src.process.manufacturability import ManufacturabilityModel
from src.process.optimization.botorch import UnsupportedProcessOptimizationError
from src.process.optimization.process_objective import ConstraintSpec, ObjectiveSpec, ProcessOptimizationObjective
from src.process.optimization.process_space import ProcessSearchSpace
from .test_scalar_constraints import _FirstFeasibleBackend, _observations


def _model():
    return ManufacturabilityModel(seed=7).fit(pd.DataFrame({"temperature": [90, 95, 100, 105, 115, 120, 125, 130]}), ["mixing_failure", "mixing_failure", "out_of_spec", "out_of_spec", "manufacturing_success", "manufacturing_success", "manufacturing_success", "manufacturing_success"])


def test_feasibility_probability_is_learned_and_filters_candidates():
    result = ProcessOptimizationCoordinator(scalar_backend=_FirstFeasibleBackend(), manufacturability_model=_model()).propose_recipes(
        _observations(), ProcessSearchSpace.from_finite_pool(_observations().drop(columns="capacity")),
        ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")], [ConstraintSpec("manufacturing_success", "feasibility", 0.8)]),
    )
    assert result[0].controls == {"temperature": 120.0}
    assert result[0].feasibility_probability is not None and result[0].feasibility_probability >= 0.8
    assert result[0].provenance["feasibility_model_fingerprint"]


def test_feasibility_without_a_matching_model_fails_closed():
    with pytest.raises(UnsupportedProcessOptimizationError):
        ProcessOptimizationCoordinator(scalar_backend=_FirstFeasibleBackend()).propose_recipes(
            _observations(), ProcessSearchSpace.from_finite_pool(_observations().drop(columns="capacity")),
            ProcessOptimizationObjective([ObjectiveSpec("capacity", "maximize")], [ConstraintSpec("manufacturing_success", "feasibility", 0.8)]),
        )
