from __future__ import annotations

import torch
import pandas as pd
import pytest

from src.process.optimization.botorch import OfficialMultiObjectiveBoTorch, UnsupportedProcessOptimizationError
from src.process.optimization.process_objective import ConstraintSpec, ObjectiveSpec, ProcessOptimizationObjective
from src.process.optimization.process_space import ProcessSearchSpace


def test_minimize_constraint_is_transformed_for_maximizing_botorch_space() -> None:
    objective = ProcessOptimizationObjective([ObjectiveSpec("impedance", "minimize"), ObjectiveSpec("capacity", "maximize")], [ConstraintSpec("impedance", "upper", 5)])
    functions = OfficialMultiObjectiveBoTorch._outcome_constraints(objective.constraints, ["impedance", "capacity"], objective.objectives)
    assert float(functions[0](torch.tensor([[-4.0, 1.0]]))) <= 0


def test_multiobjective_rejects_unknown_constraint_instead_of_ignoring_it() -> None:
    with pytest.raises(UnsupportedProcessOptimizationError, match="must name"):
        OfficialMultiObjectiveBoTorch._validate_constraint_semantics([ConstraintSpec("unknown", "lower", 1)], ["speed"], ["capacity"])


def test_multiobjective_scales_against_the_full_finite_recipe_pool() -> None:
    space = ProcessSearchSpace(pd.DataFrame({"recipe_id": ["a", "b", "c"], "speed": [10.0, 20.0, 30.0]}))
    observations, encoded_space = space.official_botorch_view(
        pd.DataFrame({"recipe_id": ["a", "b"], "speed": [10.0, 20.0], "capacity": [1.0, 2.0]})
    )
    history, candidates = OfficialMultiObjectiveBoTorch._scaled_inputs(
        observations, encoded_space.candidates.loc[encoded_space.candidates["recipe_id"] == "c"], encoded_space,
    )
    assert history.tolist() == [[0.0], [0.5]] and candidates.tolist() == [[1.0]]
