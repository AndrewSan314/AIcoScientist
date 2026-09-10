from __future__ import annotations

import torch

from src.process.optimization.botorch import OfficialMultiObjectiveBoTorch
from src.process.optimization.process_objective import ConstraintSpec, ObjectiveSpec, ProcessOptimizationObjective


def test_minimize_constraint_is_transformed_for_maximizing_botorch_space() -> None:
    objective = ProcessOptimizationObjective([ObjectiveSpec("impedance", "minimize"), ObjectiveSpec("capacity", "maximize")], [ConstraintSpec("impedance", "upper", 5)])
    functions = OfficialMultiObjectiveBoTorch._outcome_constraints(objective.constraints, ["impedance", "capacity"], objective.objectives)
    assert float(functions[0](torch.tensor([[-4.0, 1.0]]))) <= 0
