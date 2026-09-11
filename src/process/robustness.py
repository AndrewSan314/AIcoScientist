from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class RobustnessReport:
    nominal_predicted_objective: float
    mean_perturbed_objective: float
    worst_case_objective: float
    objective_variance: float
    feasibility_rate: float | None
    constraint_violation_rate: float | None
    perturbation_count: int


def evaluate_recipe_robustness(
    nominal_controls: Mapping[str, float],
    perturbations: list[Mapping[str, float]],
    predict: Callable[[Mapping[str, float]], float],
    *,
    feasible: Callable[[Mapping[str, float]], bool] | None = None,
    maximize: bool = True,
) -> RobustnessReport:
    """Evaluate caller-supplied bounded perturbations without changing BO acquisition."""
    if not perturbations:
        raise ValueError("source-justified perturbations are required")
    values = [float(predict(controls)) for controls in perturbations]
    nominal = float(predict(nominal_controls))
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    feasibility_rate = None if feasible is None else sum(bool(feasible(controls)) for controls in perturbations) / len(perturbations)
    return RobustnessReport(
        nominal_predicted_objective=nominal,
        mean_perturbed_objective=mean,
        worst_case_objective=min(values) if maximize else max(values),
        objective_variance=variance,
        feasibility_rate=feasibility_rate,
        constraint_violation_rate=None if feasibility_rate is None else 1.0 - feasibility_rate,
        perturbation_count=len(values),
    )
