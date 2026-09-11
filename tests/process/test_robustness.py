from __future__ import annotations

import pytest

from src.process.robustness import evaluate_recipe_robustness


def test_robustness_reports_nominal_worst_variance_and_feasibility() -> None:
    report = evaluate_recipe_robustness(
        {"temperature": 100.0},
        [{"temperature": 99.0}, {"temperature": 100.0}, {"temperature": 101.0}],
        lambda controls: controls["temperature"],
        feasible=lambda controls: controls["temperature"] >= 100.0,
    )
    assert report.nominal_predicted_objective == 100.0
    assert report.mean_perturbed_objective == 100.0
    assert report.worst_case_objective == 99.0
    assert report.feasibility_rate == 2 / 3
    assert report.constraint_violation_rate == pytest.approx(1 / 3)
