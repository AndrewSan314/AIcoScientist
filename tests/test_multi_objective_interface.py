from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.multi_objective import (
    MultiObjectiveSpec,
    Objective,
    PlaceholderqNEHVI,
    compute_pareto_front,
    is_pareto_dominated,
)


def test_pareto_dominance_2d() -> None:
    # Maximize capacity, minimize charging_time
    objectives = [
        Objective(name="capacity", direction="maximize"),
        Objective(name="charging_time", direction="minimize"),
    ]
    directions = [obj.direction for obj in objectives]

    all_vals = np.array([
        [100.0, 10.0],  # Candidate A
        [120.0, 10.0],  # Candidate B (higher capacity, same time -> dominates A)
        [90.0, 5.0],    # Candidate C (lower capacity, faster time -> non-dominated)
    ])

    # Candidate A is dominated by Candidate B
    assert is_pareto_dominated(all_vals[0], all_vals[[1, 2]], directions) is True
    # Candidate B is non-dominated
    assert is_pareto_dominated(all_vals[1], all_vals[[0, 2]], directions) is False
    # Candidate C is non-dominated
    assert is_pareto_dominated(all_vals[2], all_vals[[0, 1]], directions) is False


def test_compute_pareto_front_dataframe() -> None:
    df = pd.DataFrame([
        {"id": "c1", "capacity": 100.0, "retention": 90.0, "cost": 50.0},
        {"id": "c2", "capacity": 110.0, "retention": 95.0, "cost": 40.0},  # Dominates c1 in all
        {"id": "c3", "capacity": 120.0, "retention": 85.0, "cost": 60.0},  # Tradeoff
    ])

    objectives = [
        Objective(name="capacity", direction="maximize"),
        Objective(name="retention", direction="maximize"),
        Objective(name="cost", direction="minimize"),
    ]

    pareto = compute_pareto_front(df, objectives)
    assert len(pareto) == 2
    assert set(pareto["id"]) == {"c2", "c3"}


def test_multi_objective_spec_and_qnehvi_contract() -> None:
    spec = MultiObjectiveSpec(
        objectives=[
            Objective(name="capacity", direction="maximize"),
            Objective(name="lifetime", direction="maximize"),
            Objective(name="degradation_rate", direction="minimize"),
        ]
    )
    assert spec.objective_names == ["capacity", "lifetime", "degradation_rate"]

    acq = PlaceholderqNEHVI(n_mc_samples=100)
    cand_means = np.array([[100.0, 800.0, 0.05], [120.0, 1000.0, 0.02]])
    cand_stds = np.array([[5.0, 50.0, 0.01], [10.0, 100.0, 0.01]])
    obs_means = np.array([[90.0, 700.0, 0.06]])

    scores = acq.evaluate(cand_means, cand_stds, obs_means, spec)
    assert len(scores) == 2
    assert np.all(np.isfinite(scores))
