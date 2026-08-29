from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.search_space import (
    Constraint,
    ContinuousVariable,
    DerivedVariable,
    DiscreteVariable,
    SearchSpace,
)


def test_continuous_variable_validation_and_sampling() -> None:
    var = ContinuousVariable("temp", lower=20.0, upper=100.0)
    assert var.is_valid(50.0)
    assert var.is_valid(20.0)
    assert var.is_valid(100.0)
    assert not var.is_valid(10.0)
    assert not var.is_valid(110.0)
    assert not var.is_valid(float("nan"))

    rng = np.random.default_rng(42)
    samples = var.sample_uniform(rng, size=50)
    assert len(samples) == 50
    assert np.all(samples >= 20.0) and np.all(samples <= 100.0)

    with pytest.raises(ValueError, match="lower bound .* must be < upper bound"):
        ContinuousVariable("invalid", lower=100.0, upper=50.0)


def test_discrete_variable_validation_and_sampling() -> None:
    var = DiscreteVariable("solvent", values=("water", "ethanol", "dmso"))
    assert var.is_valid("water")
    assert var.is_valid("ethanol")
    assert not var.is_valid("methanol")

    rng = np.random.default_rng(42)
    samples = var.sample(rng, size=20)
    assert len(samples) == 20
    assert set(samples).issubset({"water", "ethanol", "dmso"})

    with pytest.raises(ValueError, match="must contain at least one value"):
        DiscreteVariable("empty", values=())


def test_derived_variable_and_constraint_evaluation() -> None:
    # Generic formulation: A in [0, 1], B in [0, 1], C = 1 - (A + B)
    var_a = ContinuousVariable("A", 0.0, 1.0)
    var_b = ContinuousVariable("B", 0.0, 1.0)
    var_c = DerivedVariable("C", compute_fn=lambda x: 1.0 - (float(x["A"]) + float(x["B"])), depends_on=("A", "B"))

    con_sum = Constraint("c_non_negative", predicate=lambda x: float(x["C"]) >= 0.0, description="C must be >= 0")

    space = SearchSpace(
        variables=[var_a, var_b],
        derived_variables=[var_c],
        constraints=[con_sum],
        name="ternary_mixture",
    )

    assert space.is_feasible({"A": 0.3, "B": 0.4})
    # Derived C = 0.3, feasible

    assert not space.is_feasible({"A": 0.6, "B": 0.7})
    # Derived C = -0.3, infeasible

    # Test feasible sampling
    df_feasible = space.sample_feasible(n=100, seed=42)
    assert len(df_feasible) == 100
    assert np.all(df_feasible["A"] >= 0.0)
    assert np.all(df_feasible["B"] >= 0.0)
    assert np.all(df_feasible["C"] >= 0.0)
    sums = df_feasible["A"] + df_feasible["B"] + df_feasible["C"]
    assert np.allclose(sums, 1.0)


def test_search_space_novelty_detection() -> None:
    var_x = ContinuousVariable("x", 0.0, 10.0)
    var_y = ContinuousVariable("y", 0.0, 10.0)
    space = SearchSpace(variables=[var_x, var_y])

    ref_grid = pd.DataFrame(
        [
            {"x": 1.0, "y": 1.0},
            {"x": 5.0, "y": 5.0},
            {"x": 9.0, "y": 9.0},
        ]
    )

    test_cands = pd.DataFrame(
        [
            {"x": 1.0, "y": 1.0},        # Exact duplicate
            {"x": 1.0001, "y": 0.9999},  # Near duplicate
            {"x": 3.0, "y": 7.0},        # Completely novel point
        ]
    )

    novelty_df = space.check_novelty(test_cands, reference_points=ref_grid, tol=1e-3)
    assert len(novelty_df) == 3
    assert novelty_df["is_novel"].iloc[0] is False or bool(novelty_df["is_novel"].iloc[0]) is False
    assert novelty_df["is_novel"].iloc[1] is False or bool(novelty_df["is_novel"].iloc[1]) is False
    assert bool(novelty_df["is_novel"].iloc[2]) is True
    assert novelty_df["min_distance"].iloc[2] > 0.1
