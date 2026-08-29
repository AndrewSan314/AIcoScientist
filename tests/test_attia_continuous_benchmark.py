from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.datasets.attia import AttiaAdapter
from src.evaluation.attia_continuous_benchmark import (
    DISCRETE_GRID_OPTIMUM_TRUE,
    run_attia_continuous_benchmark,
    run_single_attia_continuous_trajectory,
)


def test_attia_continuous_search_space_properties() -> None:
    adapter = AttiaAdapter()
    space = adapter.continuous_search_space()

    assert space.name == "attia_continuous_fast_charging"
    assert space.free_variable_names == ["C1", "C2", "C3"]
    assert space.all_variable_names == ["C1", "C2", "C3", "C4"]

    # Sample feasible points
    feasible_df = space.sample_feasible(n=200, seed=42)
    assert len(feasible_df) == 200

    assert np.all(feasible_df["C1"] >= 3.6) and np.all(feasible_df["C1"] <= 8.0)
    assert np.all(feasible_df["C2"] >= 3.6) and np.all(feasible_df["C2"] <= 7.0)
    assert np.all(feasible_df["C3"] >= 3.6) and np.all(feasible_df["C3"] <= 5.6)
    assert np.all(feasible_df["C4"] >= 0.1) and np.all(feasible_df["C4"] <= 4.81)

    # Verify 10-minute total charging time
    t_total = 0.2 / feasible_df["C1"] + 0.2 / feasible_df["C2"] + 0.2 / feasible_df["C3"] + 0.2 / feasible_df["C4"]
    assert np.allclose(t_total, 1.0 / 6.0, atol=1e-5)


def test_attia_continuous_candidates_exist_outside_224_grid() -> None:
    adapter = AttiaAdapter()
    space = adapter.continuous_search_space()
    discrete_pool = adapter.load_candidate_pool()

    # Draw continuous samples
    continuous_samples = space.sample_feasible(n=100, seed=123)
    novelty_df = space.check_novelty(
        continuous_samples,
        reference_points=discrete_pool,
        feature_cols=["C1", "C2", "C3", "C4"],
        tol=1e-3,
    )

    # The vast majority of random continuous points should be novel off-grid points
    assert novelty_df["is_novel"].mean() > 0.95


def test_attia_continuous_trajectory_execution(tmp_path: Path) -> None:
    adapter = AttiaAdapter()
    space = adapter.continuous_search_space()
    discrete_pool = adapter.load_candidate_pool()

    init_indices = [0, 10, 20, 30, 40]

    # Test random trajectory
    hist_random = run_single_attia_continuous_trajectory(
        search_space=space,
        discrete_pool=discrete_pool,
        init_indices=init_indices,
        total_queries=3,
        strategy="random",
        optimizer_seed=42,
        n_candidates_per_step=100,
        refine_continuous=False,
    )
    assert len(hist_random) == 4  # step 0 + 3 queries

    # Test gp_ucb trajectory
    hist_ucb = run_single_attia_continuous_trajectory(
        search_space=space,
        discrete_pool=discrete_pool,
        init_indices=init_indices,
        total_queries=3,
        strategy="gp_ucb",
        optimizer_seed=42,
        n_candidates_per_step=100,
        refine_continuous=True,
    )
    assert len(hist_ucb) == 4
    for row in hist_ucb:
        assert "best_observed_lifetime" in row
        assert "reference_true_lifetime" in row
        assert "is_novel" in row


def test_attia_continuous_benchmark_mini_end_to_end(tmp_path: Path) -> None:
    adapter = AttiaAdapter()
    summary = run_attia_continuous_benchmark(
        adapter=adapter,
        budgets=(6, 7),
        initial_policies=5,
        n_seeds=2,
        output_dir=tmp_path,
    )

    assert "benchmark" in summary
    assert "discrete_grid_optimum" in summary
    assert summary["discrete_grid_optimum"]["reference_true_lifetime"] == DISCRETE_GRID_OPTIMUM_TRUE
    assert "best_continuous_protocol_discovered" in summary
    assert (tmp_path / "benchmark_summary.json").is_file()
    assert (tmp_path / "search_space_summary.json").is_file()
    assert (tmp_path / "optimization_history.csv").is_file()
    assert (tmp_path / "proposed_protocols.csv").is_file()
