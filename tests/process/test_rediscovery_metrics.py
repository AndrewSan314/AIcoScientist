"""Tests for rediscovery evaluation metrics and statistical aggregations."""

from __future__ import annotations

import pandas as pd
import pytest

from src.process.benchmarks.rediscovery import (
    RediscoveryReplay,
    RediscoveryTrajectory,
    ReplayStepRecord,
    summarize_trajectories,
)


def test_simple_regret_is_non_negative_and_monotonic() -> None:
    pool = pd.DataFrame([
        {"candidate_id": f"R_{i}", "x": float(i), "capacity": float(i * 2.0)}
        for i in range(8)
    ])
    replay = RediscoveryReplay(candidate_pool=pool, candidate_id_column="candidate_id", target_column="capacity")
    traj = replay.run(strategy="expected_improvement", seed=42, initial_size=2, max_steps=5)

    regrets = [s.simple_regret for s in traj.steps]
    # Simple regret must be non-negative
    assert all(r >= 0.0 for r in regrets)
    # Simple regret must be non-increasing (monotonic improvement)
    for i in range(len(regrets) - 1):
        assert regrets[i + 1] <= regrets[i] + 1e-9


def test_cumulative_regret_equals_sum_of_simple_regret() -> None:
    pool = pd.DataFrame([
        {"candidate_id": f"R_{i}", "x": float(i), "capacity": float(i * 2.0)}
        for i in range(6)
    ])
    replay = RediscoveryReplay(candidate_pool=pool, candidate_id_column="candidate_id", target_column="capacity")
    traj = replay.run(strategy="greedy", seed=55, initial_size=2, max_steps=3)

    running_sum = 0.0
    for s in traj.steps:
        running_sum += s.simple_regret
        assert pytest.approx(s.cumulative_regret, abs=1e-6) == running_sum


def test_summarize_trajectories_computes_accurate_statistics() -> None:
    t1 = RediscoveryTrajectory(
        policy="test_pol", seed=1, initial_candidate_ids=["a", "b"],
        hidden_best_id="best", hidden_best_value=10.0, initial_best_value=5.0,
        rediscovered=True, experiments_to_best=2, final_simple_regret=0.0, final_cumulative_regret=5.0,
        steps=[
            ReplayStepRecord(1, "c", 7.0, 7.0, 3.0, 3.0, 2, False),
            ReplayStepRecord(2, "best", 10.0, 10.0, 0.0, 3.0, 1, True),
        ]
    )
    t2 = RediscoveryTrajectory(
        policy="test_pol", seed=2, initial_candidate_ids=["c", "d"],
        hidden_best_id="best", hidden_best_value=10.0, initial_best_value=6.0,
        rediscovered=True, experiments_to_best=1, final_simple_regret=0.0, final_cumulative_regret=0.0,
        steps=[
            ReplayStepRecord(1, "best", 10.0, 10.0, 0.0, 0.0, 1, True),
            ReplayStepRecord(2, "e", 4.0, 10.0, 0.0, 0.0, 1, False),
        ]
    )

    summary = summarize_trajectories([t1, t2])
    assert summary.policy == "test_pol"
    assert summary.num_seeds == 2
    assert summary.success_rate == 1.0
    assert summary.mean_experiments_to_best == 1.5
    assert summary.median_experiments_to_best == 1.5
    assert summary.mean_simple_regret == 0.0
    assert summary.hit_rate_at_step[1] == 0.5
    assert summary.hit_rate_at_step[2] == 1.0
