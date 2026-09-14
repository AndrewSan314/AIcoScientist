"""Tests verifying Hit@k, simple regret, and cumulative regret metrics."""

from __future__ import annotations
import pytest
import pandas as pd
from src.process.benchmarks.rediscovery import (
    ReplayStepRecord,
    RediscoveryTrajectory,
    summarize_trajectories,
)

def test_budget_metrics_summarization() -> None:
    steps = [
        ReplayStepRecord(1, "r1", 300.0, 300.0, 100.0, 100.0, 3, False),
        ReplayStepRecord(2, "r2", 400.0, 400.0, 0.0, 100.0, 1, True),
    ]
    t = RediscoveryTrajectory(
        policy="test",
        seed=42,
        initial_candidate_ids=["r0"],
        hidden_best_id="r2",
        hidden_best_value=400.0,
        initial_best_value=250.0,
        steps=steps,
        rediscovered=True,
        experiments_to_best=2,
        top3_rediscovered=True,
        experiments_to_top3=2,
        final_simple_regret=0.0,
        final_cumulative_regret=100.0,
    )
    summary = summarize_trajectories([t])
    assert summary.hit_rate_at_step[1] == 0.0
    assert summary.hit_rate_at_step[2] == 1.0
    assert summary.mean_simple_regret == 0.0
    assert summary.mean_cumulative_regret == 100.0
