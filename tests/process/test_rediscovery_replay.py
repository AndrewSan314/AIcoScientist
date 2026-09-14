"""Tests for closed-loop rediscovery replay coordinator."""

from __future__ import annotations

import pandas as pd
import pytest

from src.process.benchmarks.rediscovery import RediscoveryReplay


@pytest.fixture
def candidate_pool() -> pd.DataFrame:
    return pd.DataFrame([
        {"candidate_id": f"R_{i}", "x1": float(i * 5.0), "x2": float(10.0 - i), "capacity": float(1.0 + 2.0 * i)}
        for i in range(7)
    ])


def test_replay_is_deterministic_given_seed(candidate_pool: pd.DataFrame) -> None:
    replay = RediscoveryReplay(
        candidate_pool=candidate_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
    )
    t1 = replay.run(strategy="expected_improvement", seed=42, initial_size=2, max_steps=4)
    t2 = replay.run(strategy="expected_improvement", seed=42, initial_size=2, max_steps=4)

    assert t1.initial_candidate_ids == t2.initial_candidate_ids
    assert len(t1.steps) == len(t2.steps)
    for s1, s2 in zip(t1.steps, t2.steps):
        assert s1.step == s2.step
        assert s1.selected_id == s2.selected_id
        assert s1.revealed_target == s2.revealed_target
        assert s1.best_so_far == s2.best_so_far
        assert s1.simple_regret == s2.simple_regret


def test_replay_exhausts_budget(candidate_pool: pd.DataFrame) -> None:
    # 7 candidates total; initial_size = 2 -> 5 remaining
    replay = RediscoveryReplay(
        candidate_pool=candidate_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
    )
    traj = replay.run(strategy="random", seed=99, initial_size=2, max_steps=10)
    # Should stop after revealing all 5 remaining
    assert len(traj.steps) == 5
    assert traj.rediscovered
    assert traj.experiments_to_best is not None
    assert 1 <= traj.experiments_to_best <= 5
    assert traj.final_simple_regret == 0.0


@pytest.mark.parametrize("policy", [
    "random",
    "greedy",
    "gp_ucb",
    "expected_improvement",
    "noisy_expected_improvement",
])
def test_replay_all_supported_policies(candidate_pool: pd.DataFrame, policy: str) -> None:
    replay = RediscoveryReplay(
        candidate_pool=candidate_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
    )
    traj = replay.run(strategy=policy, seed=123, initial_size=2, max_steps=3)
    assert len(traj.steps) == 3
    assert traj.policy == policy
    assert traj.initial_best_value < traj.hidden_best_value
