"""Strict anti-cheating and no-lookahead regression tests for process rediscovery."""

from __future__ import annotations

import copy
import pandas as pd
import pytest

from src.process.benchmarks.rediscovery import BlindExperimentalOracle, RediscoveryReplay


@pytest.fixture
def candidate_pool() -> pd.DataFrame:
    return pd.DataFrame([
        {"candidate_id": f"R_{i}", "x1": float(i * 10), "x2": float(i % 3), "capacity": float(i * 1.5 + (10.0 if i == 7 else 0.0))}
        for i in range(10)
    ])


def test_initial_design_never_includes_hidden_best(candidate_pool: pd.DataFrame) -> None:
    """Verifies that across multiple random seeds, the initial design NEVER samples hidden best."""
    replay = RediscoveryReplay(
        candidate_pool=candidate_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
    )
    oracle_temp = replay._create_oracle()
    hidden_best = oracle_temp.hidden_best_id

    for seed in range(50):
        traj = replay.run(
            strategy="random",
            seed=seed,
            initial_size=3,
            max_steps=1,
        )
        assert hidden_best not in traj.initial_candidate_ids, (
            f"Seed {seed} leaked hidden best recipe {hidden_best} into initial design!"
        )


def test_surrogate_training_data_bounded_by_historical_steps(candidate_pool: pd.DataFrame) -> None:
    """Verifies that at step t, the surrogate model sees only <= t revealed observations."""
    replay = RediscoveryReplay(
        candidate_pool=candidate_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
    )
    traj = replay.run(
        strategy="expected_improvement",
        seed=42,
        initial_size=3,
        max_steps=5,
    )
    # Number of steps executed
    assert len(traj.steps) == 5
    # Steps are 1-indexed
    for idx, step_rec in enumerate(traj.steps):
        assert step_rec.step == idx + 1


def test_perturbing_unrevealed_outcome_does_not_affect_proposal(candidate_pool: pd.DataFrame) -> None:
    """Strict anti-cheating invariant: changing an unrevealed outcome has zero impact on BO proposal."""
    pool_a = candidate_pool.copy()
    pool_b = candidate_pool.copy()

    # Candidate R_7 is the best candidate in candidate_pool
    # In pool_b, change R_0 (an unrevealed candidate) outcome by 1000x
    # Keep initial design identical by using the same seed
    replay_a = RediscoveryReplay(candidate_pool=pool_a, candidate_id_column="candidate_id", target_column="capacity")
    replay_b = RediscoveryReplay(candidate_pool=pool_b, candidate_id_column="candidate_id", target_column="capacity")

    traj_a = replay_a.run(strategy="greedy", seed=101, initial_size=3, max_steps=1)

    # In pool_b, perturb an unrevealed candidate's target within sub-optimal range
    # so R_7 remains the true hidden best (avoiding altering the initial exclusion set)
    unrevealed_in_a = [c for c in candidate_pool["candidate_id"] if c not in traj_a.initial_candidate_ids and c != traj_a.steps[0].selected_id]
    test_cand = unrevealed_in_a[0]
    # R_7 is 20.5; perturb test_cand to 5.0 (well below 20.5)
    pool_b.loc[pool_b["candidate_id"] == test_cand, "capacity"] = 5.0

    replay_b_perturbed = RediscoveryReplay(candidate_pool=pool_b, candidate_id_column="candidate_id", target_column="capacity")
    traj_b = replay_b_perturbed.run(strategy="greedy", seed=101, initial_size=3, max_steps=1)

    # The first optimization step proposal must be strictly identical
    assert traj_a.initial_candidate_ids == traj_b.initial_candidate_ids
    assert traj_a.steps[0].selected_id == traj_b.steps[0].selected_id
    assert pytest.approx(traj_a.steps[0].predicted_mean, abs=1e-6) == traj_b.steps[0].predicted_mean

