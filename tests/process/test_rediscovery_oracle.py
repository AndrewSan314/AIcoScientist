"""Tests for the firewall-protected BlindExperimentalOracle."""

from __future__ import annotations

import pandas as pd
import pytest

from src.process.benchmarks.rediscovery import BlindExperimentalOracle


@pytest.fixture
def sample_pool() -> pd.DataFrame:
    return pd.DataFrame([
        {"candidate_id": "c1", "speed": 80.0, "gap": 0.1, "capacity": 5.0},
        {"candidate_id": "c2", "speed": 100.0, "gap": 0.2, "capacity": 9.5},
        {"candidate_id": "c3", "speed": 120.0, "gap": 0.3, "capacity": 11.0},  # hidden best
        {"candidate_id": "c4", "speed": 60.0, "gap": 0.1, "capacity": 3.2},
    ])


def test_oracle_initialization_and_properties(sample_pool: pd.DataFrame) -> None:
    oracle = BlindExperimentalOracle(
        sample_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
        control_columns=["speed", "gap"],
        minimize=False,
    )
    assert oracle.num_total == 4
    assert oracle.num_revealed == 0
    assert oracle.num_unrevealed == 4
    assert oracle.hidden_best_id == "c3"
    assert oracle.hidden_best_value == 11.0
    assert oracle.control_columns == ["speed", "gap"]
    assert not oracle.minimize


def test_oracle_visible_candidates_firewalls_targets(sample_pool: pd.DataFrame) -> None:
    oracle = BlindExperimentalOracle(
        sample_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
    )
    visible = oracle.visible_candidates()
    assert "capacity" not in visible.columns
    assert "candidate_id" in visible.columns
    assert "speed" in visible.columns
    assert "gap" in visible.columns
    assert len(visible) == 4


def test_oracle_reveal_single_candidate(sample_pool: pd.DataFrame) -> None:
    oracle = BlindExperimentalOracle(
        sample_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
    )
    assert not oracle.is_revealed("c1")
    record = oracle.reveal("c1")
    assert record["candidate_id"] == "c1"
    assert record["capacity"] == 5.0
    assert record["speed"] == 80.0
    assert oracle.is_revealed("c1")
    assert oracle.num_revealed == 1
    assert oracle.num_unrevealed == 3

    # Check visible_candidates updated
    visible = oracle.visible_candidates()
    assert len(visible) == 3
    assert "c1" not in visible["candidate_id"].values

    # Check revealed_history
    history = oracle.revealed_history()
    assert len(history) == 1
    assert history.iloc[0]["candidate_id"] == "c1"
    assert history.iloc[0]["capacity"] == 5.0


def test_oracle_rejects_double_reveal(sample_pool: pd.DataFrame) -> None:
    oracle = BlindExperimentalOracle(
        sample_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
    )
    oracle.reveal("c1")
    with pytest.raises(ValueError, match="already been revealed"):
        oracle.reveal("c1")


def test_oracle_rejects_unknown_candidate(sample_pool: pd.DataFrame) -> None:
    oracle = BlindExperimentalOracle(
        sample_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
    )
    with pytest.raises(KeyError, match="not found"):
        oracle.reveal("non_existent_id")


def test_oracle_secret_target_anti_cheating(sample_pool: pd.DataFrame) -> None:
    oracle = BlindExperimentalOracle(
        sample_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
    )
    # Direct access forbidden by default
    with pytest.raises(PermissionError, match="forbidden by oracle firewall"):
        oracle.secret_target("c3")

    # Allowed only with explicit evaluation flag
    val = oracle.secret_target("c3", allow_evaluation=True)
    assert val == 11.0


def test_oracle_anonymization(sample_pool: pd.DataFrame) -> None:
    oracle = BlindExperimentalOracle(
        sample_pool,
        candidate_id_column="candidate_id",
        target_column="capacity",
        anonymize=True,
        seed=42,
    )
    visible = oracle.visible_candidates()
    # IDs should be anon like cand_000, cand_001
    assert all(cid.startswith("cand_") for cid in visible["candidate_id"])
    assert len(visible) == 4
