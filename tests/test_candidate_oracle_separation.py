from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.dynamic_cycling import DYNAMIC_CYCLING_FEATURE_COLUMNS, DynamicCyclingAdapter


def test_candidate_pool_zero_oracle_data():
    adapter = DynamicCyclingAdapter()
    pool = adapter.load_candidate_pool()

    expected_cols = {"protocol_id", *DYNAMIC_CYCLING_FEATURE_COLUMNS}
    assert set(pool.columns) == expected_cols

    forbidden_oracle_columns = {
        "target_mean",
        "target_std",
        "efc_lifetime",
        "cycles_90",
        "n_replicates",
        "cell_id",
        "cell_name",
    }
    present_forbidden = set(pool.columns) & forbidden_oracle_columns
    assert len(present_forbidden) == 0, f"Candidate pool leaked oracle columns: {present_forbidden}"


def test_candidate_space_zero_oracle_data():
    adapter = DynamicCyclingAdapter()
    obs = pd.DataFrame({"protocol_id": ["P1", "P2"]})
    cands = adapter.candidate_space(obs)

    expected_cols = {"protocol_id", *DYNAMIC_CYCLING_FEATURE_COLUMNS}
    assert set(cands.columns) == expected_cols

    forbidden_oracle_columns = {
        "target_mean",
        "target_std",
        "efc_lifetime",
        "cycles_90",
        "n_replicates",
    }
    assert len(set(cands.columns) & forbidden_oracle_columns) == 0


def test_dynamic_hidden_oracle_cell_level_and_replicate_aware():
    from src.evaluation.oracle import OfflineOracle

    adapter = DynamicCyclingAdapter()
    hidden_oracle = adapter.load_hidden_oracle()

    # Hidden oracle must contain all 92 individual cell records
    assert len(hidden_oracle) == 92
    assert "cell_id" in hidden_oracle.columns
    assert "efc_lifetime" in hidden_oracle.columns

    oracle = OfflineOracle(hidden_oracle, adapter.spec, replicate_policy="mean")
    candidate_pool = adapter.load_candidate_pool()

    # Query first candidate protocol
    first_proto = candidate_pool.iloc[0]
    resp = oracle.query(first_proto)

    assert resp.target > 0
    assert resp.metadata["n_replicates"] >= 1
    assert "target_std" in resp.metadata
    # Oracle response hides raw row array from optimizer
    assert isinstance(resp.target, float)
    with pytest.raises(KeyError, match="not accessible"):
        _ = resp["raw_rows"]

