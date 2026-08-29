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
