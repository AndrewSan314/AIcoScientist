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


def test_candidate_schema_strict_id_and_conflict_handling():
    from src.optimization.candidates import normalize_candidate_schema, remove_observed

    adapter = DynamicCyclingAdapter()
    spec = adapter.spec

    # 1. candidate_id spec + missing protocol_id -> fail
    cand_missing_id = pd.DataFrame({
        feat: [0.1, 0.2] for feat in spec.candidate_columns
    })
    with pytest.raises(ValueError, match="missing required candidate ID column"):
        normalize_candidate_schema(cand_missing_id, spec)

    # 2. Two distinct IDs with identical design vectors remain distinct
    cand_identical_vectors = pd.DataFrame({
        "protocol_id": ["P17", "P23"],
        **{feat: [0.5, 0.5] for feat in spec.candidate_columns},
    })
    normalized = normalize_candidate_schema(cand_identical_vectors, spec)
    assert len(normalized) == 2
    assert set(normalized["protocol_id"]) == {"P17", "P23"}

    # 3. Observed P17 removes P17 only; P23 remains
    observed_p17 = pd.DataFrame({"protocol_id": ["P17"]})
    remaining = remove_observed(normalized, observed_p17, spec)
    assert len(remaining) == 1
    assert remaining.iloc[0]["protocol_id"] == "P23"

    # 4. Duplicate protocol_id rows with identical vectors deduplicate cleanly
    cand_duplicates = pd.DataFrame({
        "protocol_id": ["P01", "P01"],
        **{feat: [0.3, 0.3] for feat in spec.candidate_columns},
    })
    deduped = normalize_candidate_schema(cand_duplicates, spec)
    assert len(deduped) == 1
    assert deduped.iloc[0]["protocol_id"] == "P01"

    # 5. Conflicting duplicate protocol_id with different feature vectors raises ValueError
    cand_conflicting = pd.DataFrame({
        "protocol_id": ["P01", "P01"],
        **{feat: [0.3, 0.9] for feat in spec.candidate_columns},
    })
    with pytest.raises(ValueError, match="conflicting design feature vectors"):
        normalize_candidate_schema(cand_conflicting, spec)


