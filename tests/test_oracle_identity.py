from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.base import DatasetSpec
from src.evaluation.oracle import OfflineOracle


@pytest.fixture
def sample_spec() -> DatasetSpec:
    return DatasetSpec(
        name="test_dataset",
        id_column="protocol_id",
        entity_id_column="cell_id",
        candidate_id_column="protocol_id",
        feature_columns=["rate", "temp"],
        target_column="target",
        objective="maximize",
        candidate_columns=["rate", "temp"],
    )


def test_oracle_lookup_by_candidate_id(sample_spec):
    df = pd.DataFrame(
        {
            "protocol_id": ["P1", "P2", "P3"],
            "rate": [1.0, 2.0, 3.0],
            "temp": [25.0, 30.0, 35.0],
            "target": [500.0, 600.0, 700.0],
        }
    )
    oracle = OfflineOracle(df, sample_spec)

    # Query with protocol_id alone
    resp = oracle.query({"protocol_id": "P2"})
    assert resp.target == 600.0
    assert resp.candidate["protocol_id"] == "P2"
    assert resp.candidate["rate"] == 2.0
    assert resp.candidate["temp"] == 30.0


def test_oracle_validates_design_coordinate_tolerance(sample_spec):
    df = pd.DataFrame(
        {
            "protocol_id": ["P1"],
            "rate": [1.00000],
            "temp": [25.00000],
            "target": [500.0],
        }
    )
    oracle = OfflineOracle(df, sample_spec)

    # Slight float delta within atol=1e-4 succeeds
    resp = oracle.query({"protocol_id": "P1", "rate": 1.00002, "temp": 25.00001})
    assert resp.target == 500.0

    # Contradicting design coordinates fail with ValueError
    with pytest.raises(ValueError, match="conflicts with ground truth"):
        oracle.query({"protocol_id": "P1", "rate": 2.5, "temp": 25.0})


def test_oracle_candidate_id_missing_raises_key_error(sample_spec):
    df = pd.DataFrame(
        {
            "protocol_id": ["P1"],
            "rate": [1.0],
            "temp": [25.0],
            "target": [500.0],
        }
    )
    oracle = OfflineOracle(df, sample_spec)

    with pytest.raises(KeyError, match="No exact ground-truth candidate exists"):
        oracle.query({"protocol_id": "P999"})


def test_oracle_replicate_policy_handling(sample_spec):
    df_replicates = pd.DataFrame(
        {
            "protocol_id": ["P1", "P1"],
            "cell_id": ["C1", "C2"],
            "rate": [1.0, 1.0],
            "temp": [25.0, 25.0],
            "target": [400.0, 600.0],
        }
    )
    # Error mode
    oracle_err = OfflineOracle(df_replicates, sample_spec, replicate_policy="error")
    with pytest.raises(ValueError, match="ambiguous"):
        oracle_err.query({"protocol_id": "P1"})

    # Mean mode
    oracle_mean = OfflineOracle(df_replicates, sample_spec, replicate_policy="mean")
    resp = oracle_mean.query({"protocol_id": "P1"})
    assert resp.target == 500.0
    assert resp.metadata["n_replicates"] == 2
    assert abs(resp.metadata["target_std"] - 141.421356) < 1e-3


def test_candidate_identity_preserves_distinct_protocols_with_identical_features(sample_spec):
    from src.optimization.candidates import normalize_candidate_schema, remove_observed

    # P17 and P23 have identical design features
    candidates = pd.DataFrame(
        {
            "protocol_id": ["P17", "P23", "P30"],
            "rate": [1.5, 1.5, 3.0],
            "temp": [30.0, 30.0, 45.0],
        }
    )

    norm_cands = normalize_candidate_schema(candidates, sample_spec)
    assert len(norm_cands) == 3
    assert set(norm_cands["protocol_id"]) == {"P17", "P23", "P30"}

    # Observing P17 removes only P17, P23 remains available
    observed = pd.DataFrame(
        {
            "protocol_id": ["P17"],
            "rate": [1.5],
            "temp": [30.0],
            "target": [550.0],
        }
    )
    remaining = remove_observed(candidates, observed, sample_spec)
    assert len(remaining) == 2
    assert "P17" not in remaining["protocol_id"].values
    assert "P23" in remaining["protocol_id"].values
    assert "P30" in remaining["protocol_id"].values

