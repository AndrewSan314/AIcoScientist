from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.datasets.dynamic_cycling import (
    DYNAMIC_CYCLING_FEATURE_COLUMNS,
    DynamicCyclingAdapter,
)
from src.evaluation.dynamic_cycling_benchmark import (
    evaluate_surrogate_prediction,
    run_dynamic_cycling_benchmark,
    run_single_optimization_trajectory,
)
from src.evaluation.oracle import OfflineOracle
from src.train_model import make_train_test_split


@pytest.fixture(scope="module")
def dynamic_adapter() -> DynamicCyclingAdapter:
    return DynamicCyclingAdapter()


def test_dynamic_cycling_spec_guards(dynamic_adapter: DynamicCyclingAdapter):
    """Verifies DatasetSpec integrity and anti-leakage properties."""
    spec = dynamic_adapter.spec

    assert spec.target_column not in spec.feature_columns
    assert spec.split_group_columns == ["protocol_id"]
    assert spec.candidate_id_column == "protocol_id"
    assert spec.entity_id_column == "cell_id"

    # Oracle columns disjoint from feature columns
    overlap = set(spec.feature_columns) & set(spec.oracle_columns)
    assert not overlap, f"Feature columns overlap with oracle columns: {overlap}"


def test_dynamic_cycling_two_representations(dynamic_adapter: DynamicCyclingAdapter):
    """Verifies cell-level (92 rows) and protocol-level (47 rows) datasets and replicate aggregations."""
    cells_df = dynamic_adapter.load_cells()
    protocols_df = dynamic_adapter.load_protocols()

    assert len(cells_df) == 92
    assert len(protocols_df) == 47

    assert set(cells_df["protocol_id"].unique()) == set(protocols_df["protocol_id"])

    # Verify replicate mean and std calculations
    for _, proto_row in protocols_df.head(10).iterrows():
        p_id = proto_row["protocol_id"]
        matching_cells = cells_df[cells_df["protocol_id"] == p_id]
        expected_mean = matching_cells["efc_lifetime"].mean()
        assert np.isclose(proto_row["target_mean"], expected_mean, atol=1e-5)
        if len(matching_cells) > 1:
            expected_std = matching_cells["efc_lifetime"].std(ddof=1)
            assert np.isclose(proto_row["target_std"], expected_std, atol=1e-5)


def test_dynamic_cycling_replicate_grouped_split(dynamic_adapter: DynamicCyclingAdapter):
    """Proves that replicate cells with the same protocol_id never cross train/test splits."""
    cells_df = dynamic_adapter.load_cells()
    spec = dynamic_adapter.spec

    train_idx, test_idx = make_train_test_split(cells_df, spec, test_size=0.30, random_state=42)

    train_protocols = set(cells_df.iloc[train_idx]["protocol_id"])
    test_protocols = set(cells_df.iloc[test_idx]["protocol_id"])

    overlap = train_protocols & test_protocols
    assert not overlap, f"Protocol replicates leaked across train/test partitions: {overlap}"


def test_dynamic_cycling_oracle_replicate_mean_protection(dynamic_adapter: DynamicCyclingAdapter):
    """Verifies OfflineOracle with replicate_policy='mean' returns mean target and hides private rows."""
    cells_df = dynamic_adapter.load_cells()
    protocols_df = dynamic_adapter.load_protocols()
    spec = dynamic_adapter.spec

    # Querying oracle initialized with cell replicates
    cell_oracle = OfflineOracle(cells_df, spec, replicate_policy="mean")

    sample_proto = protocols_df.iloc[0]
    cand_dict = {col: sample_proto[col] for col in spec.candidate_columns}
    cand_dict["protocol_id"] = sample_proto["protocol_id"]

    response = cell_oracle.query(cand_dict)
    assert np.isclose(response.target, sample_proto["target_mean"], atol=1e-5)
    assert response.metadata["n_replicates"] == sample_proto["n_replicates"]
    if sample_proto["n_replicates"] > 1:
        assert np.isclose(response.metadata["target_std"], sample_proto["target_std"], atol=1e-5)

    # Trying to access raw hidden dataframe or arbitrary private keys must fail
    with pytest.raises(KeyError):
        _ = response["raw_row"]



def test_dynamic_cycling_candidate_space_unseen_filter(dynamic_adapter: DynamicCyclingAdapter):
    """Verifies candidate_space strictly removes observed protocols to prevent redundant queries."""
    protocols_df = dynamic_adapter.load_protocols()
    observed_sample = pd.DataFrame(
        [
            {"protocol_id": protocols_df["protocol_id"].iloc[0]},
            {"protocol_id": protocols_df["protocol_id"].iloc[1]},
        ]
    )

    cands = dynamic_adapter.candidate_space(observed_sample)
    assert len(cands) == len(protocols_df) - 2
    assert protocols_df["protocol_id"].iloc[0] not in set(cands["protocol_id"])
    assert protocols_df["protocol_id"].iloc[1] not in set(cands["protocol_id"])


def test_dynamic_cycling_optimization_trajectories(dynamic_adapter: DynamicCyclingAdapter):
    """Verifies closed-loop BO execution across Random, Greedy, and GP-UCB."""
    protocols_df = dynamic_adapter.load_protocols()
    oracle = OfflineOracle(protocols_df, dynamic_adapter.spec, replicate_policy="mean")
    feature_cols = list(dynamic_adapter.spec.feature_columns)

    for strat in ["random", "greedy", "gp_ucb"]:
        rng = np.random.default_rng(42)
        history = run_single_optimization_trajectory(
            protocols_df=protocols_df,
            oracle=oracle,
            feature_cols=feature_cols,
            strategy=strat,
            initial_protocols=5,
            budget=10,
            rng=rng,
        )
        assert len(history) == 11  # step 0 to step 10
        # Best seen must be non-decreasing monotonically
        best_vals = [h["best_seen"] for h in history]
        for i in range(1, len(best_vals)):
            assert best_vals[i] >= best_vals[i - 1], f"Best seen decreased for strategy {strat}"


def test_dynamic_cycling_benchmark_end_to_end(tmp_path):
    """Verifies dynamic cycling benchmark runs end-to-end and saves all metrics."""
    summary = run_dynamic_cycling_benchmark(
        output_dir=tmp_path,
        initial_protocols=4,
        budget=8,
        n_seeds=3,
    )

    assert "surrogate_evaluation" in summary
    assert "strategy_comparison" in summary
    assert "random" in summary["strategy_comparison"]
    assert "greedy" in summary["strategy_comparison"]
    assert "gp_ucb" in summary["strategy_comparison"]

    assert (tmp_path / "model_metrics.json").exists()
    assert (tmp_path / "optimization_history.csv").exists()
    assert (tmp_path / "benchmark_summary.json").exists()
