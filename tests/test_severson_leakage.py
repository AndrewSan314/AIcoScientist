from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from src.datasets.base import DatasetSpec
from src.datasets.severson import (
    SEVERSON_FEATURE_COLUMNS,
    SEVERSON_ORACLE_COLUMNS,
    SeversonAdapter,
    extract_features,
    truncate_to_horizon,
)
from src.evaluation.severson_benchmark import run_severson_benchmark
from src.train_model import make_train_test_split


@pytest.fixture(scope="module")
def severson_df() -> pd.DataFrame:
    adapter = SeversonAdapter()
    return adapter.load()


def test_severson_spec_leakage_guards():
    """Verifies DatasetSpec enforces no overlap between features and oracle/target columns."""
    adapter = SeversonAdapter()
    spec = adapter.spec

    assert spec.target_column not in spec.feature_columns
    assert spec.target_column == "cycle_life"
    assert spec.feature_horizon == 100

    # Ensure oracle columns and feature columns are strictly disjoint
    overlap = set(spec.feature_columns) & set(spec.oracle_columns)
    assert not overlap, f"Feature columns and oracle columns overlap: {overlap}"

    # Cycle life must never appear in feature columns
    assert "cycle_life" not in spec.feature_columns
    assert "split" not in spec.feature_columns


def test_severson_physical_cell_reconstruction_uniqueness(severson_df: pd.DataFrame):
    """Proves that after reconstruction, each physical cell is unique and carry-over cells are merged."""
    assert len(severson_df) == 124

    # Every physical_cell_id must be unique
    assert severson_df["physical_cell_id"].nunique() == 124

    # The 5 carry-over cells from Batch 2 (b2c7, b2c8, b2c9, b2c15, b2c16) must NOT exist as independent cells
    b2_carryover = {"b2c7", "b2c8", "b2c9", "b2c15", "b2c16"}
    existing_ids = set(severson_df["physical_cell_id"])
    assert not (existing_ids & b2_carryover), f"Carryover cells appeared as independent cells: {existing_ids & b2_carryover}"

    # The merged Batch 1 cells must be present
    b1_merged = {"b1c0", "b1c1", "b1c2", "b1c3", "b1c4"}
    assert b1_merged.issubset(existing_ids)

    # Excluded noisy and non-reaching cells must NOT exist
    b1_excluded = {"b1c8", "b1c10", "b1c12", "b1c13", "b1c22"}
    b3_excluded = {"b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"}
    assert not (existing_ids & b1_excluded)
    assert not (existing_ids & b3_excluded)


def test_severson_hard_horizon_sentinel_invariance():
    """ADVERSARIAL: Replaces all data for cycles > 100 with extreme sentinel values and verifies output is identical."""
    # Synthetic cell data with 500 cycles
    n_cycles = 500
    cycles = np.arange(1, n_cycles + 1)
    q_discharge = 1.0 - 0.0002 * cycles
    ir = 0.015 + 0.00001 * cycles
    t_avg = np.full(n_cycles, 30.0)
    t_max = np.full(n_cycles, 35.0)
    t_min = np.full(n_cycles, 25.0)
    chargetime = np.full(n_cycles, 12.0)
    qdlin = [np.linspace(1.0, 0.0, 1000) * (1.0 - 0.0002 * c) for c in range(n_cycles)]

    clean_cell = {
        "cycle_life": 1500.0,
        "policy": "test_policy",
        "summary": {
            "cycle": cycles,
            "QDischarge": q_discharge.copy(),
            "IR": ir.copy(),
            "Tavg": t_avg.copy(),
            "Tmax": t_max.copy(),
            "Tmin": t_min.copy(),
            "chargetime": chargetime.copy(),
        },
        "qdlin": [arr.copy() for arr in qdlin],
    }

    # Corrupted / Sentinel cell where cycle 101+ has extreme corruptions
    corrupted_cell = copy.deepcopy(clean_cell)
    corrupted_cell["summary"]["QDischarge"][100:] = 999999.0  # extreme sentinel
    corrupted_cell["summary"]["IR"][100:] = 888888.0
    corrupted_cell["summary"]["Tavg"][100:] = -9999.0
    corrupted_cell["summary"]["Tmax"][100:] = 777777.0
    corrupted_cell["summary"]["Tmin"][100:] = -55555.0
    corrupted_cell["summary"]["chargetime"][100:] = 123456.0
    for idx in range(100, n_cycles):
        corrupted_cell["qdlin"][idx] = np.full(1000, -99999.0)

    # 1. Truncate both to horizon=100
    trunc_clean = truncate_to_horizon(clean_cell, horizon=100)
    trunc_corrupted = truncate_to_horizon(corrupted_cell, horizon=100)

    # 2. Extract features
    clean_feats = extract_features(trunc_clean)
    corrupted_feats = extract_features(trunc_corrupted)

    # 3. Assert exact equality for every single feature
    for feat_name in SEVERSON_FEATURE_COLUMNS:
        assert np.isclose(
            clean_feats[feat_name], corrupted_feats[feat_name], rtol=1e-9, atol=1e-9
        ), f"Feature {feat_name} changed when cycle > 100 was altered! Clean: {clean_feats[feat_name]}, Corrupted: {corrupted_feats[feat_name]}"


def test_severson_split_isolation_and_no_leakage(severson_df: pd.DataFrame):
    """Verifies train and test partitions have zero overlap of physical cells."""
    train_cells = set(severson_df[severson_df["split"] == "train"]["physical_cell_id"])
    prim_test_cells = set(severson_df[severson_df["split"] == "primary_test"]["physical_cell_id"])
    sec_test_cells = set(severson_df[severson_df["split"] == "secondary_test"]["physical_cell_id"])

    assert len(train_cells) == 41
    assert len(prim_test_cells) == 43
    assert len(sec_test_cells) == 40

    assert not (train_cells & prim_test_cells), "Train and primary test overlap!"
    assert not (train_cells & sec_test_cells), "Train and secondary test overlap!"
    assert not (prim_test_cells & sec_test_cells), "Primary and secondary test overlap!"


def test_severson_make_train_test_split_grouped(severson_df: pd.DataFrame):
    """Verifies make_train_test_split respects physical_cell_id grouping."""
    adapter = SeversonAdapter()
    train_idx, test_idx = make_train_test_split(severson_df, adapter.spec, test_size=0.33, random_state=42)

    train_cells = set(severson_df.iloc[train_idx]["physical_cell_id"])
    test_cells = set(severson_df.iloc[test_idx]["physical_cell_id"])

    assert not (train_cells & test_cells), f"Leaked cells across split: {train_cells & test_cells}"


def test_severson_benchmark_end_to_end(tmp_path):
    """Verifies Severson benchmark runs end-to-end and creates expected outputs."""
    summary = run_severson_benchmark(output_dir=tmp_path)

    assert "rf_overall_test_mae" in summary
    assert "gp_overall_test_mae" in summary
    assert summary["horizon_cycles"] == 100

    assert (tmp_path / "model_metrics.json").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "feature_importance.csv").exists()
    assert (tmp_path / "benchmark_summary.json").exists()
    assert (tmp_path / "trained_model.pkl").exists()

    pred_df = pd.read_csv(tmp_path / "predictions.csv")
    assert len(pred_df) == 124
    assert set(pred_df.columns) == {
        "physical_cell_id",
        "split",
        "batch_id",
        "charging_policy",
        "actual_cycle_life",
        "predicted_cycle_life_rf",
        "predicted_cycle_life_gp",
    }
