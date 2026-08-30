import numpy as np
import pandas as pd
import pytest

from src.datasets.feconi import (
    FeCoNiAdapter,
    FeCoNiExperimentOracle,
    compute_derived_ni,
    generate_feconi_candidate_id,
    load_raw_feconi_mat,
)
from src.datasets.registry import get_dataset_adapter


def test_mat_loader_produces_exact_921_rows():
    raw = load_raw_feconi_mat()
    assert raw["C"].shape == (921, 3)
    assert raw["XRD"].shape == (921, 89)
    assert len(raw["Coer"]) == 921
    assert len(raw["Kerr"]) == 921
    assert len(raw["TTH"]) == 89


def test_composition_row_sums_approximate_100():
    raw = load_raw_feconi_mat()
    sums = np.sum(raw["C"], axis=1)
    assert np.all(sums >= 99.8)
    assert np.all(sums <= 100.2)
    assert np.isclose(np.mean(sums), 100.0, atol=0.1)


def test_candidate_ids_are_unique_and_formatted():
    adapter = FeCoNiAdapter(target="Kerr")
    pool = adapter.get_candidate_pool()
    assert len(pool) == 921
    assert len(pool["candidate_id"].unique()) == 921
    assert pool["candidate_id"].iloc[0] == "FECONI_000"
    assert pool["candidate_id"].iloc[920] == "FECONI_920"


def test_derived_third_composition_coordinate_reconstructs_correctly():
    adapter = FeCoNiAdapter(target="Kerr")
    df = adapter.load_data()
    for _, row in df.iterrows():
        co = row["Co"]
        fe = row["Fe"]
        ni = row["Ni"]
        reconstructed_ni = compute_derived_ni(co, fe)
        # Sum of row is within rounding tolerance of 100
        assert np.isclose(reconstructed_ni, ni, atol=0.2)


def test_hidden_target_columns_never_in_candidate_pool():
    adapter = FeCoNiAdapter(target="Kerr")
    pool = adapter.get_candidate_pool()
    assert "Kerr" not in pool.columns
    assert "Coer" not in pool.columns
    assert "XRD" not in pool.columns
    assert "TTH" not in pool.columns
    assert set(pool.columns) == {"candidate_id", "sample_index", "Co", "Fe", "Ni"}


def test_oracle_refuses_unknown_candidate_id():
    adapter = FeCoNiAdapter(target="Kerr")
    oracle = adapter.create_oracle()
    with pytest.raises(KeyError, match="not a valid measured physical material"):
        oracle.query("FECONI_9999")
    with pytest.raises(KeyError, match="not a valid measured physical material"):
        oracle.query("UNKNOWN_CANDIDATE")


def test_oracle_refuses_duplicate_reveal():
    adapter = FeCoNiAdapter(target="Kerr")
    oracle = adapter.create_oracle(allow_duplicate_queries=False)
    res1 = oracle.query("FECONI_000")
    assert res1["candidate_id"] == "FECONI_000"
    assert "Kerr" in res1
    with pytest.raises(ValueError, match="Duplicate experimental measurement"):
        oracle.query("FECONI_000")


def test_target_selection_kerr_vs_coer_works_independently():
    kerr_adapter = FeCoNiAdapter(target="Kerr")
    coer_adapter = FeCoNiAdapter(target="Coer")

    assert kerr_adapter.get_spec().target_column == "Kerr"
    assert coer_adapter.get_spec().target_column == "Coer"

    kerr_oracle = kerr_adapter.create_oracle()
    coer_oracle = coer_adapter.create_oracle()

    assert kerr_oracle.target_column == "Kerr"
    assert coer_oracle.target_column == "Coer"
    assert np.isclose(kerr_oracle.global_best_value, 0.82504, atol=1e-4)
    assert np.isclose(coer_oracle.global_best_value, 10.9340, atol=1e-3)


def test_registry_resolution():
    adapter_kerr = get_dataset_adapter("feconi_kerr")
    assert isinstance(adapter_kerr, FeCoNiAdapter)
    assert adapter_kerr.target == "Kerr"

    adapter_coer = get_dataset_adapter("feconi_coercivity")
    assert isinstance(adapter_coer, FeCoNiAdapter)
    assert adapter_coer.target == "Coer"


def test_xrd_data_access_without_optimizer_leakage():
    adapter = FeCoNiAdapter(target="Kerr")
    xrd, tth = adapter.get_xrd_data()
    assert xrd.shape == (921, 89)
    assert tth.shape == (89,)
    # Candidate pool still clean
    pool = adapter.get_candidate_pool()
    assert "XRD" not in pool.columns
