import hashlib
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import scipy.io

from src.datasets.feconi import (
    FECONI_CANDIDATE_ID_COLUMN,
    FECONI_FEATURE_COLUMNS,
    FECONI_ORACLE_COLUMNS,
    FeCoNiAdapter,
    FeCoNiExperimentOracle,
    compute_derived_ni,
    generate_feconi_candidate_id,
    load_raw_feconi_mat,
)
from src.datasets.registry import get_dataset_adapter


@pytest.fixture
def synthetic_feconi_df() -> pd.DataFrame:
    """Creates a self-contained synthetic Fe-Co-Ni candidate & oracle DataFrame."""
    rng = np.random.default_rng(42)
    n_samples = 30
    co = rng.uniform(5.0, 80.0, size=n_samples)
    fe = rng.uniform(5.0, 80.0 - co * 0.5, size=n_samples)
    ni = 100.0 - co - fe

    # Synthetic Kerr and Coer
    kerr = 0.3 + 0.5 * (fe / 100.0) + 0.05 * rng.standard_normal(n_samples)
    coer = 1.0 + 8.0 * (co / 100.0) * (ni / 100.0) + 0.1 * rng.standard_normal(n_samples)

    return pd.DataFrame(
        {
            FECONI_CANDIDATE_ID_COLUMN: [generate_feconi_candidate_id(i) for i in range(n_samples)],
            "sample_index": np.arange(n_samples, dtype=int),
            "Co": co,
            "Fe": fe,
            "Ni": ni,
            "Kerr": kerr,
            "Coer": coer,
        }
    )


# ---------------------------------------------------------------------------
# Self-Contained Tests (Runs in Core CI without external MAT files)
# ---------------------------------------------------------------------------

def test_sha256_mismatch_raises_value_error(tmp_path: Path):
    """Test 2: Strict SHA256 mismatch raises ValueError by default, but can be overridden."""
    dummy_mat_path = tmp_path / "dummy_feconi.mat"
    # Create a valid MAT structure with dummy content that will have mismatched SHA256
    C_dummy = np.zeros((921, 3))
    C_dummy[:, 0] = 50.0
    C_dummy[:, 1] = 30.0
    C_dummy[:, 2] = 20.0
    Coer_dummy = np.ones((921, 1))
    Kerr_dummy = np.ones((921, 1)) * 0.5
    TTH_dummy = np.linspace(40, 62, 89).reshape(1, -1)
    XRD_dummy = np.ones((921, 89))

    scipy.io.savemat(
        str(dummy_mat_path),
        {"C": C_dummy, "Coer": Coer_dummy, "Kerr": Kerr_dummy, "TTH": TTH_dummy, "XRD": XRD_dummy},
    )

    # Default strict behavior -> raises ValueError
    with pytest.raises(ValueError, match="SHA256 hash mismatch for Fe-Co-Ni dataset"):
        load_raw_feconi_mat(mat_path=dummy_mat_path, allow_unverified_hash=False)

    # With explicit allow_unverified_hash=True -> succeeds
    raw = load_raw_feconi_mat(mat_path=dummy_mat_path, allow_unverified_hash=True)
    assert raw["C"].shape == (921, 3)


def test_oracle_refuses_unknown_candidate_id_synthetic(synthetic_feconi_df: pd.DataFrame):
    """Test 4: Oracle refuses unknown candidate IDs."""
    oracle = FeCoNiExperimentOracle(full_records_df=synthetic_feconi_df, target_column="Kerr")
    with pytest.raises(KeyError, match="not a valid measured physical material"):
        oracle.query("FECONI_9999")
    with pytest.raises(KeyError, match="not a valid measured physical material"):
        oracle.query("NON_EXISTENT_ID")


def test_oracle_refuses_duplicate_reveal_synthetic(synthetic_feconi_df: pd.DataFrame):
    """Test 5: Oracle refuses duplicate measurements when allow_duplicate_queries=False."""
    oracle = FeCoNiExperimentOracle(full_records_df=synthetic_feconi_df, target_column="Kerr", allow_duplicate_queries=False)
    res1 = oracle.query("FECONI_000")
    assert res1["candidate_id"] == "FECONI_000"
    assert "Kerr" in res1
    with pytest.raises(ValueError, match="Duplicate experimental measurement"):
        oracle.query("FECONI_000")


def test_oracle_target_selection_kerr_vs_coer_independent(synthetic_feconi_df: pd.DataFrame):
    """Test independent target initialization in Oracle."""
    kerr_oracle = FeCoNiExperimentOracle(full_records_df=synthetic_feconi_df, target_column="Kerr")
    coer_oracle = FeCoNiExperimentOracle(full_records_df=synthetic_feconi_df, target_column="Coer")

    assert kerr_oracle.target_column == "Kerr"
    assert coer_oracle.target_column == "Coer"
    assert "Kerr" in kerr_oracle.query("FECONI_001")
    assert "Coer" in coer_oracle.query("FECONI_001")


def test_derived_third_composition_coordinate_reconstructs_correctly(synthetic_feconi_df: pd.DataFrame):
    """Test 15: Derived Ni composition coordinate matches sum constraint."""
    for _, row in synthetic_feconi_df.iterrows():
        co = row["Co"]
        fe = row["Fe"]
        ni = row["Ni"]
        reconstructed_ni = compute_derived_ni(co, fe)
        assert np.isclose(reconstructed_ni, ni, atol=1e-6)


def test_candidate_pool_excludes_targets_and_xrd_synthetic(synthetic_feconi_df: pd.DataFrame):
    """Test 3: Candidate pool only contains design features, never oracle targets or XRD."""
    visible_cols = [FECONI_CANDIDATE_ID_COLUMN, "sample_index", "Co", "Fe", "Ni"]
    cand_pool = synthetic_feconi_df[visible_cols].copy()
    for col in FECONI_ORACLE_COLUMNS:
        assert col not in cand_pool.columns


def test_registry_resolution():
    """Test registry resolves FeCoNi adapters without requiring immediate file loading."""
    adapter_kerr = get_dataset_adapter("feconi_kerr")
    assert isinstance(adapter_kerr, FeCoNiAdapter)
    assert adapter_kerr.target == "Kerr"

    adapter_coer = get_dataset_adapter("feconi_coercivity")
    assert isinstance(adapter_coer, FeCoNiAdapter)
    assert adapter_coer.target == "Coer"


# ---------------------------------------------------------------------------
# Real MAT Data Tests (Marked with @pytest.mark.external_data)
# ---------------------------------------------------------------------------

@pytest.mark.external_data
def test_mat_loader_produces_exact_921_rows():
    """Test 1: Real MAT file loads exact 921 physical samples."""
    raw = load_raw_feconi_mat()
    assert raw["C"].shape == (921, 3)
    assert raw["XRD"].shape == (921, 89)
    assert len(raw["Coer"]) == 921
    assert len(raw["Kerr"]) == 921
    assert len(raw["TTH"]) == 89


@pytest.mark.external_data
def test_composition_row_sums_approximate_100():
    raw = load_raw_feconi_mat()
    sums = np.sum(raw["C"], axis=1)
    assert np.all(sums >= 99.8)
    assert np.all(sums <= 100.2)
    assert np.isclose(np.mean(sums), 100.0, atol=0.1)


@pytest.mark.external_data
def test_candidate_ids_are_unique_and_formatted():
    adapter = FeCoNiAdapter(target="Kerr")
    pool = adapter.get_candidate_pool()
    assert len(pool) == 921
    assert len(pool["candidate_id"].unique()) == 921
    assert pool["candidate_id"].iloc[0] == "FECONI_000"
    assert pool["candidate_id"].iloc[920] == "FECONI_920"


@pytest.mark.external_data
def test_xrd_data_access_without_optimizer_leakage():
    adapter = FeCoNiAdapter(target="Kerr")
    xrd, tth = adapter.get_xrd_data()
    assert xrd.shape == (921, 89)
    assert tth.shape == (89,)
    pool = adapter.get_candidate_pool()
    assert "XRD" not in pool.columns
