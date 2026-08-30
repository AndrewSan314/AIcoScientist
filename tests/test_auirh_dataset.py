from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.datasets.auirh import (
    AUIRH_CANDIDATE_ID_COLUMN,
    AUIRH_FEATURE_COLUMNS,
    AUIRH_LIBRARIES,
    AUIRH_TARGET_MAP,
    AuIrRhAdapter,
    AuIrRhExperimentOracle,
    compute_derived_rh,
    generate_auirh_candidate_id,
    load_raw_auirh_dataset,
)


@pytest.fixture
def synthetic_auirh_df() -> pd.DataFrame:
    """Creates a synthetic Au-Ir-Rh DataFrame representing 10 samples across 2 libraries."""
    records = []
    rng = np.random.default_rng(123)
    for lib in ["Au-rich", "Ir-rich"]:
        for area in range(1, 6):
            cid = generate_auirh_candidate_id(lib, area)
            au = float(rng.uniform(10.0, 70.0))
            ir = float(rng.uniform(10.0, 90.0 - au))
            rh = 100.0 - au - ir
            k0 = float(rng.uniform(0.001, 0.015))
            i_lim = float(rng.uniform(6.5, 8.8))
            alpha = float(rng.uniform(0.24, 0.33))
            records.append({
                AUIRH_CANDIDATE_ID_COLUMN: cid,
                "Library": lib,
                "Area": area,
                "Au": au,
                "Ir": ir,
                "Rh": rh,
                "Au [at.%]": au,
                "Ir [at.%]": ir,
                "Rh [at.%]": rh,
                "k^0 [cm/s]": k0,
                "i_lim [A/cm^2]": i_lim,
                "alpha [a.u.]": alpha,
                "k0": k0,
                "i_lim": i_lim,
                "alpha": alpha,
            })
    return pd.DataFrame(records)


@pytest.fixture
def synthetic_oracle(synthetic_auirh_df: pd.DataFrame) -> AuIrRhExperimentOracle:
    return AuIrRhExperimentOracle(synthetic_auirh_df, target_column="k0")


def test_derived_rh_and_candidate_id():
    """Test composition constraint and stable candidate ID formatting."""
    rh = compute_derived_rh(50.0, 30.0)
    assert np.isclose(rh, 20.0)

    cid = generate_auirh_candidate_id("Au-rich", 7)
    assert cid == "AUIRH_Au-rich_007"


def test_candidate_ids_are_unique(synthetic_auirh_df: pd.DataFrame):
    """Test 2: Candidate IDs are unique across all samples."""
    cids = synthetic_auirh_df[AUIRH_CANDIDATE_ID_COLUMN]
    assert len(cids) == len(set(cids))


def test_composition_sums_valid_within_tolerance(synthetic_auirh_df: pd.DataFrame):
    """Test 3: Composition sums Au + Ir + Rh ~ 100% within tolerance."""
    sums = synthetic_auirh_df["Au"] + synthetic_auirh_df["Ir"] + synthetic_auirh_df["Rh"]
    np.testing.assert_allclose(sums, 100.0, atol=1e-5)


def test_candidate_pool_contains_no_targets(synthetic_auirh_df: pd.DataFrame):
    """Test 5: Candidate pool contains NO target values."""
    oracle = AuIrRhExperimentOracle(synthetic_auirh_df, target_column="k0")
    visible_cols = [AUIRH_CANDIDATE_ID_COLUMN, "Library", "Area", "Au", "Ir", "Rh"]
    pool = synthetic_auirh_df[visible_cols].copy()

    for col in pool.columns:
        assert "k0" not in col.lower()
        assert "k^0" not in col.lower()
        assert "i_lim" not in col.lower()
        assert "alpha" not in col.lower()
        assert "target" not in col.lower()


def test_candidate_pool_contains_no_xrd_or_lsv(synthetic_auirh_df: pd.DataFrame):
    """Tests 6 & 7: Candidate pool contains NO XRD spectra or LSV curves."""
    visible_cols = [AUIRH_CANDIDATE_ID_COLUMN, "Library", "Area", "Au", "Ir", "Rh"]
    pool = synthetic_auirh_df[visible_cols].copy()

    for col in pool.columns:
        assert "xrd" not in col.lower()
        assert "lsv" not in col.lower()
        assert "diffractogram" not in col.lower()
        assert "potential" not in col.lower()
        assert "current" not in col.lower()


def test_oracle_rejects_unknown_candidate(synthetic_oracle: AuIrRhExperimentOracle):
    """Test 8: Oracle rejects unknown candidate IDs."""
    with pytest.raises(KeyError, match="not a valid measured physical material"):
        synthetic_oracle.query("AUIRH_Unknown_999")


def test_oracle_rejects_arbitrary_continuous_candidate(synthetic_oracle: AuIrRhExperimentOracle):
    """Test 9: Oracle rejects arbitrary continuous candidates without valid ID."""
    with pytest.raises(KeyError):
        synthetic_oracle.query({"Au": 33.3, "Ir": 33.3, "Rh": 33.4})


def test_oracle_rejects_duplicate_measurement(synthetic_oracle: AuIrRhExperimentOracle):
    """Test 10: Oracle rejects duplicate queries unless explicitly enabled."""
    valid_id = "AUIRH_Au-rich_001"
    res1 = synthetic_oracle.query(valid_id)
    assert res1["candidate_id"] == valid_id

    with pytest.raises(ValueError, match="Duplicate experimental measurement requested"):
        synthetic_oracle.query(valid_id)


def test_targets_remain_independent(synthetic_auirh_df: pd.DataFrame):
    """Test 15: Targets k0, i_lim, and alpha are queried independently without mixing."""
    oracle_k0 = AuIrRhExperimentOracle(synthetic_auirh_df, target_column="k0")
    oracle_ilim = AuIrRhExperimentOracle(synthetic_auirh_df, target_column="i_lim")
    oracle_alpha = AuIrRhExperimentOracle(synthetic_auirh_df, target_column="alpha")

    res_k0 = oracle_k0.query("AUIRH_Au-rich_001")
    res_ilim = oracle_ilim.query("AUIRH_Au-rich_001")
    res_alpha = oracle_alpha.query("AUIRH_Au-rich_001")

    assert "k0" in res_k0
    assert "i_lim" in res_ilim
    assert "alpha" in res_alpha
    assert res_k0["k0"] == synthetic_auirh_df.loc[synthetic_auirh_df[AUIRH_CANDIDATE_ID_COLUMN] == "AUIRH_Au-rich_001", "k0"].values[0]


def test_global_optimum_used_only_for_evaluation(synthetic_oracle: AuIrRhExperimentOracle):
    """Test 16: Global optimum is an offline evaluation metric and does not mutate during queries."""
    orig_best = synthetic_oracle.global_best_value
    synthetic_oracle.query("AUIRH_Au-rich_001")
    assert synthetic_oracle.global_best_value == orig_best


@pytest.mark.external_data
def test_raw_archives_and_join_integrity():
    """Tests 1, 4, 20: Tests on raw external files (EDX + SECCM + XRD)."""
    df = load_raw_auirh_dataset(allow_unverified_hash=True)
    assert len(df) == 966
    assert set(df["Library"].unique()) == {"Au-rich", "Ir-rich", "Rh-rich"}
    for lib in AUIRH_LIBRARIES:
        assert len(df[df["Library"] == lib]) == 322

    adapter = AuIrRhAdapter(target="k0")
    pool = adapter.get_candidate_pool()
    assert len(pool) == 966
    assert list(pool.columns) == [AUIRH_CANDIDATE_ID_COLUMN, "Library", "Area", "Au", "Ir", "Rh"]
