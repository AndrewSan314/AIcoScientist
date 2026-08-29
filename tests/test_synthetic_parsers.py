from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import pytest

from src.datasets.dynamic_cycling import (
    DYNAMIC_CYCLING_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMN_MAP,
    load_raw_dynamic_cycling_data,
)
from src.datasets.severson import (
    extract_features,
    load_raw_batch_hdf5,
    truncate_to_horizon,
)


def test_synthetic_severson_h5py_structure(tmp_path: Path):
    """Tests Severson HDF5 parser logic with a minimal synthetic HDF5 file."""
    mat_path = tmp_path / "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
    with h5py.File(mat_path, "w") as f:
        batch_grp = f.create_group("batch")

        cl_ds = f.create_dataset("cl0", data=np.array([[1200.0]]))
        pol_ds = f.create_dataset("pol0", data=np.array([ord(c) for c in "3.6C(80%)-3.6C"], dtype=np.uint8))
        vdlin_ds = f.create_dataset("vd0", data=np.linspace(2.0, 3.6, 1000))

        sum_grp = f.create_group("sum0")
        sum_grp.create_dataset("cycle", data=np.arange(1, 150, dtype=float))
        sum_grp.create_dataset("QDischarge", data=np.linspace(1.1, 0.8, 149))
        sum_grp.create_dataset("IR", data=np.linspace(0.015, 0.025, 149))
        sum_grp.create_dataset("Tavg", data=np.linspace(30.0, 32.0, 149))
        sum_grp.create_dataset("Tmax", data=np.linspace(35.0, 37.0, 149))
        sum_grp.create_dataset("Tmin", data=np.linspace(25.0, 26.0, 149))
        sum_grp.create_dataset("chargetime", data=np.linspace(40.0, 45.0, 149))

        cyc_grp = f.create_group("cyc0")
        qdlin_sub_refs = []
        for cyc_idx in range(105):
            q_sub = f.create_dataset(f"qdlin_{cyc_idx}", data=np.linspace(1.0, 0.0, 1000))
            qdlin_sub_refs.append(q_sub.ref)

        ref_type = h5py.special_dtype(ref=h5py.Reference)
        qdlin_ds = cyc_grp.create_dataset("Qdlin", shape=(len(qdlin_sub_refs), 1), dtype=ref_type)
        for idx, r in enumerate(qdlin_sub_refs):
            qdlin_ds[idx, 0] = r

        # batch references
        batch_grp.create_dataset("cycle_life", shape=(1, 1), dtype=ref_type)
        batch_grp["cycle_life"][0, 0] = cl_ds.ref

        batch_grp.create_dataset("policy_readable", shape=(1, 1), dtype=ref_type)
        batch_grp["policy_readable"][0, 0] = pol_ds.ref

        batch_grp.create_dataset("Vdlin", shape=(1, 1), dtype=ref_type)
        batch_grp["Vdlin"][0, 0] = vdlin_ds.ref

        batch_grp.create_dataset("summary", shape=(1, 1), dtype=ref_type)
        batch_grp["summary"][0, 0] = sum_grp.ref

        batch_grp.create_dataset("cycles", shape=(1, 1), dtype=ref_type)
        batch_grp["cycles"][0, 0] = cyc_grp.ref

    cells = load_raw_batch_hdf5(mat_path)
    assert "c0" in cells
    assert cells["c0"]["cycle_life"] == 1200.0

    # Test truncation and feature extraction
    truncated = truncate_to_horizon(cells["c0"], horizon=100)
    assert truncated["summary"]["cycle"].max() <= 100
    feats = extract_features(truncated)
    assert "delta_q_var" in feats
    assert "q_discharge_100" in feats
    assert np.isfinite(feats["delta_q_var"])


def test_synthetic_dynamic_cycling_replicate_mismatch(tmp_path: Path):
    """Tests that conflicting replicate design coordinates are detected and rejected."""
    metadata = pd.DataFrame(
        {
            "cell_name": ["cell_01", "cell_02"],
            "protocol_type": ["Fast", "Fast"],
            "protocol_variant": ["v1", "v1"],
            "protocol_name": ["P1", "P1"],
            "avg_crate_exp": [1.0, 1.0],
        }
    )
    metadata.to_pickle(tmp_path / "metadata.pkl")

    proto_feat = pd.DataFrame(
        {
            "cell_name": ["cell_01", "cell_02"],
            "Average Current": [1.0, 1.0],
            "Normalized Current Variance": [0.1, 0.1],
            "Maximum Discharge Current": [1.5, 1.5],
            "Relative Charge Fraction": [0.5, 0.5],
            "Rest Fraction at High SOC": [0.2, 0.2],
            "Rest SOC": [0.8, 0.8],
            "Peak Frequency 1": [10.0, 25.0],  # Major mismatch (spread = 15.0 > 0.20)
            "Peak Frequency 2": [5.0, 5.0],
        }
    )
    proto_feat.to_pickle(tmp_path / "protocol_features.pkl")

    soh90 = pd.DataFrame(
        {
            "cell_name": ["cell_01", "cell_02"],
            "EFCs (with Diagnostic)": [800.0, 820.0],
            "Cycles": [850.0, 870.0],
        }
    )
    soh90.to_csv(tmp_path / "soh90.csv", index=False)

    with pytest.raises(ValueError, match="conflicting design coordinates"):
        load_raw_dynamic_cycling_data(tmp_path, expected_records=2, expected_protocols=1)


def test_synthetic_dynamic_cycling_missing_cells_raises(tmp_path: Path):
    """Tests that missing cells in raw files raise alignment errors."""
    metadata = pd.DataFrame(
        {
            "cell_name": ["cell_01", "cell_02"],
            "protocol_type": ["Fast", "Fast"],
            "protocol_variant": ["v1", "v1"],
            "protocol_name": ["P1", "P1"],
            "avg_crate_exp": [1.0, 1.0],
        }
    )
    metadata.to_pickle(tmp_path / "metadata.pkl")

    proto_feat = pd.DataFrame(
        {
            "cell_name": ["cell_01"],  # Missing cell_02
            "Average Current": [1.0],
            "Normalized Current Variance": [0.1],
            "Maximum Discharge Current": [1.5],
            "Relative Charge Fraction": [0.5],
            "Rest Fraction at High SOC": [0.2],
            "Rest SOC": [0.8],
            "Peak Frequency 1": [10.0],
            "Peak Frequency 2": [5.0],
        }
    )
    proto_feat.to_pickle(tmp_path / "protocol_features.pkl")

    soh90 = pd.DataFrame(
        {
            "cell_name": ["cell_01", "cell_02"],
            "EFCs (with Diagnostic)": [800.0, 820.0],
            "Cycles": [850.0, 870.0],
        }
    )
    soh90.to_csv(tmp_path / "soh90.csv", index=False)

    with pytest.raises(ValueError, match="Expected exactly 2 aligned cell records"):
        load_raw_dynamic_cycling_data(tmp_path, expected_records=2, expected_protocols=1)


def test_severson_malformed_batch_group_raises(tmp_path: Path):
    """Tests that a MAT file without 'batch' group fails loudly."""
    mat_path = tmp_path / "empty.mat"
    with h5py.File(mat_path, "w") as f:
        f.create_group("wrong_group")

    with pytest.raises(ValueError, match="does not contain required 'batch' group"):
        load_raw_batch_hdf5(mat_path)


def test_severson_missing_structural_dataset_raises(tmp_path: Path):
    """Tests that missing required datasets in batch group raise ValueError."""
    mat_path = tmp_path / "missing_datasets.mat"
    with h5py.File(mat_path, "w") as f:
        batch = f.create_group("batch")
        batch.create_dataset("cycle_life", data=np.array([[100.0]]))
        # Missing policy_readable, summary, cycles, Vdlin

    with pytest.raises(ValueError, match="batch group is malformed.*missing datasets"):
        load_raw_batch_hdf5(mat_path)


def test_severson_missing_qdlin_in_cycles_raises(tmp_path: Path):
    """Tests that a cell missing Qdlin in cycles group fails with ValueError."""
    mat_path = tmp_path / "missing_qdlin.mat"
    ref_type = h5py.special_dtype(ref=h5py.Reference)
    with h5py.File(mat_path, "w") as f:
        batch = f.create_group("batch")
        cl = f.create_dataset("cl", data=np.array([[500.0]]))
        pol = f.create_dataset("pol", data=np.array([ord("c")], dtype=np.uint8))
        vd = f.create_dataset("vd", data=np.linspace(2.0, 3.6, 10))
        sum_grp = f.create_group("sum_grp")
        cyc_grp = f.create_group("cyc_grp")  # No Qdlin!

        batch.create_dataset("cycle_life", shape=(1, 1), dtype=ref_type)
        batch["cycle_life"][0, 0] = cl.ref
        batch.create_dataset("policy_readable", shape=(1, 1), dtype=ref_type)
        batch["policy_readable"][0, 0] = pol.ref
        batch.create_dataset("Vdlin", shape=(1, 1), dtype=ref_type)
        batch["Vdlin"][0, 0] = vd.ref
        batch.create_dataset("summary", shape=(1, 1), dtype=ref_type)
        batch["summary"][0, 0] = sum_grp.ref
        batch.create_dataset("cycles", shape=(1, 1), dtype=ref_type)
        batch["cycles"][0, 0] = cyc_grp.ref

    with pytest.raises(ValueError, match="is missing 'Qdlin' in cycles group"):
        load_raw_batch_hdf5(mat_path)


def test_dynamic_cycling_shuffled_features_rejected(tmp_path: Path):
    """Tests that unindexed shuffled features with invariant violation are rejected."""
    metadata = pd.DataFrame(
        {
            "cell_name": ["cell_01", "cell_02"],
            "protocol_type": ["Fast", "Slow"],
            "protocol_variant": ["v1", "v2"],
            "protocol_name": ["P1", "P2"],
            "avg_crate_exp": [3.0, 1.0],
        }
    )
    metadata.to_pickle(tmp_path / "metadata.pkl")

    # Shuffled without cell_name index (Average Current: 1.0, 3.0 vs avg_crate_exp: 3.0, 1.0)
    proto_feat = pd.DataFrame(
        {
            "Average Current": [1.0, 3.0],
            "Normalized Current Variance": [0.1, 0.2],
            "Maximum Discharge Current": [1.5, 3.5],
            "Relative Charge Fraction": [0.5, 0.6],
            "Rest Fraction at High SOC": [0.2, 0.3],
            "Rest SOC": [0.8, 0.9],
            "Peak Frequency 1": [10.0, 20.0],
            "Peak Frequency 2": [5.0, 8.0],
        }
    )
    proto_feat.to_pickle(tmp_path / "protocol_features.pkl")

    soh90 = pd.DataFrame(
        {
            "cell_name": ["cell_01", "cell_02"],
            "EFCs (with Diagnostic)": [800.0, 600.0],
            "Cycles": [850.0, 650.0],
        }
    )
    soh90.to_csv(tmp_path / "soh90.csv", index=False)

    with pytest.raises(ValueError, match="does not align with metadata"):
        load_raw_dynamic_cycling_data(tmp_path, expected_records=2, expected_protocols=2)


