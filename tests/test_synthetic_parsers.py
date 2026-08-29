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


def test_synthetic_dynamic_cycling_replicate_validation(tmp_path: Path):
    """Tests strict replicate design vector validation including tolerances and failures."""
    metadata = pd.DataFrame(
        {
            "cell_name": ["cell_01", "cell_02"],
            "protocol_type": ["Fast", "Fast"],
            "protocol_variant": ["v1", "v1"],
            "protocol_name": ["P1", "P1"],
        }
    )
    metadata.to_pickle(tmp_path / "metadata.pkl")

    base_proto = {
        "cell_name": ["cell_01", "cell_02"],
        "Average Current": [1.0, 1.0],
        "Normalized Current Variance": [0.1, 0.1],
        "Maximum Discharge Current": [1.5, 1.5],
        "Relative Charge Fraction": [0.5, 0.5],
        "Rest Fraction at High SOC": [0.2, 0.2],
        "Rest SOC": [0.8, 0.8],
        "Peak Frequency 1": [0.001, 0.001],
        "Peak Frequency 2": [0.005, 0.005],
    }

    soh90 = pd.DataFrame(
        {
            "cell_name": ["cell_01", "cell_02"],
            "EFCs (with Diagnostic)": [800.0, 820.0],
            "Cycles": [850.0, 870.0],
        }
    )
    soh90.to_csv(tmp_path / "soh90.csv", index=False)

    # 1. Exact replicate vector passes
    proto_df = pd.DataFrame(base_proto)
    proto_df.to_pickle(tmp_path / "protocol_features.pkl")
    cells, protos = load_raw_dynamic_cycling_data(tmp_path, expected_records=2, expected_protocols=1)
    assert len(cells) == 2 and len(protos) == 1

    # 2. Tiny numerical serialization noise passes
    noisy_proto = dict(base_proto)
    noisy_proto["Average Current"] = [1.0, 1.0 + 1e-6]
    pd.DataFrame(noisy_proto).to_pickle(tmp_path / "protocol_features.pkl")
    cells, protos = load_raw_dynamic_cycling_data(tmp_path, expected_records=2, expected_protocols=1)
    assert len(cells) == 2

    # 3. Average current mismatch of 1e-3 fails when above justified tolerance
    mismatch_current = dict(base_proto)
    mismatch_current["Average Current"] = [1.0, 1.0 + 1e-3]
    pd.DataFrame(mismatch_current).to_pickle(tmp_path / "protocol_features.pkl")
    with pytest.raises(ValueError, match="replicate conflict on feature 'average_current'"):
        load_raw_dynamic_cycling_data(
            tmp_path,
            expected_records=2,
            expected_protocols=1,
            feature_tolerances={"average_current": 1e-4},
        )

    # 4. Peak frequency mismatch at 1e-4 scale fails when above tolerance
    mismatch_freq = dict(base_proto)
    mismatch_freq["Peak Frequency 1"] = [0.00005, 0.00050]  # Diff = 4.5e-4
    pd.DataFrame(mismatch_freq).to_pickle(tmp_path / "protocol_features.pkl")
    with pytest.raises(ValueError, match="replicate conflict on feature 'peak_frequency_1'"):
        load_raw_dynamic_cycling_data(
            tmp_path,
            expected_records=2,
            expected_protocols=1,
            feature_tolerances={"peak_frequency_1": 1e-4},
        )

    # 5. One mismatched feature causes the whole protocol to fail
    mismatch_single = dict(base_proto)
    mismatch_single["Maximum Discharge Current"] = [1.5, 3.0]
    pd.DataFrame(mismatch_single).to_pickle(tmp_path / "protocol_features.pkl")
    with pytest.raises(ValueError, match="replicate conflict on feature 'maximum_discharge_current'"):
        load_raw_dynamic_cycling_data(tmp_path, expected_records=2, expected_protocols=1)


def test_synthetic_dynamic_cycling_deterministic_id_and_set_equality(tmp_path: Path):
    """Tests that unindexed features without cell_name and ID set mismatches are strictly rejected."""
    metadata = pd.DataFrame(
        {
            "cell_name": ["cell_01", "cell_02"],
            "protocol_type": ["Fast", "Slow"],
            "protocol_variant": ["v1", "v2"],
            "protocol_name": ["P1", "P2"],
        }
    )
    metadata.to_pickle(tmp_path / "metadata.pkl")

    soh90 = pd.DataFrame(
        {
            "cell_name": ["cell_01", "cell_02"],
            "EFCs (with Diagnostic)": [800.0, 600.0],
            "Cycles": [850.0, 650.0],
        }
    )
    soh90.to_csv(tmp_path / "soh90.csv", index=False)

    # 1. Feature rows with NO cell_name column or index fail (even if ordered)
    proto_feat_no_id = pd.DataFrame(
        {
            "Average Current": [1.0, 2.0],
            "Normalized Current Variance": [0.1, 0.2],
            "Maximum Discharge Current": [1.5, 2.5],
            "Relative Charge Fraction": [0.5, 0.6],
            "Rest Fraction at High SOC": [0.2, 0.3],
            "Rest SOC": [0.8, 0.9],
            "Peak Frequency 1": [0.01, 0.02],
            "Peak Frequency 2": [0.05, 0.08],
        }
    )
    proto_feat_no_id.to_pickle(tmp_path / "protocol_features.pkl")
    with pytest.raises(ValueError, match="missing explicit deterministic 'cell_name'"):
        load_raw_dynamic_cycling_data(tmp_path, expected_records=2, expected_protocols=2)

    # 2. Explicit cell_name index succeeds
    proto_feat_with_idx = proto_feat_no_id.copy()
    proto_feat_with_idx.index = ["cell_01", "cell_02"]
    proto_feat_with_idx.to_pickle(tmp_path / "protocol_features.pkl")
    cells, protos = load_raw_dynamic_cycling_data(tmp_path, expected_records=2, expected_protocols=2)
    assert len(cells) == 2

    # 3. Duplicate cell IDs in features fail
    proto_feat_dup = proto_feat_no_id.copy()
    proto_feat_dup["cell_name"] = ["cell_01", "cell_01"]
    proto_feat_dup.to_pickle(tmp_path / "protocol_features.pkl")
    with pytest.raises(ValueError, match="duplicate cell_name entries"):
        load_raw_dynamic_cycling_data(tmp_path, expected_records=2, expected_protocols=2)

    # 4. Feature IDs mismatch metadata IDs fails set equality check before merge
    proto_feat_mismatch = proto_feat_no_id.copy()
    proto_feat_mismatch["cell_name"] = ["cell_01", "cell_99"]
    proto_feat_mismatch.to_pickle(tmp_path / "protocol_features.pkl")
    with pytest.raises(ValueError, match="Mismatch between metadata cell IDs and protocol feature cell IDs"):
        load_raw_dynamic_cycling_data(tmp_path, expected_records=2, expected_protocols=2)

    # 5. SOH90 IDs mismatch metadata IDs fails set equality check
    proto_feat_ok = proto_feat_no_id.copy()
    proto_feat_ok["cell_name"] = ["cell_01", "cell_02"]
    proto_feat_ok.to_pickle(tmp_path / "protocol_features.pkl")

    soh90_mismatch = pd.DataFrame(
        {
            "cell_name": ["cell_01", "cell_88"],
            "EFCs (with Diagnostic)": [800.0, 600.0],
            "Cycles": [850.0, 650.0],
        }
    )
    soh90_mismatch.to_csv(tmp_path / "soh90.csv", index=False)
    with pytest.raises(ValueError, match="Mismatch between metadata cell IDs and SOH90 target cell IDs"):
        load_raw_dynamic_cycling_data(tmp_path, expected_records=2, expected_protocols=2)

    # 6. Null or empty cell IDs fail
    metadata_null = metadata.copy()
    metadata_null.iloc[0, metadata_null.columns.get_loc("cell_name")] = ""
    metadata_null.to_pickle(tmp_path / "metadata.pkl")
    with pytest.raises(ValueError, match="null or empty cell_name"):
        load_raw_dynamic_cycling_data(tmp_path, expected_records=2, expected_protocols=2)



