from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
from scipy import stats

from src.datasets.base import DatasetAdapter, DatasetSpec

logger = logging.getLogger(__name__)

SEVERSON_FEATURE_COLUMNS: list[str] = [
    # Paper-inspired Delta Q_100-10(V) statistics
    "delta_q_var",
    "delta_q_min",
    "delta_q_mean",
    "delta_q_skew",
    "delta_q_kurt",
    # Early discharge capacity statistics
    "q_discharge_2",
    "q_discharge_100",
    "q_diff_100_2",
    "q_max_diff",
    "q_slope",
    "q_intercept",
    # Internal resistance statistics
    "ir_2",
    "ir_100",
    "ir_diff_100_2",
    "ir_mean",
    # Temperature statistics
    "t_avg_mean",
    "t_max_max",
    "t_min_min",
    # Charging time statistics
    "charge_time_mean",
    "charge_time_diff_100_2",
]

SEVERSON_ORACLE_COLUMNS: list[str] = [
    "cycle_life",
    "split",
    "batch_id",
    "original_cell_id",
    "charging_policy",
]


def _read_hdf5_str(f: h5py.File, ref: Any) -> str:
    if not isinstance(ref, h5py.Reference):
        return str(ref)
    obj = f[ref]
    arr = obj[()]
    if arr.dtype in (np.uint16, np.uint8):
        return "".join(chr(c) for c in arr.flatten())
    return str(arr)


def load_raw_batch_hdf5(mat_path: Path, max_cycle_extract: int = 105) -> dict[str, dict[str, Any]]:
    """Loads a single Severson MATLAB 7.3 HDF5 batch file with strict structural validation."""
    cells: dict[str, dict[str, Any]] = {}
    with h5py.File(mat_path, "r") as f:
        if "batch" not in f:
            raise ValueError(f"MAT file {mat_path} does not contain required 'batch' group")
        batch = f["batch"]
        required_keys = ["cycle_life", "policy_readable", "summary", "cycles", "Vdlin"]
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise ValueError(f"MAT file {mat_path} batch group is malformed (missing datasets: {missing_keys})")

        num_cells = batch["cycle_life"].shape[0]

        vdlin_ref = batch["Vdlin"][0, 0] if batch["Vdlin"].ndim == 2 else batch["Vdlin"][0]
        if isinstance(vdlin_ref, h5py.Reference):
            if not bool(vdlin_ref):
                raise ValueError(f"MAT file {mat_path} contains null Vdlin HDF5 reference")
            try:
                vdlin = f[vdlin_ref][()].flatten()
            except Exception as exc:
                raise ValueError(f"MAT file {mat_path} contains invalid Vdlin HDF5 reference: {exc}") from exc
        else:
            vdlin = np.asarray(vdlin_ref).flatten()

        for i in range(num_cells):
            cl_ref = batch["cycle_life"][i, 0] if batch["cycle_life"].ndim == 2 else batch["cycle_life"][i]
            if isinstance(cl_ref, h5py.Reference):
                if not bool(cl_ref):
                    raise ValueError(f"Cell {i} in {mat_path} contains null cycle_life HDF5 reference")
                try:
                    cl = float(f[cl_ref][0, 0])
                except Exception as exc:
                    raise ValueError(f"Cell {i} in {mat_path} contains invalid cycle_life HDF5 reference: {exc}") from exc
            else:
                cl = float(cl_ref)

            pref = batch["policy_readable"][i, 0] if batch["policy_readable"].ndim == 2 else batch["policy_readable"][i]
            policy = _read_hdf5_str(f, pref)

            summary: dict[str, np.ndarray] = {}
            sref = batch["summary"][i, 0] if batch["summary"].ndim == 2 else batch["summary"][i]
            if isinstance(sref, h5py.Reference):
                if not bool(sref):
                    raise ValueError(f"Cell {i} in {mat_path} contains null summary HDF5 reference")
                try:
                    s_grp = f[sref]
                except Exception as exc:
                    raise ValueError(f"Cell {i} in {mat_path} contains invalid summary HDF5 reference: {exc}") from exc
                for k in s_grp.keys():
                    summary[k] = s_grp[k][()].flatten()
            else:
                raise ValueError(f"Cell {i} in {mat_path} summary is not an HDF5 group reference")

            cycref = batch["cycles"][i, 0] if batch["cycles"].ndim == 2 else batch["cycles"][i]
            qdlin_list: list[np.ndarray] = []
            if isinstance(cycref, h5py.Reference):
                if not bool(cycref):
                    raise ValueError(f"Cell {i} in {mat_path} contains null cycles HDF5 reference")
                try:
                    cyc_grp = f[cycref]
                except Exception as exc:
                    raise ValueError(f"Cell {i} in {mat_path} contains invalid cycles HDF5 reference: {exc}") from exc
                if "Qdlin" not in cyc_grp:
                    raise ValueError(f"Cell {i} in {mat_path} is missing 'Qdlin' in cycles group")
                qdlin_ds = cyc_grp["Qdlin"]
                num_cyc = qdlin_ds.shape[0]
                for c_idx in range(min(num_cyc, max_cycle_extract)):
                    qref = qdlin_ds[c_idx, 0] if qdlin_ds.ndim == 2 else qdlin_ds[c_idx]
                    if isinstance(qref, h5py.Reference):
                        if not bool(qref):
                            raise ValueError(f"Cell {i} cycle {c_idx} contains null Qdlin reference")
                        try:
                            qdlin_list.append(f[qref][()].flatten())
                        except Exception as exc:
                            raise ValueError(f"Cell {i} cycle {c_idx} contains invalid Qdlin reference: {exc}") from exc
                    else:
                        qdlin_list.append(np.asarray(qref).flatten())
            else:
                raise ValueError(f"Cell {i} in {mat_path} cycles is not an HDF5 group reference")

            cell_key = f"c{i}"
            cells[cell_key] = {
                "cycle_life": cl,
                "policy": policy,
                "summary": summary,
                "qdlin": qdlin_list,
                "vdlin": vdlin,
            }
    return cells




def reconstruct_severson_cells(
    raw_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Reconstructs physical cells across batches following Severson et al. (Nature Energy 2019).

    Official reference rules:
    - Batch 1 (2017-05-12):
        Remove cells failing to reach 80% capacity: b1c8, b1c10, b1c12, b1c13, b1c22 (41 remaining).
        Cells continuing into Batch 2:
            b1c0 <- b2c7  (+662 cycles)
            b1c1 <- b2c8  (+981 cycles)
            b1c2 <- b2c9  (+1060 cycles)
            b1c3 <- b2c15 (+208 cycles)
            b1c4 <- b2c16 (+482 cycles)
    - Batch 2 (2017-06-30):
        Delete carry-over cells b2c7, b2c8, b2c9, b2c15, b2c16 (43 remaining).
    - Batch 3 (2018-04-12):
        Remove noisy channels: b3c37, b3c2, b3c23, b3c32, b3c42, b3c43 (40 remaining).

    Total physical cells = 41 + 43 + 40 = 124 cells.
    """
    b1_path = raw_dir / "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
    b2_path = raw_dir / "2017-06-30_batchdata_updated_struct_errorcorrect.mat"
    b3_path = raw_dir / "2018-04-12_batchdata_updated_struct_errorcorrect.mat"

    for p in (b1_path, b2_path, b3_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required Severson MAT file: {p}")

    b1_raw = load_raw_batch_hdf5(b1_path)
    b2_raw = load_raw_batch_hdf5(b2_path)
    b3_raw = load_raw_batch_hdf5(b3_path)

    batch1: dict[str, dict[str, Any]] = {f"b1{k}": v for k, v in b1_raw.items()}
    batch2: dict[str, dict[str, Any]] = {f"b2{k}": v for k, v in b2_raw.items()}
    batch3: dict[str, dict[str, Any]] = {f"b3{k}": v for k, v in b3_raw.items()}

    # 1. Remove Batch 1 cells that did not reach 80% capacity
    b1_to_remove = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]
    for k in b1_to_remove:
        batch1.pop(k, None)

    # 2. Merge carry-over cells from Batch 1 into Batch 2
    b1_cont_keys = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
    b2_cont_keys = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
    add_len = [662, 981, 1060, 208, 482]

    for i, b1k in enumerate(b1_cont_keys):
        b2k = b2_cont_keys[i]
        if b1k in batch1 and b2k in batch2:
            # Update cycle life
            batch1[b1k]["cycle_life"] += add_len[i]

            # Merge summary data
            s1 = batch1[b1k]["summary"]
            s2 = batch2[b2k]["summary"]
            for skey in list(s1.keys()):
                if skey in s2:
                    if skey == "cycle":
                        s1["cycle"] = np.hstack((s1["cycle"], s2["cycle"] + len(s1["cycle"])))
                    else:
                        s1[skey] = np.hstack((s1[skey], s2[skey]))

            # Append Qdlin if needed
            batch1[b1k]["qdlin"].extend(batch2[b2k]["qdlin"])

    # 3. Delete carry-over keys from Batch 2
    for k in b2_cont_keys:
        batch2.pop(k, None)

    # 4. Remove noisy channels from Batch 3
    b3_to_remove = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]
    for k in b3_to_remove:
        batch3.pop(k, None)

    # Combine reconstructed cells
    all_cells: dict[str, dict[str, Any]] = {}
    all_cells.update(batch1)
    all_cells.update(batch2)
    all_cells.update(batch3)

    if len(all_cells) != 124:
        raise RuntimeError(f"Expected exactly 124 physical cells after reconstruction, got {len(all_cells)}")

    # Assign official published splits
    # Batch 1 + Batch 2: 84 cells. Train = odd indices (1..81, 41 cells), Primary Test = even indices (0..82) + 83 (43 cells)
    # Batch 3: 40 cells -> Secondary Test
    cell_keys = list(all_cells.keys())
    split_map: dict[str, str] = {}

    num_bat1_2 = len(batch1) + len(batch2)  # 41 + 43 = 84
    train_indices = set(range(1, num_bat1_2 - 1, 2))  # 1, 3, 5, ..., 81 (41 cells)
    test_indices = set(range(0, num_bat1_2, 2)) | {83}  # 0, 2, 4, ..., 82, 83 (43 cells)
    secondary_test_indices = set(range(num_bat1_2, len(cell_keys)))  # 84..123 (40 cells)

    for idx, key in enumerate(cell_keys):
        if idx in train_indices:
            split_map[key] = "train"
        elif idx in test_indices:
            split_map[key] = "primary_test"
        elif idx in secondary_test_indices:
            split_map[key] = "secondary_test"
        else:
            split_map[key] = "train"

    return all_cells, split_map


def truncate_to_horizon(cell_data: dict[str, Any], horizon: int = 100) -> dict[str, Any]:
    """Hard-truncates all cycle data to cycle <= horizon BEFORE feature extraction.

    Guarantees no data from cycle > horizon is accessible to feature engineering.
    """
    summary = cell_data.get("summary", {})
    cycles_arr = summary.get("cycle", np.array([]))

    if len(cycles_arr) > 0:
        mask = cycles_arr <= horizon
        truncated_summary = {k: v[mask[: len(v)]] for k, v in summary.items()}
    else:
        truncated_summary = {k: v[:horizon] for k, v in summary.items()}

    truncated_qdlin = cell_data.get("qdlin", [])[:horizon]

    return {
        "cycle_life": cell_data.get("cycle_life"),
        "policy": cell_data.get("policy"),
        "vdlin": cell_data.get("vdlin"),
        "summary": truncated_summary,
        "qdlin": truncated_qdlin,
    }


def extract_features(truncated_cell: dict[str, Any]) -> dict[str, float]:
    """Extracts early-life features strictly from cycle <= 100 truncated cell data."""
    summary = truncated_cell.get("summary", {})
    qdlin = truncated_cell.get("qdlin", [])

    q_discharge = summary.get("QDischarge", np.array([]))
    ir = summary.get("IR", np.array([]))
    t_avg = summary.get("Tavg", np.array([]))
    t_max = summary.get("Tmax", np.array([]))
    t_min = summary.get("Tmin", np.array([]))
    chargetime = summary.get("chargetime", np.array([]))

    # Delta Q_100-10(V) curve features
    # Note: cycle 10 is index 9, cycle 100 is index 99 (or last available up to 100)
    idx_10 = min(9, len(qdlin) - 1) if len(qdlin) > 0 else -1
    idx_100 = min(99, len(qdlin) - 1) if len(qdlin) > 0 else -1

    if idx_10 >= 0 and idx_100 >= 0 and len(qdlin[idx_10]) > 0 and len(qdlin[idx_100]) > 0:
        delta_q = qdlin[idx_100] - qdlin[idx_10]
        var_val = float(np.var(delta_q))
        min_val = float(np.min(delta_q))
        delta_q_var = float(np.log10(max(var_val, 1e-12)))
        delta_q_min = float(np.log10(max(abs(min_val), 1e-12)))
        delta_q_mean = float(np.mean(delta_q))
        delta_q_skew = float(stats.skew(delta_q)) if len(delta_q) > 2 else 0.0
        delta_q_kurt = float(stats.kurtosis(delta_q)) if len(delta_q) > 2 else 0.0
    else:
        delta_q_var = 0.0
        delta_q_min = 0.0
        delta_q_mean = 0.0
        delta_q_skew = 0.0
        delta_q_kurt = 0.0

    # Discharge capacity features (cycles 2 to 100)
    if len(q_discharge) >= 2:
        q_discharge_2 = float(q_discharge[1]) if len(q_discharge) > 1 else float(q_discharge[0])
        q_idx_100 = min(99, len(q_discharge) - 1)
        q_discharge_100 = float(q_discharge[q_idx_100])
        q_diff_100_2 = q_discharge_100 - q_discharge_2

        q_window = q_discharge[1 : q_idx_100 + 1] if len(q_discharge) > 1 else q_discharge[:1]
        q_max_diff = float(np.max(q_window)) - q_discharge_2

        if len(q_window) >= 2:
            x_cycles = np.arange(2, 2 + len(q_window))
            slope, intercept, _, _, _ = stats.linregress(x_cycles, q_window)
            q_slope = float(slope)
            q_intercept = float(intercept)
        else:
            q_slope = 0.0
            q_intercept = q_discharge_2
    else:
        q_discharge_2 = 1.0
        q_discharge_100 = 1.0
        q_diff_100_2 = 0.0
        q_max_diff = 0.0
        q_slope = 0.0
        q_intercept = 1.0

    # Internal resistance features
    if len(ir) >= 2:
        ir_2 = float(ir[1]) if len(ir) > 1 else float(ir[0])
        ir_idx_100 = min(99, len(ir) - 1)
        ir_100 = float(ir[ir_idx_100])
        ir_diff_100_2 = ir_100 - ir_2
        ir_window = ir[1 : ir_idx_100 + 1] if len(ir) > 1 else ir[:1]
        ir_mean = float(np.mean(ir_window))
    else:
        ir_2 = 0.0
        ir_100 = 0.0
        ir_diff_100_2 = 0.0
        ir_mean = 0.0

    # Temperature features
    t_avg_mean = float(np.mean(t_avg[1:100])) if len(t_avg) > 1 else 30.0
    t_max_max = float(np.max(t_max[1:100])) if len(t_max) > 1 else 35.0
    t_min_min = float(np.min(t_min[1:100])) if len(t_min) > 1 else 25.0

    # Charge time features
    if len(chargetime) >= 2:
        ct_2 = float(chargetime[1]) if len(chargetime) > 1 else float(chargetime[0])
        ct_idx_100 = min(99, len(chargetime) - 1)
        ct_100 = float(chargetime[ct_idx_100])
        charge_time_diff_100_2 = ct_100 - ct_2
        ct_window = chargetime[1 : ct_idx_100 + 1] if len(chargetime) > 1 else chargetime[:1]
        charge_time_mean = float(np.mean(ct_window))
    else:
        charge_time_diff_100_2 = 0.0
        charge_time_mean = 10.0

    return {
        "delta_q_var": delta_q_var,
        "delta_q_min": delta_q_min,
        "delta_q_mean": delta_q_mean,
        "delta_q_skew": delta_q_skew,
        "delta_q_kurt": delta_q_kurt,
        "q_discharge_2": q_discharge_2,
        "q_discharge_100": q_discharge_100,
        "q_diff_100_2": q_diff_100_2,
        "q_max_diff": q_max_diff,
        "q_slope": q_slope,
        "q_intercept": q_intercept,
        "ir_2": ir_2,
        "ir_100": ir_100,
        "ir_diff_100_2": ir_diff_100_2,
        "ir_mean": ir_mean,
        "t_avg_mean": t_avg_mean,
        "t_max_max": t_max_max,
        "t_min_min": t_min_min,
        "charge_time_mean": charge_time_mean,
        "charge_time_diff_100_2": charge_time_diff_100_2,
    }


ADAPTER_SCHEMA_VERSION = "2.0.0"

# Feature Categorization
FEATURE_TAXONOMY: dict[str, str] = {
    # Paper-inspired Delta Q_100-10(V) variance model (Severson et al., Nature Energy 2019)
    "delta_q_var": "paper_inspired_delta_q_variance",
    "delta_q_min": "paper_inspired_delta_q_min",
    "delta_q_mean": "paper_inspired_delta_q_mean",
    "delta_q_skew": "paper_inspired_delta_q_skew",
    "delta_q_kurt": "paper_inspired_delta_q_kurtosis",
    # Early discharge capacity trajectory
    "q_discharge_2": "discharge_capacity_cycle_2",
    "q_discharge_100": "discharge_capacity_cycle_100",
    "q_diff_100_2": "discharge_capacity_diff_100_2",
    "q_max_diff": "discharge_capacity_max_diff_100_2",
    "q_slope": "discharge_capacity_linear_slope",
    "q_intercept": "discharge_capacity_linear_intercept",
    # Internal resistance trajectory
    "ir_2": "internal_resistance_cycle_2",
    "ir_100": "internal_resistance_cycle_100",
    "ir_diff_100_2": "internal_resistance_diff_100_2",
    "ir_mean": "internal_resistance_mean_100",
    # Thermal statistics
    "t_avg_mean": "temperature_average_mean_100",
    "t_max_max": "temperature_max_peak_100",
    "t_min_min": "temperature_min_valley_100",
    # Charging time statistics
    "charge_time_mean": "charge_time_mean_100",
    "charge_time_diff_100_2": "charge_time_diff_100_2",
}


class SeversonAdapter(DatasetAdapter):
    """Adapter for Severson 2019 early-life battery cycle life prediction benchmark.

    Prediction-only benchmark:
    - Entity identity: physical battery cell (124 physical cells).
    - Design optimization: Not applicable (closed historical early-prediction benchmark).
    - Feature horizon: strictly 100 cycles.
    """

    ADAPTER_SCHEMA_VERSION = ADAPTER_SCHEMA_VERSION

    def __init__(
        self,
        raw_dir: Path | None = None,
        processed_dir: Path | None = None,
        raw_manifest_path: Path | None = None,
        horizon: int = 100,
    ) -> None:
        project_root = Path(__file__).resolve().parent.parent.parent
        self.raw_dir = raw_dir or project_root / "data" / "external" / "severson_2019" / "raw"
        self.processed_dir = (
            processed_dir or project_root / "data" / "external" / "severson_2019" / "processed"
        )
        self.raw_manifest_path = (
            raw_manifest_path or project_root / "data" / "external" / "severson_2019" / "manifest.json"
        )
        self.horizon = horizon

        self._spec = DatasetSpec(
            name="severson",
            id_column="physical_cell_id",
            entity_id_column="physical_cell_id",
            candidate_id_column=None,
            feature_columns=list(SEVERSON_FEATURE_COLUMNS),
            target_column="cycle_life",
            objective="maximize",
            candidate_columns=[],
            pre_experiment_features=[],
            post_experiment_characterization=list(SEVERSON_FEATURE_COLUMNS),
            targets=["cycle_life"],
            constraints=[],
            candidate_variables=[],
            supports_prediction=True,
            supports_optimization=False,
            split_group_columns=["physical_cell_id"],
            oracle_columns=list(SEVERSON_ORACLE_COLUMNS),
            feature_horizon=horizon,
            source_dataset="severson_2019",
            source_version="nature_energy_2019",
        )

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    def load(self, force_recompute: bool = False) -> pd.DataFrame:
        from src.datasets.cache import validate_processed_cache, write_processed_manifest

        processed_file = self.processed_dir / "cells.csv"
        is_cache_valid = validate_processed_cache(
            processed_dir=self.processed_dir,
            raw_manifest_path=self.raw_manifest_path,
            adapter_schema_version=self.ADAPTER_SCHEMA_VERSION,
            feature_horizon=self.horizon,
            expected_files=["cells.csv"],
        )

        if is_cache_valid and not force_recompute:
            return pd.read_csv(processed_file)

        reconstructed_cells, split_map = reconstruct_severson_cells(self.raw_dir)
        rows: list[dict[str, Any]] = []

        for cell_id, cell_data in reconstructed_cells.items():
            # 1. Truncate strictly at horizon=100
            truncated = truncate_to_horizon(cell_data, horizon=self.horizon)

            # 2. Extract features strictly from truncated cell
            features = extract_features(truncated)

            batch_prefix = cell_id[:2]
            batch_id = "batch1" if batch_prefix == "b1" else ("batch2" if batch_prefix == "b2" else "batch3")

            row = {
                "physical_cell_id": cell_id,
                "batch_id": batch_id,
                "original_cell_id": cell_id,
                "charging_policy": str(cell_data.get("policy", "")),
                "split": split_map.get(cell_id, "train"),
                "cycle_life": float(cell_data.get("cycle_life", 0.0)),
                **features,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_file, index=False)

        write_processed_manifest(
            processed_dir=self.processed_dir,
            dataset="severson",
            source_version="nature_energy_2019",
            raw_manifest_path=self.raw_manifest_path,
            adapter_schema_version=self.ADAPTER_SCHEMA_VERSION,
            feature_horizon=self.horizon,
            processed_files=[processed_file],
        )
        return df

