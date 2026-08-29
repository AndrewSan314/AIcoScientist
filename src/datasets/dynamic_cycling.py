from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.datasets.base import DatasetAdapter, DatasetSpec

logger = logging.getLogger(__name__)

DYNAMIC_CYCLING_FEATURE_COLUMNS: list[str] = [
    "average_current",
    "normalized_current_variance",
    "maximum_discharge_current",
    "relative_charge_fraction",
    "rest_fraction_at_high_soc",
    "rest_soc",
    "peak_frequency_1",
    "peak_frequency_2",
]

RAW_FEATURE_COLUMN_MAP: dict[str, str] = {
    "Average Current": "average_current",
    "Normalized Current Variance": "normalized_current_variance",
    "Maximum Discharge Current": "maximum_discharge_current",
    "Relative Charge Fraction": "relative_charge_fraction",
    "Rest Fraction at High SOC": "rest_fraction_at_high_soc",
    "Rest SOC": "rest_soc",
    "Peak Frequency 1": "peak_frequency_1",
    "Peak Frequency 2": "peak_frequency_2",
}

ADAPTER_SCHEMA_VERSION = "3.1.0"

DYNAMIC_CYCLING_ORACLE_COLUMNS: list[str] = [
    "efc_lifetime",
    "target_mean",
    "target_std",
    "n_replicates",
    "cycles_90",
]


def compute_replicate_feature_differences(
    cells_df: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Computes empirical pairwise replicate differences across the dataset and returns a summary DataFrame.

    Realized waveform features from cycler logs show slight experimental replicate variation.
    This function reports max, median, 95th percentile, and mean absolute pairwise replicate difference per feature.
    """
    from itertools import combinations

    feature_cols = [c for c in DYNAMIC_CYCLING_FEATURE_COLUMNS if c in cells_df.columns]
    diff_rows: list[dict[str, Any]] = []

    for proto_id, grp in cells_df.groupby("protocol_id"):
        if len(grp) < 2:
            continue
        c_ids = grp["cell_id"].tolist()
        for c1, c2 in combinations(c_ids, 2):
            row1 = grp[grp["cell_id"] == c1].iloc[0]
            row2 = grp[grp["cell_id"] == c2].iloc[0]
            entry = {"protocol_id": proto_id, "cell_1": c1, "cell_2": c2}
            for f in feature_cols:
                entry[f] = abs(float(row1[f]) - float(row2[f]))
            diff_rows.append(entry)

    diff_df = pd.DataFrame(diff_rows)
    summary_rows: list[dict[str, Any]] = []

    for f in feature_cols:
        if not diff_df.empty and f in diff_df.columns:
            vals = diff_df[f].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "feature": f,
                    "max_abs_difference": float(np.max(vals)),
                    "median_abs_difference": float(np.median(vals)),
                    "p95_abs_difference": float(np.percentile(vals, 95)),
                    "mean_abs_difference": float(np.mean(vals)),
                }
            )
        else:
            summary_rows.append(
                {
                    "feature": f,
                    "max_abs_difference": 0.0,
                    "median_abs_difference": 0.0,
                    "p95_abs_difference": 0.0,
                    "mean_abs_difference": 0.0,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
    return summary_df


def load_raw_dynamic_cycling_data(
    raw_dir: Path,
    expected_records: int | None = None,
    expected_protocols: int | None = None,
    feature_tolerances: Mapping[str, float] | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw Dynamic Cycling 2024 data with deterministic ID alignment and strict numeric integrity.

    Scientific Feature Semantics:
    - The 8 waveform features extracted in `protocol_features.pkl` are *realized per-cell measurements*
      derived from each physical battery cell's actual cycler current/voltage time-series.
    - Cell-level data (`cells_df`) retains each cell's realized features to preserve empirical within-protocol variance.
    - Protocol-level data (`protocols_df`) aggregates realized features across replicates of that protocol using arithmetic mean.

    Numeric Integrity Contracts:
    - All 8 design features must be finite (no NaN, Inf).
    - `efc_lifetime` and `cycles_90` must be finite and strictly > 0.
    - Deterministic ID set equality across metadata, features, and targets is enforced before merge.
    """
    metadata_path = raw_dir / "metadata.pkl"
    protocol_features_pkl = raw_dir / "protocol_features.pkl"
    protocol_features_csv = raw_dir / "protocol_features.csv"
    soh90_path = raw_dir / "soh90.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.pkl at {metadata_path}")
    if not (protocol_features_pkl.exists() or protocol_features_csv.exists()):
        raise FileNotFoundError(f"Missing protocol_features at {raw_dir}")
    if not soh90_path.exists():
        raise FileNotFoundError(f"Missing soh90.csv at {soh90_path}")

    # 1. Load metadata
    metadata = pd.read_pickle(metadata_path)
    if "cell_name" not in metadata.columns or "protocol_name" not in metadata.columns:
        raise ValueError("metadata.pkl is missing required 'cell_name' or 'protocol_name' columns")

    if metadata["cell_name"].isnull().any() or (metadata["cell_name"] == "").any():
        raise ValueError("metadata.pkl contains null or empty cell_name entries")
    if not metadata["cell_name"].is_unique:
        raise ValueError("metadata.pkl contains duplicate cell_name entries")

    metadata_ids = set(metadata["cell_name"].astype(str))

    # 2. Load protocol features with explicit deterministic cell_name identifier
    if protocol_features_pkl.exists():
        proto_feat_raw = pd.read_pickle(protocol_features_pkl)
        if "cell_name" in proto_feat_raw.columns:
            proto_feat = proto_feat_raw.rename(columns=RAW_FEATURE_COLUMN_MAP)
        elif proto_feat_raw.index.name == "cell_name" or (
            len(proto_feat_raw.index) > 0
            and isinstance(proto_feat_raw.index[0], str)
            and proto_feat_raw.index[0].startswith("cell_")
        ):
            proto_feat = proto_feat_raw.reset_index().rename(columns={"index": "cell_name", **RAW_FEATURE_COLUMN_MAP})
        else:
            raise ValueError(
                "protocol_features.pkl is missing explicit deterministic 'cell_name' column or index. "
                "Positional or feature-matching inference is strictly prohibited."
            )
    else:
        proto_feat_df = pd.read_csv(protocol_features_csv)
        if "cell_name" in proto_feat_df.columns:
            proto_feat = proto_feat_df.rename(columns=RAW_FEATURE_COLUMN_MAP)
        elif "Unnamed: 0" in proto_feat_df.columns and str(proto_feat_df["Unnamed: 0"].iloc[0]).startswith("cell_"):
            proto_feat = proto_feat_df.rename(columns={"Unnamed: 0": "cell_name", **RAW_FEATURE_COLUMN_MAP})
        else:
            raise ValueError(
                "protocol_features.csv is missing explicit deterministic 'cell_name' identifier column or index. "
                "Positional or feature-matching inference is strictly prohibited."
            )

    if proto_feat["cell_name"].isnull().any() or (proto_feat["cell_name"] == "").any():
        raise ValueError("protocol_features contains null or empty cell_name entries")
    if not proto_feat["cell_name"].is_unique:
        raise ValueError("protocol_features contains duplicate cell_name entries")

    feature_ids = set(proto_feat["cell_name"].astype(str))

    # 3. Load SOH90 targets
    soh90 = pd.read_csv(soh90_path)
    if "cell_name" in soh90.columns:
        soh90_df = soh90.copy()
    elif "Unnamed: 0" in soh90.columns and str(soh90["Unnamed: 0"].iloc[0]).startswith("cell_"):
        soh90_df = soh90.rename(columns={"Unnamed: 0": "cell_name"})
    else:
        raise ValueError("soh90.csv must contain explicit deterministic 'cell_name' identifier column")

    if soh90_df["cell_name"].isnull().any() or (soh90_df["cell_name"] == "").any():
        raise ValueError("soh90.csv contains null or empty cell_name entries")
    if not soh90_df["cell_name"].is_unique:
        raise ValueError("soh90.csv contains duplicate cell_name entries")

    target_ids = set(soh90_df["cell_name"].astype(str))

    # 4. Strict Set Equality Verification BEFORE Merge (No subset / silent inner join reduction)
    if metadata_ids != feature_ids:
        diff_meta = metadata_ids - feature_ids
        diff_feat = feature_ids - metadata_ids
        raise ValueError(
            f"Mismatch between metadata cell IDs and protocol feature cell IDs: "
            f"metadata only: {diff_meta}, features only: {diff_feat}"
        )

    if metadata_ids != target_ids:
        diff_meta = metadata_ids - target_ids
        diff_tgt = target_ids - metadata_ids
        raise ValueError(
            f"Mismatch between metadata cell IDs and SOH90 target cell IDs: "
            f"metadata only: {diff_meta}, targets only: {diff_tgt}"
        )

    if expected_records is not None and len(metadata_ids) != expected_records:
        raise ValueError(
            f"Expected exactly {expected_records} unique cell records, got {len(metadata_ids)}"
        )

    # 5. Explicit Join on deterministic cell_name key
    cells_df = metadata[["cell_name", "protocol_type", "protocol_variant", "protocol_name"]].copy()
    cells_df = cells_df.rename(columns={"cell_name": "cell_id", "protocol_name": "protocol_id"})

    feat_cols_to_merge = ["cell_name", *[c for c in DYNAMIC_CYCLING_FEATURE_COLUMNS if c in proto_feat.columns]]
    proto_feat_subset = proto_feat[feat_cols_to_merge].rename(columns={"cell_name": "cell_id"})
    cells_df = pd.merge(cells_df, proto_feat_subset, on="cell_id", how="inner")

    target_cols_to_merge = ["cell_name", "EFCs (with Diagnostic)", "Cycles"]
    soh90_subset = soh90_df[target_cols_to_merge].rename(
        columns={
            "cell_name": "cell_id",
            "EFCs (with Diagnostic)": "efc_lifetime",
            "Cycles": "cycles_90",
        }
    )
    cells_df = pd.merge(cells_df, soh90_subset, on="cell_id", how="inner")

    if len(cells_df) != len(metadata_ids):
        raise ValueError(
            f"Merged cell records ({len(cells_df)}) do not match unique metadata ID count ({len(metadata_ids)})"
        )

    # Determine replicate IDs (A, B, ...)
    cells_df["replicate_id"] = cells_df.groupby("protocol_id").cumcount().apply(lambda x: chr(ord("A") + x))

    # 6. Numeric Integrity Validation
    for feat in DYNAMIC_CYCLING_FEATURE_COLUMNS:
        if feat not in cells_df.columns:
            raise ValueError(f"Missing required design feature column {feat!r}")
        feat_vals = pd.to_numeric(cells_df[feat], errors="coerce").to_numpy(dtype=float)
        non_finite = ~np.isfinite(feat_vals)
        if non_finite.any():
            bad_cells = cells_df.loc[non_finite, "cell_id"].tolist()
            bad_vals = cells_df.loc[non_finite, feat].tolist()
            raise ValueError(
                f"Non-finite values detected in design feature {feat!r} for cell IDs {bad_cells}: {bad_vals}"
            )
        cells_df[feat] = feat_vals

    for tgt_col in ["efc_lifetime", "cycles_90"]:
        if tgt_col not in cells_df.columns:
            raise ValueError(f"Missing required target column {tgt_col!r}")
        tgt_vals = pd.to_numeric(cells_df[tgt_col], errors="coerce").to_numpy(dtype=float)
        invalid = (~np.isfinite(tgt_vals)) | (tgt_vals <= 0)
        if invalid.any():
            bad_cells = cells_df.loc[invalid, "cell_id"].tolist()
            bad_vals = cells_df.loc[invalid, tgt_col].tolist()
            raise ValueError(
                f"Invalid target values detected in {tgt_col!r} (must be finite and > 0) for cell IDs {bad_cells}: {bad_vals}"
            )
        cells_df[tgt_col] = tgt_vals

    # Optional explicit tolerance check if user passes custom feature_tolerances
    if feature_tolerances is not None or rtol is not None or atol is not None:
        for proto_id, group in cells_df.groupby("protocol_id"):
            if len(group) <= 1:
                continue
            canon_row = group.iloc[0]
            for rep_idx in range(1, len(group)):
                cand_row = group.iloc[rep_idx]
                for feat in DYNAMIC_CYCLING_FEATURE_COLUMNS:
                    c_val = float(canon_row[feat])
                    cand_val = float(cand_row[feat])
                    abs_diff = abs(cand_val - c_val)

                    if rtol is not None or atol is not None:
                        eff_atol = atol if atol is not None else 1e-8
                        eff_rtol = rtol if rtol is not None else 1e-5
                        is_close = np.isclose(cand_val, c_val, rtol=eff_rtol, atol=eff_atol, equal_nan=False)
                    else:
                        feat_tol = feature_tolerances.get(feat, 1e-6) if feature_tolerances else 1e-6
                        is_close = abs_diff <= feat_tol

                    if not is_close:
                        raise ValueError(
                            f"Protocol {proto_id!r} replicate conflict on feature {feat!r}: "
                            f"canonical={c_val}, conflicting={cand_val}, abs_diff={abs_diff:.6e}"
                        )

    # 7. Build protocol-level data (Design coordinates aggregated as mean across realized replicates)
    proto_rows: list[dict[str, Any]] = []
    for proto_id, group in cells_df.groupby("protocol_id", sort=False):
        first_row = group.iloc[0]
        targets = group["efc_lifetime"].to_numpy(dtype=float)
        n_reps = len(targets)
        t_mean = float(np.mean(targets))
        t_std = float(np.std(targets, ddof=1)) if n_reps > 1 else 0.0

        prow = {
            "protocol_id": proto_id,
            "protocol_type": first_row["protocol_type"],
            "protocol_variant": first_row["protocol_variant"],
            "n_replicates": n_reps,
            "target_mean": t_mean,
            "target_std": t_std,
            "efc_lifetime": t_mean,
            "cycles_90": float(group["cycles_90"].mean()),
        }
        for feat in DYNAMIC_CYCLING_FEATURE_COLUMNS:
            prow[feat] = float(group[feat].mean())

        proto_rows.append(prow)

    protocols_df = pd.DataFrame(proto_rows).sort_values("protocol_id").reset_index(drop=True)

    cell_cols = [
        "cell_id",
        "protocol_id",
        "replicate_id",
        "protocol_type",
        "protocol_variant",
        "efc_lifetime",
        "cycles_90",
        *DYNAMIC_CYCLING_FEATURE_COLUMNS,
    ]
    cells_df = cells_df[cell_cols].sort_values(["protocol_id", "replicate_id"]).reset_index(drop=True)

    # Hard Invariant assertions on final datasets
    if expected_records is not None:
        if len(cells_df) != expected_records:
            raise ValueError(f"Expected exactly {expected_records} cell records, got {len(cells_df)}")
        if cells_df["cell_id"].nunique() != expected_records:
            raise ValueError(f"Expected {expected_records} unique cell IDs, got {cells_df['cell_id'].nunique()}")

    n_protocols = cells_df["protocol_id"].nunique()
    if expected_protocols is not None:
        if n_protocols != expected_protocols:
            raise ValueError(f"Expected exactly {expected_protocols} unique protocol IDs in cells, got {n_protocols}")
        if len(protocols_df) != expected_protocols:
            raise ValueError(f"Expected exactly {expected_protocols} rows in protocols_df, got {len(protocols_df)}")
        if protocols_df["protocol_id"].nunique() != expected_protocols:
            raise ValueError(f"Expected {expected_protocols} unique protocol IDs in protocols_df, got {protocols_df['protocol_id'].nunique()}")

    return cells_df, protocols_df


class DynamicCyclingAdapter(DatasetAdapter):
    """Adapter for Dynamic Cycling 2024 protocol optimization and offline BO benchmark."""

    ADAPTER_SCHEMA_VERSION = ADAPTER_SCHEMA_VERSION

    def __init__(
        self,
        raw_dir: Path | None = None,
        processed_dir: Path | None = None,
        raw_manifest_path: Path | None = None,
        level: str = "cell",
        expected_records: int | None = 92,
        expected_protocols: int | None = 47,
    ) -> None:
        project_root = Path(__file__).resolve().parent.parent.parent
        self.raw_dir = (
            raw_dir
            or project_root
            / "data"
            / "external"
            / "dynamic_cycling_2024"
            / "paper_code"
            / "data"
        )
        self.processed_dir = (
            processed_dir
            or project_root
            / "data"
            / "external"
            / "dynamic_cycling_2024"
            / "processed"
        )
        self.raw_manifest_path = (
            raw_manifest_path
            or project_root
            / "data"
            / "external"
            / "dynamic_cycling_2024"
            / "manifest.json"
        )
        self.level = level
        self.expected_records = expected_records
        self.expected_protocols = expected_protocols

        id_col = "protocol_id" if level == "protocol" else "cell_id"
        target_col = "target_mean" if level == "protocol" else "efc_lifetime"

        self._spec = DatasetSpec(
            name="dynamic_cycling",
            id_column=id_col,
            entity_id_column="cell_id",
            candidate_id_column="protocol_id",
            feature_columns=list(DYNAMIC_CYCLING_FEATURE_COLUMNS),
            target_column=target_col,
            objective="maximize",
            candidate_columns=list(DYNAMIC_CYCLING_FEATURE_COLUMNS),
            supports_prediction=True,
            supports_optimization=True,
            split_group_columns=["protocol_id"],
            oracle_columns=list(DYNAMIC_CYCLING_ORACLE_COLUMNS),
            source_dataset="dynamic_cycling_2024",
            source_version="nat_energy_2024",
        )

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    def _ensure_processed(self, force_recompute: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Canonical internal method ensuring cache metadata validity before returning processed data."""
        from src.datasets.cache import validate_processed_cache, write_processed_manifest

        cells_file = self.processed_dir / "cells.csv"
        protocols_file = self.processed_dir / "protocols.csv"

        is_cache_valid = validate_processed_cache(
            processed_dir=self.processed_dir,
            raw_manifest_path=self.raw_manifest_path,
            adapter_schema_version=self.ADAPTER_SCHEMA_VERSION,
            feature_horizon=None,
            expected_files=["cells.csv", "protocols.csv"],
        )

        if is_cache_valid and not force_recompute:
            cells_df = pd.read_csv(cells_file)
            protocols_df = pd.read_csv(protocols_file)
            exp_rec = self.expected_records if self.expected_records is not None else 0
            exp_proto = self.expected_protocols if self.expected_protocols is not None else 0
            if (exp_rec == 0 or len(cells_df) == exp_rec) and (exp_proto == 0 or len(protocols_df) == exp_proto):
                return cells_df, protocols_df

        cells_df, protocols_df = load_raw_dynamic_cycling_data(
            self.raw_dir,
            expected_records=self.expected_records,
            expected_protocols=self.expected_protocols,
        )
        if self.expected_records is not None:
            if len(cells_df) != self.expected_records or cells_df["cell_id"].nunique() != self.expected_records:
                raise ValueError(
                    f"Dynamic cycling adapter invariant violation: expected {self.expected_records} unique cells, "
                    f"got len={len(cells_df)}, nunique={cells_df['cell_id'].nunique()}"
                )
        if self.expected_protocols is not None:
            if (
                len(protocols_df) != self.expected_protocols
                or protocols_df["protocol_id"].nunique() != self.expected_protocols
                or cells_df["protocol_id"].nunique() != self.expected_protocols
            ):
                raise ValueError(
                    f"Dynamic cycling adapter invariant violation: expected {self.expected_protocols} unique protocols, "
                    f"got len(protocols_df)={len(protocols_df)}, "
                    f"nunique(protocols)={protocols_df['protocol_id'].nunique()}, "
                    f"nunique(cells_protocol)={cells_df['protocol_id'].nunique()}"
                )

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        cells_df.to_csv(cells_file, index=False)
        protocols_df.to_csv(protocols_file, index=False)

        write_processed_manifest(
            processed_dir=self.processed_dir,
            dataset="dynamic_cycling",
            source_version="nat_energy_2024",
            raw_manifest_path=self.raw_manifest_path,
            adapter_schema_version=self.ADAPTER_SCHEMA_VERSION,
            feature_horizon=None,
            processed_files=[cells_file, protocols_file],
        )
        return cells_df, protocols_df

    def load(self, force_recompute: bool = False) -> pd.DataFrame:
        cells_df, protocols_df = self._ensure_processed(force_recompute=force_recompute)
        if self.level == "protocol":
            return protocols_df
        return cells_df

    def load_protocols(self, force_recompute: bool = False) -> pd.DataFrame:
        _, protocols_df = self._ensure_processed(force_recompute=force_recompute)
        return protocols_df

    def load_cells(self, force_recompute: bool = False) -> pd.DataFrame:
        cells_df, _ = self._ensure_processed(force_recompute=force_recompute)
        return cells_df

    def candidate_space(self, observed: pd.DataFrame) -> pd.DataFrame:
        """Returns unseen candidate pool containing ONLY protocol_id and design features."""
        protocols_df = self.load_protocols()
        cand_cols = ["protocol_id", *self.spec.candidate_columns]

        if observed.empty:
            return protocols_df[cand_cols].drop_duplicates().reset_index(drop=True)

        observed_protocols = set()
        if "protocol_id" in observed.columns:
            observed_protocols.update(observed["protocol_id"].dropna().astype(str))
        if self.spec.candidate_id_column and self.spec.candidate_id_column in observed.columns:
            observed_protocols.update(observed[self.spec.candidate_id_column].dropna().astype(str))

        unseen = protocols_df[~protocols_df["protocol_id"].astype(str).isin(observed_protocols)].copy()
        return unseen[cand_cols].drop_duplicates().reset_index(drop=True)

    def load_candidate_pool(self, force_recompute: bool = False) -> pd.DataFrame:
        """Returns the full candidate pool containing ONLY protocol_id and design features (zero oracle data)."""
        protocols_df = self.load_protocols(force_recompute=force_recompute)
        cand_cols = ["protocol_id", *self.spec.candidate_columns]
        return protocols_df[cand_cols].copy().reset_index(drop=True)

    def load_hidden_oracle(self, force_recompute: bool = False) -> pd.DataFrame:
        """Returns the hidden cell-level oracle dataset (92 replicate cell rows) for replicate-aware querying."""
        return self.load_cells(force_recompute=force_recompute)

    def build_candidate_features(
        self,
        candidates: pd.DataFrame,
        observed: pd.DataFrame,
        fill_values: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=candidates.index)
        for feat in self.spec.feature_columns:
            if feat in candidates.columns:
                result[feat] = candidates[feat]
            else:
                raise ValueError(f"Candidate dataframe missing design feature {feat!r}")
        return result[self.spec.feature_columns]

    def build_observed_row(
        self,
        candidate: Mapping[str, Any],
        response: Any,
        step: int,
    ) -> dict[str, Any]:
        target = response.target if hasattr(response, "target") else response.get("target")
        metadata = response.metadata if hasattr(response, "metadata") else response.get("metadata", {})
        cand_dict = dict(candidate)

        protocol_id = cand_dict.get("protocol_id", metadata.get("candidate_id", f"protocol_{step}"))
        row = {
            self.spec.id_column: f"step_{step}_{protocol_id}",
            "protocol_id": protocol_id,
            **{k: cand_dict[k] for k in self.spec.candidate_columns if k in cand_dict},
            self.spec.target_column: target,
        }
        return row


