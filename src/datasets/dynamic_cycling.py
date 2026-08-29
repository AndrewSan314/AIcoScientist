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

ADAPTER_SCHEMA_VERSION = "2.0.0"

DYNAMIC_CYCLING_ORACLE_COLUMNS: list[str] = [
    "efc_lifetime",
    "target_mean",
    "target_std",
    "n_replicates",
    "cycles_90",
]


def load_raw_dynamic_cycling_data(
    raw_dir: Path,
    replicate_design_tolerance: float = 0.20,
    expected_records: int | None = 92,
    expected_protocols: int | None = 47,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw Dynamic Cycling 2024 data with explicit ID alignment and replicate validation.

    - 92 physical cells across 47 discharge protocols.
    - Pre-experiment design features: 8 waveform descriptors.
    - Target: EFC lifetime to 90% SOH.
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

    # 2. Load protocol features with explicit cell_name key
    if protocol_features_pkl.exists():
        proto_feat_raw = pd.read_pickle(protocol_features_pkl)
        if "cell_name" not in proto_feat_raw.columns:
            if proto_feat_raw.index.name == "cell_name" or set(metadata["cell_name"]).issubset(set(proto_feat_raw.index)):
                proto_feat_raw = proto_feat_raw.reset_index()
                if "index" in proto_feat_raw.columns:
                    proto_feat_raw = proto_feat_raw.rename(columns={"index": "cell_name"})
            else:
                proto_feat_raw = proto_feat_raw.copy()
                proto_feat_raw["cell_name"] = metadata["cell_name"].values
        proto_feat = proto_feat_raw.rename(columns=RAW_FEATURE_COLUMN_MAP)
    else:
        proto_feat_df = pd.read_csv(protocol_features_csv)
        if "cell_name" in proto_feat_df.columns:
            proto_feat = proto_feat_df.rename(columns=RAW_FEATURE_COLUMN_MAP)
        elif "Unnamed: 0" in proto_feat_df.columns and proto_feat_df["Unnamed: 0"].iloc[0].startswith("cell_"):
            proto_feat = proto_feat_df.rename(columns={"Unnamed: 0": "cell_name", **RAW_FEATURE_COLUMN_MAP})
        else:
            # Validate invariant correspondence with metadata before assigning cell_name
            if "Average Current" in proto_feat_df.columns and "avg_crate_exp" in metadata.columns:
                diff = np.max(np.abs(metadata["avg_crate_exp"].to_numpy() - proto_feat_df["Average Current"].to_numpy()))
                if diff > 1e-4:
                    raise ValueError(f"protocol_features.csv does not align with metadata.pkl (avg current max diff: {diff})")
            proto_feat = proto_feat_df.rename(columns=RAW_FEATURE_COLUMN_MAP).copy()
            proto_feat["cell_name"] = metadata["cell_name"].values

    # 3. Load SOH90 targets
    soh90 = pd.read_csv(soh90_path)
    if "cell_name" not in soh90.columns:
        if "Unnamed: 0" in soh90.columns:
            soh90 = soh90.rename(columns={"Unnamed: 0": "cell_name"})
        else:
            raise ValueError("soh90.csv must contain 'cell_name' or cell identifier column")

    # 4. Explicit Join on cell_name
    cells_df = metadata[["cell_name", "protocol_type", "protocol_variant", "protocol_name"]].copy()
    cells_df = cells_df.rename(columns={"cell_name": "cell_id", "protocol_name": "protocol_id"})

    feat_cols_to_merge = ["cell_name", *[c for c in DYNAMIC_CYCLING_FEATURE_COLUMNS if c in proto_feat.columns]]
    proto_feat_subset = proto_feat[feat_cols_to_merge].rename(columns={"cell_name": "cell_id"})
    cells_df = pd.merge(cells_df, proto_feat_subset, on="cell_id", how="inner")

    target_cols_to_merge = ["cell_name", "EFCs (with Diagnostic)", "Cycles"]
    soh90_subset = soh90[target_cols_to_merge].rename(
        columns={
            "cell_name": "cell_id",
            "EFCs (with Diagnostic)": "efc_lifetime",
            "Cycles": "cycles_90",
        }
    )
    cells_df = pd.merge(cells_df, soh90_subset, on="cell_id", how="inner")

    # Invariant assertions
    if expected_records is not None and len(cells_df) != expected_records:
        raise ValueError(f"Expected exactly {expected_records} aligned cell records, got {len(cells_df)}")
    n_protocols = cells_df["protocol_id"].nunique()
    if expected_protocols is not None and n_protocols != expected_protocols:
        raise ValueError(f"Expected exactly {expected_protocols} unique protocol IDs, got {n_protocols}")


    # Determine replicate IDs (A, B, ...)
    cells_df["replicate_id"] = cells_df.groupby("protocol_id").cumcount().apply(lambda x: chr(ord("A") + x))

    # 5. Validate replicate design coordinates across replicates
    for proto_id, group in cells_df.groupby("protocol_id"):
        for feat in DYNAMIC_CYCLING_FEATURE_COLUMNS:
            feat_vals = group[feat].to_numpy(dtype=float)
            spread = float(np.ptp(feat_vals))
            if spread > replicate_design_tolerance:
                raise ValueError(
                    f"Replicate cells for protocol {proto_id!r} have conflicting design coordinates for {feat!r}: "
                    f"spread={spread:.5f} exceeds tolerance {replicate_design_tolerance}"
                )

    # 6. Build protocol-level data (Canonical design coordinates from first replicate + aggregated target)
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
        # Canonical first design vector (NOT averaged across features)
        for feat in DYNAMIC_CYCLING_FEATURE_COLUMNS:
            prow[feat] = float(first_row[feat])

        proto_rows.append(prow)

    protocols_df = pd.DataFrame(proto_rows).sort_values("protocol_id").reset_index(drop=True)

    # Unify cell-level nominal design coordinates with protocol canonical design vector
    canonical_lookup = protocols_df[["protocol_id", *DYNAMIC_CYCLING_FEATURE_COLUMNS]].set_index("protocol_id")
    for feat in DYNAMIC_CYCLING_FEATURE_COLUMNS:
        cells_df[feat] = cells_df["protocol_id"].map(canonical_lookup[feat]).values.astype(float)

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

    def load(self, force_recompute: bool = False) -> pd.DataFrame:
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
            if self.level == "protocol":
                return pd.read_csv(protocols_file)
            return pd.read_csv(cells_file)

        cells_df, protocols_df = load_raw_dynamic_cycling_data(self.raw_dir)
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

        if self.level == "protocol":
            return protocols_df
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

    def load_protocols(self) -> pd.DataFrame:
        protocols_file = self.processed_dir / "protocols.csv"
        if protocols_file.exists():
            return pd.read_csv(protocols_file)
        _, protocols_df = load_raw_dynamic_cycling_data(self.raw_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        protocols_df.to_csv(protocols_file, index=False)
        return protocols_df

    def load_cells(self) -> pd.DataFrame:
        cells_file = self.processed_dir / "cells.csv"
        if cells_file.exists():
            return pd.read_csv(cells_file)
        cells_df, _ = load_raw_dynamic_cycling_data(self.raw_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        cells_df.to_csv(cells_file, index=False)
        return cells_df

    def load_candidate_pool(self) -> pd.DataFrame:
        """Returns the full candidate pool containing ONLY protocol_id and design features (zero oracle data)."""
        protocols_df = self.load_protocols()
        cand_cols = ["protocol_id", *self.spec.candidate_columns]
        return protocols_df[cand_cols].copy().reset_index(drop=True)

    def load_hidden_oracle(self) -> pd.DataFrame:
        """Returns the hidden oracle dataset with protocol targets and replicate metrics."""
        return self.load_protocols()

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

