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

DYNAMIC_CYCLING_ORACLE_COLUMNS: list[str] = [
    "efc_lifetime",
    "target_mean",
    "target_std",
    "n_replicates",
    "cycles_90",
]


def load_raw_dynamic_cycling_data(
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw data and generates cell-level and protocol-level representations.

    - 92 total physical cells across 47 discharge protocols.
    - Pre-experiment design features: 8 waveform descriptors known before cycling starts.
    - Target: EFC lifetime to 90% SOH.
    """
    metadata_path = raw_dir / "metadata.pkl"
    protocol_features_path = raw_dir / "protocol_features.csv"
    soh90_path = raw_dir / "soh90.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.pkl at {metadata_path}")
    if not protocol_features_path.exists():
        raise FileNotFoundError(f"Missing protocol_features.csv at {protocol_features_path}")
    if not soh90_path.exists():
        raise FileNotFoundError(f"Missing soh90.csv at {soh90_path}")

    # Load raw data
    metadata = pd.read_pickle(metadata_path)
    proto_feat = pd.read_csv(protocol_features_path)
    soh90 = pd.read_csv(soh90_path, index_col=0)

    # Rename design features to standard snake_case
    proto_feat_clean = proto_feat.rename(columns=RAW_FEATURE_COLUMN_MAP)[DYNAMIC_CYCLING_FEATURE_COLUMNS]

    # Combine cell-level data
    cells_df = pd.concat([metadata.reset_index(drop=True), proto_feat_clean.reset_index(drop=True)], axis=1)
    cells_df["cell_id"] = cells_df["cell_name"]
    cells_df["protocol_id"] = cells_df["protocol_name"]

    # Align target from soh90 by matching cell_id
    cells_df = cells_df.set_index("cell_id")
    cells_df["efc_lifetime"] = soh90["EFCs (with Diagnostic)"]
    cells_df["cycles_90"] = soh90["Cycles"]
    cells_df = cells_df.reset_index()

    # Determine replicate IDs within each protocol
    cells_df["replicate_id"] = cells_df.groupby("protocol_id").cumcount().apply(lambda x: chr(ord("A") + x))

    # Reorder cell-level columns
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

    # Build protocol-level data (aggregated replicate ground truth)
    protocol_groups = cells_df.groupby("protocol_id", as_index=False)
    
    # Feature coordinates (deterministic mean of design coordinates per protocol)
    agg_dict: dict[str, Any] = {
        "protocol_type": "first",
        "protocol_variant": "first",
        "efc_lifetime": ["count", "mean", "std"],
        "cycles_90": "mean",
    }
    for feat in DYNAMIC_CYCLING_FEATURE_COLUMNS:
        agg_dict[feat] = "mean"

    protocols_agg = cells_df.groupby("protocol_id").agg(agg_dict)
    
    # Flatten MultiIndex columns
    protocols_df = pd.DataFrame()
    protocols_df["protocol_id"] = protocols_agg.index
    protocols_df["protocol_type"] = protocols_agg[("protocol_type", "first")].values
    protocols_df["protocol_variant"] = protocols_agg[("protocol_variant", "first")].values
    protocols_df["n_replicates"] = protocols_agg[("efc_lifetime", "count")].values.astype(int)
    protocols_df["target_mean"] = protocols_agg[("efc_lifetime", "mean")].values.astype(float)
    protocols_df["target_std"] = protocols_agg[("efc_lifetime", "std")].fillna(0.0).values.astype(float)
    protocols_df["efc_lifetime"] = protocols_df["target_mean"]
    protocols_df["cycles_90"] = protocols_agg[("cycles_90", "mean")].values.astype(float)

    for feat in DYNAMIC_CYCLING_FEATURE_COLUMNS:
        protocols_df[feat] = protocols_agg[(feat, "mean")].values.astype(float)

    protocols_df = protocols_df.reset_index(drop=True)

    # Ensure cell-level design features match the nominal protocol-level design coordinates exactly
    proto_feat_lookup = protocols_df[["protocol_id", *DYNAMIC_CYCLING_FEATURE_COLUMNS]].set_index("protocol_id")
    for feat in DYNAMIC_CYCLING_FEATURE_COLUMNS:
        cells_df[feat] = cells_df["protocol_id"].map(proto_feat_lookup[feat]).values.astype(float)

    return cells_df, protocols_df



class DynamicCyclingAdapter(DatasetAdapter):
    """Adapter for Dynamic Cycling 2024 protocol optimization and offline BO benchmark."""

    def __init__(
        self,
        raw_dir: Path | None = None,
        processed_dir: Path | None = None,
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
            split_group_columns=["protocol_id"],
            oracle_columns=list(DYNAMIC_CYCLING_ORACLE_COLUMNS),
            source_dataset="dynamic_cycling_2024",
            source_version="nat_energy_2024",
        )

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    def load(self, force_recompute: bool = False) -> pd.DataFrame:
        cells_file = self.processed_dir / "cells.csv"
        protocols_file = self.processed_dir / "protocols.csv"

        if cells_file.exists() and protocols_file.exists() and not force_recompute:
            if self.level == "protocol":
                return pd.read_csv(protocols_file)
            return pd.read_csv(cells_file)

        cells_df, protocols_df = load_raw_dynamic_cycling_data(self.raw_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        cells_df.to_csv(cells_file, index=False)
        protocols_df.to_csv(protocols_file, index=False)

        if self.level == "protocol":
            return protocols_df
        return cells_df

    def candidate_space(self, observed: pd.DataFrame) -> pd.DataFrame:
        protocols_df = self.load_protocols()
        if observed.empty:
            return protocols_df[
                ["protocol_id", *self.spec.candidate_columns]
            ].drop_duplicates().reset_index(drop=True)

        observed_protocols = set()
        if "protocol_id" in observed.columns:
            observed_protocols.update(observed["protocol_id"].dropna().astype(str))
        if self.spec.candidate_id_column and self.spec.candidate_id_column in observed.columns:
            observed_protocols.update(observed[self.spec.candidate_id_column].dropna().astype(str))

        # Filter candidates to unseen protocols only
        unseen = protocols_df[~protocols_df["protocol_id"].astype(str).isin(observed_protocols)].copy()
        return unseen[
            ["protocol_id", *self.spec.candidate_columns]
        ].drop_duplicates().reset_index(drop=True)

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
