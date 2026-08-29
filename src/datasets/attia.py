from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.datasets.base import DatasetAdapter, DatasetSpec

logger = logging.getLogger(__name__)

ATTIA_FEATURE_COLUMNS: list[str] = ["C1", "C2", "C3", "C4"]
ATTIA_CANDIDATE_COLUMNS: list[str] = ["C1", "C2", "C3", "C4"]
ATTIA_CANDIDATE_ID_COLUMN: str = "policy_id"
ADAPTER_SCHEMA_VERSION = "1.0.0"

ATTIA_ORACLE_COLUMNS: list[str] = [
    "simulated_lifetime",
]

C4_MIN_LIMIT = 0.1
C4_MAX_LIMIT = 4.81


def compute_expected_c4(c1: float | np.ndarray, c2: float | np.ndarray, c3: float | np.ndarray) -> float | np.ndarray:
    """Computes C4 from 10-minute total charging constraint across 4 equal 20% SOC steps.

    Total charge time: t = 0.2/C1 + 0.2/C2 + 0.2/C3 + 0.2/C4 = 1/6 hr (10 min).
    C4 = 0.2 / (1/6 - (0.2/C1 + 0.2/C2 + 0.2/C3)).
    """
    denom = (1.0 / 6.0) - (0.2 / c1 + 0.2 / c2 + 0.2 / c3)
    if isinstance(denom, np.ndarray):
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(denom > 0, 0.2 / denom, np.nan)
        return res
    if denom <= 0:
        return float("nan")
    return 0.2 / denom


def load_raw_attia_policies(
    raw_policies_file: Path | str,
    expected_policies: int | None = 224,
) -> pd.DataFrame:
    """Loads and validates raw candidate policies from Attia et al. 2020 policies_all.csv.

    Validation requirements:
    1. Exactly 4 numeric columns (C1, C2, C3, C4).
    2. All values must be finite.
    3. Exactly expected_policies rows (default 224 unique valid policies).
    4. C4 satisfies the 10-minute charging constraint within numerical tolerance (atol=1e-3).
    5. C4 obeys valid limits [0.1, 4.81].
    6. Excludes the baseline (4.8, 4.8, 4.8, 4.8).
    7. Assigns deterministic policy IDs: ATTIA_P000, ATTIA_P001, ...
    """
    raw_path = Path(raw_policies_file)
    if not raw_path.exists():
        raise FileNotFoundError(f"Attia policies file not found at: {raw_path}")

    # Read CSV (headerless comma-separated numbers)
    raw_df = pd.read_csv(raw_path, header=None)
    if raw_df.shape[1] != 4:
        raise ValueError(
            f"Expected exactly 4 columns for Attia policies (C1, C2, C3, C4), got {raw_df.shape[1]}"
        )

    raw_df.columns = ["C1", "C2", "C3", "C4"]

    # Check finite numeric values
    for col in ATTIA_FEATURE_COLUMNS:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
        if raw_df[col].isna().any():
            invalid_rows = raw_df[raw_df[col].isna()].index.tolist()
            raise ValueError(f"Attia policies column {col!r} contains non-numeric or NaN values at rows: {invalid_rows}")
        if not np.all(np.isfinite(raw_df[col].to_numpy(dtype=float))):
            invalid_rows = raw_df[~np.isfinite(raw_df[col])].index.tolist()
            raise ValueError(f"Attia policies column {col!r} contains non-finite values at rows: {invalid_rows}")

    # Check duplicates
    if raw_df.duplicated().any():
        dup_rows = raw_df[raw_df.duplicated()].index.tolist()
        raise ValueError(f"Attia policies file contains duplicate policy definitions at rows: {dup_rows}")

    # Validate C4 calculation and limits
    c1 = raw_df["C1"].to_numpy(dtype=float)
    c2 = raw_df["C2"].to_numpy(dtype=float)
    c3 = raw_df["C3"].to_numpy(dtype=float)
    c4 = raw_df["C4"].to_numpy(dtype=float)

    expected_c4 = compute_expected_c4(c1, c2, c3)
    if not np.all(np.isfinite(expected_c4)):
        invalid_mask = ~np.isfinite(expected_c4)
        raise ValueError(f"Attia policies contain invalid C1-C3 combinations resulting in non-positive denominator for C4 at rows: {np.where(invalid_mask)[0].tolist()}")

    # Tolerance check (author CSV stores C4 rounded to 3 decimal places)
    c4_diff = np.abs(c4 - expected_c4)
    max_c4_diff = float(np.max(c4_diff))
    if max_c4_diff > 1e-3:
        bad_idx = int(np.argmax(c4_diff))
        raise ValueError(
            f"Attia policy at row {bad_idx} violates C4 charging constraint: "
            f"actual={c4[bad_idx]:.4f}, expected={expected_c4[bad_idx]:.4f} (diff={max_c4_diff:.5f} > 1e-3)"
        )

    # Check bounds
    if np.any(c4 < C4_MIN_LIMIT) or np.any(c4 > C4_MAX_LIMIT):
        out_of_bounds = np.where((c4 < C4_MIN_LIMIT) | (c4 > C4_MAX_LIMIT))[0].tolist()
        raise ValueError(f"Attia policies contain C4 values outside [{C4_MIN_LIMIT}, {C4_MAX_LIMIT}] at rows: {out_of_bounds}")

    # Check exclusion of baseline (4.8, 4.8, 4.8, 4.8)
    baseline_mask = (c1 == 4.8) & (c2 == 4.8) & (c3 == 4.8)
    if np.any(baseline_mask):
        baseline_rows = np.where(baseline_mask)[0].tolist()
        raise ValueError(f"Attia policies must exclude the baseline policy (4.8, 4.8, 4.8, 4.8) at rows: {baseline_rows}")

    if expected_policies is not None and len(raw_df) != expected_policies:
        raise ValueError(
            f"Expected exactly {expected_policies} valid Attia policies, found {len(raw_df)}"
        )

    # Assign deterministic policy IDs
    policy_ids = [f"ATTIA_P{i:03d}" for i in range(len(raw_df))]
    raw_df.insert(0, "policy_id", policy_ids)

    return raw_df[["policy_id", "C1", "C2", "C3", "C4"]].copy()


class AttiaAdapter(DatasetAdapter):
    """Dataset adapter for Attia et al. 2020 fast-charging closed-loop optimization benchmark."""

    ADAPTER_SCHEMA_VERSION = ADAPTER_SCHEMA_VERSION

    def __init__(
        self,
        raw_policies_file: Path | str | None = None,
        processed_dir: Path | str | None = None,
        raw_manifest_path: Path | str | None = None,
        expected_policies: int | None = 224,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.raw_policies_file = Path(
            raw_policies_file
            or project_root
            / "data"
            / "external"
            / "attia_2020"
            / "data"
            / "policies_all.csv"
        )
        self.processed_dir = Path(
            processed_dir
            or project_root
            / "data"
            / "external"
            / "attia_2020"
            / "processed"
        )
        self.raw_manifest_path = Path(
            raw_manifest_path
            or project_root
            / "data"
            / "external"
            / "attia_2020"
            / "manifest.json"
        )
        self.expected_policies = expected_policies

        self._spec = DatasetSpec(
            name="attia",
            id_column="policy_id",
            candidate_id_column="policy_id",
            feature_columns=list(ATTIA_FEATURE_COLUMNS),
            target_column="simulated_lifetime",
            objective="maximize",
            candidate_columns=list(ATTIA_CANDIDATE_COLUMNS),
            supports_prediction=False,
            supports_optimization=True,
            split_group_columns=["policy_id"],
            oracle_columns=list(ATTIA_ORACLE_COLUMNS),
            source_dataset="attia_2020",
            source_version="nature_2020",
        )

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    def _ensure_processed(self, force_recompute: bool = False) -> pd.DataFrame:
        """Ensures valid processed cache exists, recomputing from raw policies if missing or invalidated."""
        from src.datasets.cache import validate_processed_cache, write_processed_manifest

        policies_file = self.processed_dir / "policies.csv"

        is_cache_valid = validate_processed_cache(
            processed_dir=self.processed_dir,
            raw_manifest_path=self.raw_manifest_path,
            adapter_schema_version=self.ADAPTER_SCHEMA_VERSION,
            feature_horizon=None,
            expected_files=["policies.csv"],
        )

        if is_cache_valid and not force_recompute:
            policies_df = pd.read_csv(policies_file)
            exp_policies = self.expected_policies if self.expected_policies is not None else 0
            if exp_policies == 0 or len(policies_df) == exp_policies:
                return policies_df
            logger.warning(
                "Attia cache row count mismatch (%d vs expected %d); recomputing.",
                len(policies_df),
                exp_policies,
            )

        logger.info("Parsing raw Attia policies from %s", self.raw_policies_file)
        policies_df = load_raw_attia_policies(
            raw_policies_file=self.raw_policies_file,
            expected_policies=self.expected_policies,
        )

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        policies_df.to_csv(policies_file, index=False)

        write_processed_manifest(
            processed_dir=self.processed_dir,
            dataset="attia",
            source_version="nature_2020",
            raw_manifest_path=self.raw_manifest_path,
            adapter_schema_version=self.ADAPTER_SCHEMA_VERSION,
            feature_horizon=None,
            processed_files=[policies_file],
        )

        return policies_df

    def load_policies(self, force_recompute: bool = False) -> pd.DataFrame:
        return self._ensure_processed(force_recompute=force_recompute)

    def load(self, force_recompute: bool = False) -> pd.DataFrame:
        return self.load_policies(force_recompute=force_recompute)

    def candidate_space(self, observed: pd.DataFrame) -> pd.DataFrame:
        """Returns unseen candidate pool containing ONLY policy_id and design features."""
        policies_df = self.load_policies()
        cand_cols = ["policy_id", *self.spec.candidate_columns]

        if observed.empty:
            return policies_df[cand_cols].drop_duplicates().reset_index(drop=True)

        observed_policies = set()
        if "policy_id" in observed.columns:
            observed_policies.update(observed["policy_id"].dropna().astype(str))
        if self.spec.candidate_id_column and self.spec.candidate_id_column in observed.columns:
            observed_policies.update(observed[self.spec.candidate_id_column].dropna().astype(str))

        unseen = policies_df[~policies_df["policy_id"].astype(str).isin(observed_policies)].copy()
        return unseen[cand_cols].drop_duplicates().reset_index(drop=True)

    def load_candidate_pool(self, force_recompute: bool = False) -> pd.DataFrame:
        """Returns the full candidate pool containing ONLY policy_id and design features (zero oracle data)."""
        policies_df = self.load_policies(force_recompute=force_recompute)
        cand_cols = ["policy_id", *self.spec.candidate_columns]
        return policies_df[cand_cols].copy().reset_index(drop=True)

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

    def continuous_search_space(self) -> Any:
        """Returns the generic constrained continuous SearchSpace for Attia 4-step fast charging."""
        from src.optimization.search_space import (
            Constraint,
            ContinuousVariable,
            DerivedVariable,
            SearchSpace,
        )

        variables = [
            ContinuousVariable("C1", lower=3.6, upper=8.0),
            ContinuousVariable("C2", lower=3.6, upper=7.0),
            ContinuousVariable("C3", lower=3.6, upper=5.6),
        ]

        def _calc_c4(c: Mapping[str, Any]) -> float:
            c1, c2, c3 = float(c["C1"]), float(c["C2"]), float(c["C3"])
            return float(compute_expected_c4(c1, c2, c3))

        derived_variables = [
            DerivedVariable("C4", compute_fn=_calc_c4, depends_on=("C1", "C2", "C3")),
        ]

        constraints = [
            Constraint(
                name="positive_c4_denominator",
                predicate=lambda c: bool((1.0 / 6.0) - (0.2 / float(c["C1"]) + 0.2 / float(c["C2"]) + 0.2 / float(c["C3"])) > 0),
                description="10-minute charging constraint denominator must be strictly positive",
            ),
            Constraint(
                name="c4_bounds",
                predicate=lambda c: bool(C4_MIN_LIMIT <= float(c.get("C4", _calc_c4(c))) <= C4_MAX_LIMIT),
                description=f"C4 charging rate must lie in [{C4_MIN_LIMIT}, {C4_MAX_LIMIT}]",
            ),
            Constraint(
                name="exclude_baseline",
                predicate=lambda c: not (
                    np.isclose(float(c["C1"]), 4.8, atol=1e-3)
                    and np.isclose(float(c["C2"]), 4.8, atol=1e-3)
                    and np.isclose(float(c["C3"]), 4.8, atol=1e-3)
                ),
                description="Exclude baseline protocol (4.8, 4.8, 4.8, 4.8)",
            ),
        ]

        return SearchSpace(
            variables=variables,
            derived_variables=derived_variables,
            constraints=constraints,
            name="attia_continuous_fast_charging",
        )

    def build_observed_row(
        self,
        candidate: Mapping[str, Any],
        response: Any,
        step: int,
    ) -> dict[str, Any]:
        target = response.target if hasattr(response, "target") else response.get("target")
        metadata = response.metadata if hasattr(response, "metadata") else response.get("metadata", {})
        cand_dict = dict(candidate)

        policy_id = cand_dict.get("policy_id", metadata.get("candidate_id", f"policy_{step}"))
        row = {
            self.spec.id_column: f"step_{step}_{policy_id}",
            "policy_id": policy_id,
            **{k: cand_dict[k] for k in self.spec.candidate_columns if k in cand_dict},
            self.spec.target_column: target,
        }
        return row


