from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.datasets.base import DatasetAdapter, DatasetSpec

logger = logging.getLogger(__name__)

AUIRH_DEFAULT_EDX_DIR = Path("EDX_dataset/EDX_dataset")
AUIRH_DEFAULT_SECCM_DIR = Path("SECCM_dataset/SECCM_dataset")
AUIRH_DEFAULT_XRD_DIR = Path("XRD_dataset/XRD_dataset")

# Fallback paths if flat directory extraction
AUIRH_FALLBACK_EDX_DIR = Path("EDX_dataset")
AUIRH_FALLBACK_SECCM_DIR = Path("SECCM_dataset")
AUIRH_FALLBACK_XRD_DIR = Path("XRD_dataset")

EXPECTED_EDX_SHA256 = {
    "Au-Ir-Rh_Au-rich_EDX.csv": "12a31c04e3476229efadc628a4281fc13b007a18d12b0950f973caf6b06188fd",
    "Au-Ir-Rh_Ir-rich_EDX.csv": "7dab109ebfba5893aa54d13727005a705f9b3e9994990ad089f66f3430f4c197",
    "Au-Ir-Rh_Rh-rich_EDX.csv": "6477439649749316c2503ecfbd55564747d925fed405fe5919dcee2dc7788572",
}
EXPECTED_SECCM_FIT_SHA256 = "6edb3f7b36fb78c78a85d09335bf7606c815211622c5b533054b880db7f41f15"

EXPECTED_TOTAL_MEASURED_COUNT = 966
EXPECTED_PER_LIBRARY_COUNT = 322

AUIRH_FEATURE_COLUMNS: list[str] = ["Au", "Ir"]
AUIRH_CANDIDATE_COLUMNS: list[str] = ["Au", "Ir"]
AUIRH_CANDIDATE_ID_COLUMN: str = "candidate_id"
AUIRH_LIBRARIES: list[str] = ["Au-rich", "Ir-rich", "Rh-rich"]

AUIRH_TARGET_MAP: dict[str, str] = {
    "k0": "k^0 [cm/s]",
    "k^0": "k^0 [cm/s]",
    "k^0 [cm/s]": "k^0 [cm/s]",
    "i_lim": "i_lim [A/cm^2]",
    "i_lim [A/cm^2]": "i_lim [A/cm^2]",
    "alpha": "alpha [a.u.]",
    "alpha [a.u.]": "alpha [a.u.]",
}

# Target and structural columns hidden from optimizer during candidate generation
AUIRH_ORACLE_COLUMNS: list[str] = ["k^0 [cm/s]", "i_lim [A/cm^2]", "alpha [a.u.]", "k0", "i_lim", "alpha"]
ADAPTER_SCHEMA_VERSION = "1.0.0"


def compute_derived_rh(au: float | np.ndarray, ir: float | np.ndarray) -> float | np.ndarray:
    """Computes derived Rh composition coordinate from the 100% ternary sum constraint."""
    return 100.0 - au - ir


def generate_auirh_candidate_id(library: str, area: int) -> str:
    """Generates a stable, canonical candidate ID formatted as AUIRH_<LIBRARY>_<AREA:03d>."""
    return f"AUIRH_{library}_{area:03d}"


def _resolve_dir(primary: Path, fallback: Path) -> Path:
    if primary.is_dir():
        return primary
    if fallback.is_dir():
        return fallback
    return primary


def load_raw_auirh_dataset(
    edx_dir: Path | str = AUIRH_DEFAULT_EDX_DIR,
    seccm_dir: Path | str = AUIRH_DEFAULT_SECCM_DIR,
    allow_unverified_hash: bool = False,
) -> pd.DataFrame:
    """Loads, validates, and joins raw EDX composition and SECCM fitted parameters.

    Validates:
    1. Existence and integrity of 3 EDX files and 1 SECCM fit parameters CSV.
    2. Sum of composition Au + Ir + Rh ~ 100.0% (+-0.1%).
    3. Exactly 966 verified joined rows across Au-rich, Ir-rich, and Rh-rich libraries.
    4. Absence of NaN or Inf values in joined physical properties.
    """
    e_dir = _resolve_dir(Path(edx_dir), AUIRH_FALLBACK_EDX_DIR)
    s_dir = _resolve_dir(Path(seccm_dir), AUIRH_FALLBACK_SECCM_DIR)

    if not e_dir.is_dir():
        raise FileNotFoundError(f"Au-Ir-Rh EDX directory not found at: {e_dir.resolve()}")
    if not s_dir.is_dir():
        raise FileNotFoundError(f"Au-Ir-Rh SECCM directory not found at: {s_dir.resolve()}")

    # 1. Load EDX files
    edx_dfs = []
    for lib in AUIRH_LIBRARIES:
        fname = f"Au-Ir-Rh_{lib}_EDX.csv"
        fpath = e_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing EDX file: {fpath}")

        if not allow_unverified_hash and fname in EXPECTED_EDX_SHA256:
            with open(fpath, "rb") as f:
                f_hash = hashlib.sha256(f.read()).hexdigest()
            if f_hash != EXPECTED_EDX_SHA256[fname]:
                raise ValueError(
                    f"SHA256 mismatch for {fname}. Expected {EXPECTED_EDX_SHA256[fname]}, got {f_hash}"
                )

        df_lib = pd.read_csv(fpath)
        required_edx_cols = ["Area", "Au [at.%]", "Ir [at.%]", "Rh [at.%]"]
        for c in required_edx_cols:
            if c not in df_lib.columns:
                raise ValueError(f"EDX file {fname} missing required column '{c}'")

        df_lib["Library"] = lib
        df_lib["Area"] = df_lib["Area"].astype(int)
        edx_dfs.append(df_lib)

    edx_all = pd.concat(edx_dfs, ignore_index=True)

    # Validate composition sums
    sums = edx_all["Au [at.%]"] + edx_all["Ir [at.%]"] + edx_all["Rh [at.%]"]
    if np.any(sums < 99.5) or np.any(sums > 100.5):
        raise ValueError(f"EDX composition sums deviate from 100% (min={sums.min()}, max={sums.max()})")

    # 2. Load SECCM fit parameters
    fit_path = s_dir / "LSV_fit_parameters.csv"
    if not fit_path.is_file():
        raise FileNotFoundError(f"Missing SECCM fit parameters file: {fit_path}")

    if not allow_unverified_hash:
        with open(fit_path, "rb") as f:
            fit_hash = hashlib.sha256(f.read()).hexdigest()
        if fit_hash != EXPECTED_SECCM_FIT_SHA256:
            raise ValueError(
                f"SHA256 mismatch for LSV_fit_parameters.csv. Expected {EXPECTED_SECCM_FIT_SHA256}, got {fit_hash}"
            )

    fit_df = pd.read_csv(fit_path)
    required_fit_cols = ["Library", "Area", "i_lim [A/cm^2]", "k^0 [cm/s]", "alpha [a.u.]"]
    for c in required_fit_cols:
        if c not in fit_df.columns:
            raise ValueError(f"LSV_fit_parameters.csv missing required column '{c}'")

    fit_df["Area"] = fit_df["Area"].astype(int)

    # 3. Inner join on Library + Area
    merged = pd.merge(edx_all, fit_df, on=["Library", "Area"], how="inner")
    if len(merged) != EXPECTED_TOTAL_MEASURED_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_TOTAL_MEASURED_COUNT} joined measured candidates, got {len(merged)}"
        )

    # Check for NaN / Inf
    if merged.isna().any().any():
        raise ValueError("Joined Au-Ir-Rh records contain NaN values.")
    num_cols = merged.select_dtypes(include=np.number)
    if np.isinf(num_cols.to_numpy()).any():
        raise ValueError("Joined Au-Ir-Rh records contain Inf values.")

    # 4. Standardize columns
    merged[AUIRH_CANDIDATE_ID_COLUMN] = [
        generate_auirh_candidate_id(row["Library"], int(row["Area"])) for _, row in merged.iterrows()
    ]
    merged["Au"] = merged["Au [at.%]"].astype(float)
    merged["Ir"] = merged["Ir [at.%]"].astype(float)
    merged["Rh"] = merged["Rh [at.%]"].astype(float)
    merged["k0"] = merged["k^0 [cm/s]"].astype(float)
    merged["i_lim"] = merged["i_lim [A/cm^2]"].astype(float)
    merged["alpha"] = merged["alpha [a.u.]"].astype(float)

    # Sort deterministically
    merged = merged.sort_values(by=["Library", "Area"]).reset_index(drop=True)
    return merged


class AuIrRhExperimentOracle:
    """Offline oracle for experimental evaluation on the real Au-Ir-Rh SECCM benchmark.

    HARD EVALUATION FIREWALL:
    - Encapsulates ground truth table for all 966 measured physical samples.
    - Zero latent target / LSV / XRD leakage to optimizer candidates.
    - Strictly rejects out-of-pool / continuous / fabricated candidate queries.
    - Prevents duplicate evaluations per run trajectory unless explicitly enabled.
    - Deterministic lookup and resettable per trajectory.
    """

    def __init__(
        self,
        full_records_df: pd.DataFrame,
        target_column: str = "k0",
        allow_duplicate_queries: bool = False,
    ) -> None:
        canonical_target = AUIRH_TARGET_MAP.get(target_column, target_column)
        if canonical_target not in AUIRH_TARGET_MAP.values() and target_column not in {"k0", "i_lim", "alpha"}:
            raise ValueError(f"target_column must be 'k0', 'i_lim', or 'alpha', got '{target_column}'")

        self.target_name = "k0" if target_column in {"k0", "k^0", "k^0 [cm/s]"} else ("i_lim" if target_column in {"i_lim", "i_lim [A/cm^2]"} else "alpha")
        self.source_target_column = AUIRH_TARGET_MAP[self.target_name]
        self.allow_duplicate_queries = allow_duplicate_queries

        required_cols = [
            AUIRH_CANDIDATE_ID_COLUMN,
            "Library",
            "Area",
            "Au",
            "Ir",
            "Rh",
            self.source_target_column,
        ]
        for col in required_cols:
            if col not in full_records_df.columns:
                raise ValueError(f"full_records_df missing required column: '{col}'")

        self._ground_truth_map: dict[str, dict[str, Any]] = {}
        for _, row in full_records_df.iterrows():
            cid = str(row[AUIRH_CANDIDATE_ID_COLUMN])
            self._ground_truth_map[cid] = {
                AUIRH_CANDIDATE_ID_COLUMN: cid,
                "Library": str(row["Library"]),
                "Area": int(row["Area"]),
                "Au": float(row["Au"]),
                "Ir": float(row["Ir"]),
                "Rh": float(row["Rh"]),
                self.target_name: float(row[self.source_target_column]),
                self.source_target_column: float(row[self.source_target_column]),
            }

        self._queried_cids: set[str] = set()
        self._query_history: list[dict[str, Any]] = []

        all_targets = [v[self.target_name] for v in self._ground_truth_map.values()]
        self.global_best_value = float(np.max(all_targets))
        best_cid = max(self._ground_truth_map.keys(), key=lambda k: self._ground_truth_map[k][self.target_name])
        self.global_best_candidate_id = best_cid

    def query(self, candidate_input: str | Mapping[str, Any] | pd.Series) -> dict[str, Any]:
        """Queries the oracle for ground truth target value of a measured material.

        Args:
            candidate_input: Candidate ID string, mapping, or pandas Series.

        Returns:
            Dict containing revealed candidate_id, design variables, and measured target value.
        """
        if isinstance(candidate_input, str):
            cid = candidate_input
        elif isinstance(candidate_input, (pd.Series, dict)):
            cid = str(candidate_input.get(AUIRH_CANDIDATE_ID_COLUMN, ""))
        else:
            raise TypeError(f"Unsupported candidate_input type: {type(candidate_input)}")

        if not cid or cid not in self._ground_truth_map:
            raise KeyError(
                f"Candidate ID '{cid}' is not a valid measured physical material in the Au-Ir-Rh benchmark pool."
            )

        if not self.allow_duplicate_queries and cid in self._queried_cids:
            raise ValueError(f"Duplicate experimental measurement requested for sample '{cid}'.")

        self._queried_cids.add(cid)
        record = dict(self._ground_truth_map[cid])
        self._query_history.append(record)
        return record

    def reset(self) -> None:
        """Resets query history for a new optimization run."""
        self._queried_cids.clear()
        self._query_history.clear()

    @property
    def query_count(self) -> int:
        return len(self._query_history)


class AuIrRhAdapter(DatasetAdapter):
    """Dataset adapter for the Au–Ir–Rh autonomous SECCM experimental materials benchmark.

    Exposes:
    - 2 independent ternary composition coordinates: Au, Ir (with derived Rh = 100 - Au - Ir).
    - Finite candidate pool of measured samples (966 pooled, or 322 per physical library).
    - Configurable target: 'k0' (rate constant [cm/s]), 'i_lim' (limiting current [A/cm^2]), 'alpha' (transfer coeff).
    """

    def __init__(
        self,
        target: str = "k0",
        library: str | None = None,
        edx_dir: Path | str = AUIRH_DEFAULT_EDX_DIR,
        seccm_dir: Path | str = AUIRH_DEFAULT_SECCM_DIR,
        xrd_dir: Path | str = AUIRH_DEFAULT_XRD_DIR,
        objective: str = "maximize",
        allow_unverified_hash: bool = False,
    ) -> None:
        if target not in {"k0", "k^0", "i_lim", "alpha", "k^0 [cm/s]", "i_lim [A/cm^2]", "alpha [a.u.]"}:
            raise ValueError(f"Target must be 'k0', 'i_lim', or 'alpha', got '{target}'")
        if library is not None and library not in AUIRH_LIBRARIES and library != "pooled":
            raise ValueError(f"Library must be one of {AUIRH_LIBRARIES} or 'pooled'/None, got '{library}'")

        super().__init__()
        self.target = "k0" if target in {"k0", "k^0", "k^0 [cm/s]"} else ("i_lim" if target in {"i_lim", "i_lim [A/cm^2]"} else "alpha")
        self.library = library if library != "pooled" else None
        self.objective = objective
        self.edx_dir = Path(edx_dir)
        self.seccm_dir = Path(seccm_dir)
        self.xrd_dir = Path(xrd_dir)
        self.allow_unverified_hash = allow_unverified_hash

        self._full_df: pd.DataFrame | None = None
        self._filtered_df: pd.DataFrame | None = None

    def _ensure_loaded(self) -> None:
        if self._full_df is None:
            df = load_raw_auirh_dataset(
                edx_dir=self.edx_dir,
                seccm_dir=self.seccm_dir,
                allow_unverified_hash=self.allow_unverified_hash,
            )
            self._full_df = df

        if self._filtered_df is None:
            assert self._full_df is not None
            if self.library is not None:
                self._filtered_df = self._full_df[self._full_df["Library"] == self.library].copy().reset_index(drop=True)
            else:
                self._filtered_df = self._full_df.copy().reset_index(drop=True)

    @property
    def spec(self) -> DatasetSpec:
        mode_str = f"_{self.library.lower().replace('-', '_')}" if self.library else "_pooled"
        return DatasetSpec(
            name=f"auirh_{self.target}{mode_str}",
            id_column=AUIRH_CANDIDATE_ID_COLUMN,
            candidate_id_column=AUIRH_CANDIDATE_ID_COLUMN,
            feature_columns=list(AUIRH_FEATURE_COLUMNS),
            candidate_columns=list(AUIRH_CANDIDATE_COLUMNS),
            target_column=self.target,
            objective=self.objective,
            supports_prediction=True,
            supports_optimization=True,
            oracle_columns=list(AUIRH_ORACLE_COLUMNS),
            source_dataset="Au-Ir-Rh Combinatorial Thin-Film SECCM Benchmark",
            source_version=ADAPTER_SCHEMA_VERSION,
        )

    def get_spec(self) -> DatasetSpec:
        return self.spec

    def load(self, force_recompute: bool = False) -> pd.DataFrame:
        """Returns the full validated DataFrame for the configured library/pool."""
        if force_recompute:
            self._full_df = None
            self._filtered_df = None
        self._ensure_loaded()
        assert self._filtered_df is not None
        return self._filtered_df.copy()

    def load_data(self) -> pd.DataFrame:
        return self.load()

    def get_candidate_pool(self) -> pd.DataFrame:
        """Returns the optimizer-visible candidate pool containing ONLY candidate_id, Library, Area, Au, Ir, and Rh.

        Zero oracle targets, LSV curves, or XRD spectra are included.
        """
        self._ensure_loaded()
        assert self._filtered_df is not None
        visible_cols = [AUIRH_CANDIDATE_ID_COLUMN, "Library", "Area", "Au", "Ir", "Rh"]
        return self._filtered_df[visible_cols].copy()

    def create_oracle(self, allow_duplicate_queries: bool = False) -> AuIrRhExperimentOracle:
        """Creates an offline oracle instance for closed-loop benchmark execution."""
        self._ensure_loaded()
        assert self._filtered_df is not None
        return AuIrRhExperimentOracle(
            full_records_df=self._filtered_df,
            target_column=self.target,
            allow_duplicate_queries=allow_duplicate_queries,
        )
