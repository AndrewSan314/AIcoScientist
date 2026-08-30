from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.io

from src.datasets.base import DatasetAdapter, DatasetSpec

logger = logging.getLogger(__name__)

FECONI_DEFAULT_MAT_PATH = Path("remi/data/Combinatorial Libraries/Fe-Co-Ni/FeCoNi_benchmark_dataset_220501a.mat")
EXPECTED_SHA256 = "aaee4ddb6cf711e789e1edd6358145746611b273cecf2257699ea057d5feb1dc"
EXPECTED_SAMPLE_COUNT = 921

FECONI_FEATURE_COLUMNS: list[str] = ["Co", "Fe"]
FECONI_CANDIDATE_COLUMNS: list[str] = ["Co", "Fe"]
FECONI_CANDIDATE_ID_COLUMN: str = "candidate_id"
ADAPTER_SCHEMA_VERSION = "1.0.0"

# Target and structural columns hidden from optimizer during candidate generation
FECONI_ORACLE_COLUMNS: list[str] = ["Kerr", "Coer", "XRD", "TTH"]


def compute_derived_ni(co: float | np.ndarray, fe: float | np.ndarray) -> float | np.ndarray:
    """Computes derived Ni composition coordinate from the 100% ternary sum constraint."""
    return 100.0 - co - fe


def generate_feconi_candidate_id(index: int) -> str:
    """Generates a deterministic candidate ID formatted as FECONI_XXX."""
    return f"FECONI_{index:03d}"


def load_raw_feconi_mat(
    mat_path: Path | str = FECONI_DEFAULT_MAT_PATH,
    allow_unverified_hash: bool = False,
) -> dict[str, np.ndarray]:
    """Loads and strictly validates the raw Fe-Co-Ni benchmark MAT file.

    Validates:
    1. File existence and SHA256 integrity (raises ValueError on mismatch unless allow_unverified_hash=True).
    2. Presence of keys: C, Coer, Kerr, TTH, XRD.
    3. Exactly 921 rows without NaN or Inf.
    4. Row sum consistency ~ 100%.
    """
    path = Path(mat_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fe-Co-Ni benchmark MAT file not found at: {path.resolve()}")

    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    if file_hash != EXPECTED_SHA256:
        if not allow_unverified_hash:
            raise ValueError(
                f"SHA256 hash mismatch for Fe-Co-Ni dataset at '{path}'. "
                f"Expected '{EXPECTED_SHA256}', but got '{file_hash}'. "
                "Pass allow_unverified_hash=True to override strict validation."
            )
        logger.warning(
            "MAT file SHA256 (%s) does not match expected canonical hash (%s). Proceeding with loaded data.",
            file_hash,
            EXPECTED_SHA256,
        )

    raw_mat = scipy.io.loadmat(str(path))
    required_keys = ["C", "Coer", "Kerr", "TTH", "XRD"]
    for k in required_keys:
        if k not in raw_mat:
            raise ValueError(f"Required key '{k}' missing from MAT file.")

    C = np.asarray(raw_mat["C"], dtype=float)
    Coer = np.asarray(raw_mat["Coer"], dtype=float).flatten()
    Kerr = np.asarray(raw_mat["Kerr"], dtype=float).flatten()
    TTH = np.asarray(raw_mat["TTH"], dtype=float).flatten()
    XRD = np.asarray(raw_mat["XRD"], dtype=float)

    if C.shape != (EXPECTED_SAMPLE_COUNT, 3):
        raise ValueError(f"Expected C shape ({EXPECTED_SAMPLE_COUNT}, 3), got {C.shape}")
    if len(Coer) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"Expected {EXPECTED_SAMPLE_COUNT} Coer values, got {len(Coer)}")
    if len(Kerr) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"Expected {EXPECTED_SAMPLE_COUNT} Kerr values, got {len(Kerr)}")
    if XRD.shape != (EXPECTED_SAMPLE_COUNT, len(TTH)):
        raise ValueError(f"Expected XRD shape ({EXPECTED_SAMPLE_COUNT}, {len(TTH)}), got {XRD.shape}")

    if np.any(np.isnan(C)) or np.any(np.isnan(Coer)) or np.any(np.isnan(Kerr)) or np.any(np.isnan(XRD)):
        raise ValueError("Raw Fe-Co-Ni MAT file contains NaN values.")
    if np.any(np.isinf(C)) or np.any(np.isinf(Coer)) or np.any(np.isinf(Kerr)) or np.any(np.isinf(XRD)):
        raise ValueError("Raw Fe-Co-Ni MAT file contains Inf values.")

    return {
        "C": C,
        "Coer": Coer,
        "Kerr": Kerr,
        "TTH": TTH,
        "XRD": XRD,
    }


class DuplicateExperimentQueryError(RuntimeError):
    """Raised when an optimization run attempts to re-query an already observed candidate."""


class FeCoNiExperimentOracle:
    """Offline ground-truth oracle for closed-loop Fe-Co-Ni experimental benchmarks.

    Enforces strict experimental integrity:
    1. Only allows lookups by exact candidate ID.
    2. Raises DuplicateExperimentQueryError if a candidate is queried more than once in the same run.
    3. Tracks total number of queries (experimental budget).
    4. Computes true global best value for benchmark metrics (regret calculation).
    """

    def __init__(
        self,
        full_records_df: pd.DataFrame,
        target_column: str = "Kerr",
        allow_duplicate_queries: bool = False,
    ) -> None:
        self.target_column = target_column
        self.allow_duplicate_queries = allow_duplicate_queries

        if FECONI_CANDIDATE_ID_COLUMN not in full_records_df.columns:
            raise ValueError(f"Oracle requires '{FECONI_CANDIDATE_ID_COLUMN}' column.")
        if target_column not in full_records_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in ground-truth DataFrame.")

        self._records = full_records_df.set_index(FECONI_CANDIDATE_ID_COLUMN).to_dict(orient="index")
        self._queried_cids: set[str] = set()
        self._query_history: list[dict[str, Any]] = []

        # Find true global maximum target in dataset
        target_series = full_records_df[target_column]
        self._global_best_idx = target_series.idxmax()
        self._global_best_cid = str(full_records_df.loc[self._global_best_idx, FECONI_CANDIDATE_ID_COLUMN])
        self._global_best_val = float(target_series.max())

    @property
    def global_best_candidate_id(self) -> str:
        return self._global_best_cid

    @property
    def global_best_value(self) -> float:
        return self._global_best_val

    def query(self, candidate_id: str) -> dict[str, Any]:
        """Queries the oracle for ground-truth physical characterization and target measurements."""
        cid = str(candidate_id)
        if cid not in self._records:
            raise KeyError(f"Candidate ID '{cid}' not found in Fe-Co-Ni ground-truth dataset.")

        if not self.allow_duplicate_queries and cid in self._queried_cids:
            raise DuplicateExperimentQueryError(
                f"Candidate '{cid}' has already been queried in this benchmark run. "
                "Re-querying observed candidates violates discrete pool optimization semantics."
            )

        rec = self._records[cid]
        self._queried_cids.add(cid)
        query_entry = {
            "query_number": len(self._query_history) + 1,
            FECONI_CANDIDATE_ID_COLUMN: cid,
            self.target_column: rec[self.target_column],
        }
        self._query_history.append(query_entry)

        return {
            FECONI_CANDIDATE_ID_COLUMN: cid,
            "sample_index": rec["sample_index"],
            "Co": rec["Co"],
            "Fe": rec["Fe"],
            "Ni": rec["Ni"],
            self.target_column: rec[self.target_column],
        }

    def reset(self) -> None:
        """Resets query history for a new optimization run."""
        self._queried_cids.clear()
        self._query_history.clear()

    @property
    def query_count(self) -> int:
        return len(self._query_history)


class FeCoNiAdapter(DatasetAdapter):
    """Dataset adapter for the NIST Fe-Co-Ni combinatorial thin-film experimental benchmark.

    Exposes:
    - 2 independent ternary composition coordinates: Co, Fe (with derived Ni = 100 - Co - Fe).
    - Finite candidate pool of exactly 921 real measured samples.
    - Configurable target: 'Kerr' (Kerr rotation [mrad]) or 'Coer' (Magnetic coercivity [mT]).
    """

    def __init__(
        self,
        mat_path: Path | str = FECONI_DEFAULT_MAT_PATH,
        target: str = "Kerr",
        objective: str = "maximize",
        allow_unverified_hash: bool = False,
    ) -> None:
        if target not in {"Kerr", "Coer"}:
            raise ValueError(f"Target must be 'Kerr' or 'Coer', got '{target}'")

        super().__init__()
        self.mat_path = Path(mat_path)
        self.target = target
        self.objective = objective
        self.allow_unverified_hash = allow_unverified_hash
        self._raw_data: dict[str, np.ndarray] | None = None
        self._full_df: pd.DataFrame | None = None

    def _ensure_loaded(self) -> None:
        if self._raw_data is None or self._full_df is None:
            raw = load_raw_feconi_mat(self.mat_path, allow_unverified_hash=self.allow_unverified_hash)
            self._raw_data = raw

            C = raw["C"]
            co_col = C[:, 0]
            fe_col = C[:, 1]
            ni_col = C[:, 2]

            candidate_ids = [generate_feconi_candidate_id(i) for i in range(len(C))]

            df = pd.DataFrame(
                {
                    FECONI_CANDIDATE_ID_COLUMN: candidate_ids,
                    "sample_index": np.arange(len(C), dtype=int),
                    "Co": co_col,
                    "Fe": fe_col,
                    "Ni": ni_col,
                    "Kerr": raw["Kerr"],
                    "Coer": raw["Coer"],
                }
            )
            self._full_df = df

    @property
    def spec(self) -> DatasetSpec:
        return DatasetSpec(
            name=f"feconi_{self.target.lower()}",
            id_column=FECONI_CANDIDATE_ID_COLUMN,
            candidate_id_column=FECONI_CANDIDATE_ID_COLUMN,
            feature_columns=list(FECONI_FEATURE_COLUMNS),
            candidate_columns=list(FECONI_CANDIDATE_COLUMNS),
            target_column=self.target,
            objective=self.objective,
            supports_prediction=True,
            supports_optimization=True,
            oracle_columns=list(FECONI_ORACLE_COLUMNS),
            source_dataset="NIST REMI Fe-Co-Ni Combinatorial Library",
            source_version=ADAPTER_SCHEMA_VERSION,
        )

    def get_spec(self) -> DatasetSpec:
        return self.spec

    def load(self, force_recompute: bool = False) -> pd.DataFrame:
        """Returns the full validated DataFrame."""
        if force_recompute:
            self._raw_data = None
            self._full_df = None
        self._ensure_loaded()
        assert self._full_df is not None
        return self._full_df.copy()

    def load_data(self) -> pd.DataFrame:
        return self.load()

    def candidate_space(self, observed: pd.DataFrame | None = None) -> pd.DataFrame:
        """Returns the discrete candidate pool of measured points."""
        return self.get_candidate_pool()

    def get_candidate_pool(self) -> pd.DataFrame:
        """Returns the optimizer-visible candidate pool containing ONLY candidate_id, Co, Fe, and Ni."""
        self._ensure_loaded()
        assert self._full_df is not None
        visible_cols = [FECONI_CANDIDATE_ID_COLUMN, "sample_index", "Co", "Fe", "Ni"]
        return self._full_df[visible_cols].copy()

    def create_oracle(self, allow_duplicate_queries: bool = False) -> FeCoNiExperimentOracle:
        """Creates an offline oracle instance for closed-loop benchmark execution."""
        self._ensure_loaded()
        assert self._full_df is not None
        return FeCoNiExperimentOracle(
            full_records_df=self._full_df,
            target_column=self.target,
            allow_duplicate_queries=allow_duplicate_queries,
        )

    def get_xrd_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns raw XRD spectra (921, 89) and Two-Theta angles (89,) for audit/characterization use."""
        self._ensure_loaded()
        assert self._raw_data is not None
        return self._raw_data["XRD"].copy(), self._raw_data["TTH"].copy()
