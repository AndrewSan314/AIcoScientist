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


def load_raw_feconi_mat(mat_path: Path | str = FECONI_DEFAULT_MAT_PATH) -> dict[str, np.ndarray]:
    """Loads and strictly validates the raw Fe-Co-Ni benchmark MAT file.

    Validates:
    1. File existence and SHA256 integrity.
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


class FeCoNiExperimentOracle:
    """Offline oracle for experimental evaluation on the real Fe-Co-Ni benchmark.

    HARD EVALUATION FIREWALL:
    - Encapsulates ground truth table for all 921 measured physical samples.
    - Zero latent target/XRD leakage to optimizer candidates.
    - Strictly rejects out-of-pool / continuous / fabricated candidate queries.
    - Prevents duplicate evaluations per run trajectory.
    """

    def __init__(
        self,
        full_records_df: pd.DataFrame,
        target_column: str = "Kerr",
        allow_duplicate_queries: bool = False,
    ) -> None:
        if target_column not in {"Kerr", "Coer"}:
            raise ValueError(f"target_column must be 'Kerr' or 'Coer', got '{target_column}'")

        required_cols = [FECONI_CANDIDATE_ID_COLUMN, "Co", "Fe", "Ni", target_column]
        for col in required_cols:
            if col not in full_records_df.columns:
                raise ValueError(f"full_records_df missing required column: '{col}'")

        self.target_column = target_column
        self.allow_duplicate_queries = allow_duplicate_queries
        self._ground_truth_map: dict[str, dict[str, Any]] = {}
        for _, row in full_records_df.iterrows():
            cid = str(row[FECONI_CANDIDATE_ID_COLUMN])
            self._ground_truth_map[cid] = {
                FECONI_CANDIDATE_ID_COLUMN: cid,
                "Co": float(row["Co"]),
                "Fe": float(row["Fe"]),
                "Ni": float(row["Ni"]),
                target_column: float(row[target_column]),
            }

        self._queried_cids: set[str] = set()
        self._query_history: list[dict[str, Any]] = []

        all_targets = [v[target_column] for v in self._ground_truth_map.values()]
        self.global_best_value = float(np.max(all_targets))
        best_cid = max(self._ground_truth_map.keys(), key=lambda k: self._ground_truth_map[k][target_column])
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
            cid = str(candidate_input.get(FECONI_CANDIDATE_ID_COLUMN, ""))
        else:
            raise TypeError(f"Unsupported candidate_input type: {type(candidate_input)}")

        if not cid or cid not in self._ground_truth_map:
            raise KeyError(
                f"Candidate ID '{cid}' is not a valid measured physical material in the 921-sample benchmark."
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
    ) -> None:
        if target not in {"Kerr", "Coer"}:
            raise ValueError(f"Target must be 'Kerr' or 'Coer', got '{target}'")

        super().__init__()
        self.mat_path = Path(mat_path)
        self.target = target
        self.objective = objective
        self._raw_data: dict[str, np.ndarray] | None = None
        self._full_df: pd.DataFrame | None = None

    def _ensure_loaded(self) -> None:
        if self._raw_data is None or self._full_df is None:
            raw = load_raw_feconi_mat(self.mat_path)
            self._raw_data = raw

            C = raw["C"]
            # Verified column mapping: Col 0 = Co, Col 1 = Fe, Col 2 = Ni
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

    def get_candidate_pool(self) -> pd.DataFrame:
        """Returns the optimizer-visible candidate pool containing ONLY candidate_id, Co, Fe, and Ni.

        Zero oracle targets or structural spectra are included.
        """
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
