from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class FiniteCandidatePool:
    """Manages finite measured candidate universe with strict physical identity tracking.

    Guarantees:
    1. Tensor conversions preserve bidirectional row_index <-> candidate_id <-> metadata mapping.
    2. Proposing from the pool always selects an actual measured candidate ID, never projecting
       a continuous point onto nearest neighbors.
    3. Exposes feature matrices and candidate subsets for unseen candidates only.
    """

    def __init__(
        self,
        candidate_pool: pd.DataFrame,
        feature_columns: Sequence[str],
        id_column: str = "candidate_id",
        metadata_columns: Sequence[str] | None = None,
        strict_identity: bool = True,
    ) -> None:
        if candidate_pool.empty:
            raise ValueError("Candidate pool dataframe cannot be empty.")
        if not feature_columns:
            raise ValueError("feature_columns must be provided.")

        self.id_column = id_column
        self.feature_columns = list(feature_columns)
        self.strict_identity = strict_identity

        pool = candidate_pool.copy().reset_index(drop=True)

        if self.strict_identity:
            if self.id_column not in pool.columns:
                raise ValueError(
                    f"Strict candidate identity requires explicit ID column {self.id_column!r} in candidate pool."
                )
            if pool[self.id_column].isna().any():
                raise ValueError(f"Candidate IDs in column {self.id_column!r} must be non-null under strict identity.")
            # Check uniqueness
            id_series = pool[self.id_column].astype(str)
            if id_series.duplicated().any():
                dupes = id_series[id_series.duplicated()].unique().tolist()
                raise ValueError(f"Duplicate candidate IDs found in candidate pool under strict identity: {dupes}")
        else:
            # Synthetic / demo mode only: Detect or construct candidate ID column if not explicitly present
            if self.id_column not in pool.columns:
                alt_ids = ["candidate_id", "sample_id", "policy_id", "sample_index", "id"]
                found_id = next((c for c in alt_ids if c in pool.columns), None)
                if found_id:
                    self.id_column = found_id
                else:
                    pool[self.id_column] = [f"CAND_{i:04d}" for i in range(len(pool))]

        # Verify all feature columns exist and are numeric
        missing = [c for c in self.feature_columns if c not in pool.columns]
        if missing:
            raise ValueError(f"Missing feature columns in candidate pool: {missing}")

        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(pool[col]):
                raise ValueError(
                    f"Candidate variable {col!r} is non-numeric (type: {pool[col].dtype}). "
                    "All candidate features must be numeric floats."
                )

        self.df = pool
        self.metadata_columns = list(metadata_columns) if metadata_columns else list(pool.columns)
        self._candidate_ids = [str(x) for x in pool[self.id_column]]
        self._id_to_index: dict[str, int] = {cid: idx for idx, cid in enumerate(self._candidate_ids)}

    def __len__(self) -> int:
        return len(self.df)

    @property
    def candidate_ids(self) -> list[str]:
        """Returns list of candidate IDs in row order."""
        return list(self._candidate_ids)

    def get_feature_matrix(self) -> np.ndarray:
        """Returns float feature matrix of shape (N, D)."""
        return self.df[self.feature_columns].to_numpy(dtype=float)

    def get_candidate_id(self, row_index: int) -> str:
        """Returns candidate ID at the given row index."""
        return self._candidate_ids[row_index]

    def get_design_variables(self, row_index: int) -> dict[str, Any]:
        """Returns design feature dictionary at the given row index."""
        row = self.df.iloc[row_index]
        return {col: float(row[col]) for col in self.feature_columns}

    def get_metadata(self, row_index: int) -> dict[str, Any]:
        """Returns complete metadata dictionary at the given row index."""
        return self.df.iloc[row_index].to_dict()

    def filter_unseen(
        self,
        observed: pd.DataFrame | Sequence[Mapping[str, Any]] | set[str] | Sequence[str],
    ) -> FiniteCandidatePool:
        """Constructs a sub-pool containing only unseen candidates."""
        seen_ids: set[str] = set()

        if isinstance(observed, set):
            seen_ids = {str(x) for x in observed}
        elif isinstance(observed, (list, tuple)):
            if observed and isinstance(observed[0], str):
                seen_ids = {str(x) for x in observed}
            elif observed and isinstance(observed[0], (dict, Mapping)):
                for r in observed:
                    cid = r.get("candidate_id") or r.get("sample_id") or r.get("policy_id") or r.get(self.id_column)
                    if cid is not None:
                        seen_ids.add(str(cid))
        elif isinstance(observed, pd.DataFrame):
            if not observed.empty:
                # Check candidate ID column
                cand_id_col = next((c for c in [self.id_column, "candidate_id", "sample_id", "policy_id"] if c in observed.columns), None)
                if cand_id_col:
                    seen_ids.update(str(x) for x in observed[cand_id_col].dropna())
                elif self.strict_identity:
                    raise ValueError(
                        f"Observed dataframe does not contain candidate ID column ({self.id_column!r}) "
                        "under strict candidate identity mode."
                    )
                elif all(c in observed.columns for c in self.feature_columns):
                    # Coordinate match fallback for non-strict synthetic mode only
                    obs_coords = set(tuple(np.round(row, 6)) for row in observed[self.feature_columns].to_numpy(dtype=float))
                    mask = [
                        tuple(np.round(row, 6)) not in obs_coords
                        for row in self.df[self.feature_columns].to_numpy(dtype=float)
                    ]
                    unseen_df = self.df.loc[mask].reset_index(drop=True)
                    if unseen_df.empty:
                        raise RuntimeError("All candidates in finite pool have already been observed.")
                    return FiniteCandidatePool(
                        unseen_df,
                        feature_columns=self.feature_columns,
                        id_column=self.id_column,
                        metadata_columns=self.metadata_columns,
                        strict_identity=self.strict_identity,
                    )

        mask = [cid not in seen_ids for cid in self._candidate_ids]
        unseen_df = self.df.loc[mask].reset_index(drop=True)
        if unseen_df.empty:
            raise RuntimeError("All candidates in finite pool have already been observed.")

        return FiniteCandidatePool(
            unseen_df,
            feature_columns=self.feature_columns,
            id_column=self.id_column,
            metadata_columns=self.metadata_columns,
            strict_identity=self.strict_identity,
        )
