from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.datasets.base import DatasetSpec


def identity_columns(spec_or_columns: DatasetSpec | Sequence[str]) -> list[str]:
    """Returns canonical identity columns for candidate distinction."""
    if isinstance(spec_or_columns, DatasetSpec):
        if spec_or_columns.candidate_id_column:
            return [spec_or_columns.candidate_id_column]
        return list(spec_or_columns.candidate_columns)
    return list(spec_or_columns)


def normalize_candidate_schema(
    candidates: pd.DataFrame,
    spec_or_columns: DatasetSpec | Sequence[str],
) -> pd.DataFrame:
    """Ensures candidate dataframe has required columns and deduplicates based on canonical identity."""
    if isinstance(spec_or_columns, DatasetSpec):
        id_cols = identity_columns(spec_or_columns)
        required_cols = list(spec_or_columns.candidate_columns)
        if spec_or_columns.candidate_id_column and spec_or_columns.candidate_id_column in candidates.columns:
            if spec_or_columns.candidate_id_column not in required_cols:
                required_cols.append(spec_or_columns.candidate_id_column)
    else:
        id_cols = list(spec_or_columns)
        required_cols = list(spec_or_columns)

    missing = sorted(set(required_cols) - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate pool is missing required columns: {missing}")

    # Use id_cols that exist in candidates
    dedup_cols = [col for col in id_cols if col in candidates.columns]
    return candidates.drop_duplicates(subset=dedup_cols).reset_index(drop=True)


def remove_observed(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    spec_or_columns: DatasetSpec | Sequence[str],
) -> pd.DataFrame:
    """Filters out already-observed candidates using canonical identity semantics."""
    if isinstance(spec_or_columns, DatasetSpec):
        id_cols = identity_columns(spec_or_columns)
    else:
        id_cols = list(spec_or_columns)

    if observed.empty:
        return candidates.reset_index(drop=True)

    for name, frame in (("candidate", candidates), ("observed", observed)):
        missing = sorted(set(id_cols) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} data is missing identity columns: {missing}")

    seen = set(observed[id_cols].astype(str).itertuples(index=False, name=None))
    mask = [
        tuple(row) not in seen
        for row in candidates[id_cols].astype(str).itertuples(index=False, name=None)
    ]
    return candidates.loc[mask].reset_index(drop=True)

