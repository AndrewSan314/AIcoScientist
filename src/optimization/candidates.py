from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def normalize_candidate_schema(
    candidates: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    missing = sorted(set(columns) - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate pool is missing identity columns: {missing}")
    return candidates.drop_duplicates(subset=list(columns)).reset_index(drop=True)


def remove_observed(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    columns = list(columns)
    for name, frame in (("candidate", candidates), ("observed", observed)):
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} data is missing identity columns: {missing}")
    seen = set(observed[columns].itertuples(index=False, name=None))
    mask = [tuple(row) not in seen for row in candidates[columns].itertuples(index=False, name=None)]
    return candidates.loc[mask].reset_index(drop=True)
