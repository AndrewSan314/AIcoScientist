from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def normalized_nearest_neighbor_distance(
    candidate: Mapping[str, float] | pd.Series,
    observed: pd.DataFrame,
    columns: Sequence[str],
) -> float:
    columns = list(columns)
    if observed.empty:
        raise ValueError("Cannot calculate distance from an empty observed dataset")
    candidate_columns = set(candidate.index if isinstance(candidate, pd.Series) else candidate)
    missing = sorted(set(columns) - set(observed.columns) - candidate_columns)
    if missing:
        raise ValueError(f"Distance data is missing columns: {missing}")
    existing = observed[columns].apply(pd.to_numeric, errors="raise")
    point = pd.Series(candidate)[columns].apply(pd.to_numeric, errors="raise")
    ranges = (existing.max() - existing.min()).replace(0, 1)
    return float(((existing - point) / ranges).pow(2).sum(axis=1).pow(0.5).min())


def nearest_neighbor_distances(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    columns: Sequence[str],
) -> pd.Series:
    columns = list(columns)
    if observed.empty:
        raise ValueError("Cannot calculate distance from an empty observed dataset")
    missing = sorted(set(columns) - set(observed.columns) - set(candidates.columns))
    if missing:
        raise ValueError(f"Distance data is missing columns: {missing}")
    existing = observed[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    points = candidates[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    ranges = observed[columns].max().to_numpy(float) - observed[columns].min().to_numpy(float)
    ranges[ranges == 0] = 1
    distances = np.sqrt(
        (((points[:, None, :] - existing[None, :, :]) / ranges) ** 2).sum(axis=2)
    )
    return pd.Series(distances.min(axis=1), index=candidates.index)


nearest_distance = normalized_nearest_neighbor_distance
