from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.datasets.base import DatasetSpec


class OfflineOracle:
    def __init__(self, df: pd.DataFrame, spec: DatasetSpec):
        required = [*spec.candidate_columns, spec.target_column]
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f"Oracle dataset is missing required columns: {missing}")
        self.df = df.copy()
        self.spec = spec

    def query(self, candidate: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
        values = candidate.to_dict() if isinstance(candidate, pd.Series) else candidate
        missing = sorted(set(self.spec.candidate_columns) - set(values))
        if missing:
            raise ValueError(f"Candidate is missing identity columns: {missing}")
        mask = pd.Series(True, index=self.df.index)
        for column in self.spec.candidate_columns:
            mask &= self.df[column].eq(values[column])
        matches = self.df.loc[mask]
        if matches.empty:
            identity = {column: values[column] for column in self.spec.candidate_columns}
            raise KeyError(f"No exact ground-truth candidate exists: {identity}")
        if len(matches) > 1:
            raise ValueError("Ground-truth candidate identity is ambiguous")
        row = matches.iloc[0].to_dict()
        return {"target": row[self.spec.target_column], "row": row}
