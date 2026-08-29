from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec


@dataclass(frozen=True)
class OracleResponse:
    candidate: dict[str, Any]
    observations: dict[str, Any]
    target: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key == "target":
            return self.target
        elif key == "candidate":
            return self.candidate
        elif key == "observations":
            return self.observations
        elif key == "metadata":
            return self.metadata
        raise KeyError(
            f"Key {key!r} is not accessible on OracleResponse. Raw hidden rows are private to prevent data leakage."
        )


class OfflineOracle:
    def __init__(
        self,
        df: pd.DataFrame,
        spec: DatasetSpec,
        *,
        replicate_policy: str = "error",
    ):
        if replicate_policy not in {"error", "mean"}:
            raise ValueError(f"replicate_policy must be 'error' or 'mean', got {replicate_policy!r}")
        required = [*spec.candidate_columns, spec.target_column]
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f"Oracle dataset is missing required columns: {missing}")
        self.df = df.copy()
        self.spec = spec
        self.replicate_policy = replicate_policy

    def query(self, candidate: Mapping[str, Any] | pd.Series) -> OracleResponse:
        values = candidate.to_dict() if isinstance(candidate, pd.Series) else dict(candidate)
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

        candidate_dict = {col: values[col] for col in self.spec.candidate_columns}

        if len(matches) > 1:
            if self.replicate_policy == "error":
                raise ValueError("Ground-truth candidate identity is ambiguous")
            elif self.replicate_policy == "mean":
                if self.spec.observation_columns:
                    raise ValueError(
                        "Cannot query replicated candidate with observation_columns: "
                        "observation aggregation for replicated candidates must be explicitly defined."
                    )
                targets = matches[self.spec.target_column].to_numpy(dtype=float)
                n_replicates = int(len(matches))
                target_mean = float(np.mean(targets))
                target_std = float(np.std(targets, ddof=1)) if n_replicates > 1 else 0.0

                metadata = {
                    "n_replicates": n_replicates,
                    "target_std": target_std,
                }
                if self.spec.candidate_id_column and self.spec.candidate_id_column in matches.columns:
                    metadata["candidate_id"] = matches[self.spec.candidate_id_column].iloc[0]
                elif self.spec.candidate_id_column and self.spec.candidate_id_column in values:
                    metadata["candidate_id"] = values[self.spec.candidate_id_column]

                return OracleResponse(
                    candidate=candidate_dict,
                    observations={},
                    target=target_mean,
                    metadata=metadata,
                )

        row = matches.iloc[0]
        target = float(row[self.spec.target_column])
        obs_dict = {
            col: row[col]
            for col in self.spec.observation_columns
            if col in row
        }
        metadata = {"n_replicates": 1, "target_std": 0.0} if self.replicate_policy == "mean" else {}
        if self.spec.candidate_id_column and self.spec.candidate_id_column in row:
            metadata["candidate_id"] = row[self.spec.candidate_id_column]
        elif self.spec.candidate_id_column and self.spec.candidate_id_column in values:
            metadata["candidate_id"] = values[self.spec.candidate_id_column]

        return OracleResponse(
            candidate=candidate_dict,
            observations=obs_dict,
            target=target,
            metadata=metadata,
        )
