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

        # Check required columns
        required = [spec.target_column]
        if spec.candidate_id_column and spec.candidate_id_column in df.columns:
            required.append(spec.candidate_id_column)
        else:
            required.extend(spec.candidate_columns)

        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f"Oracle dataset is missing required columns: {missing}")
        self.df = df.copy()
        self.spec = spec
        self.replicate_policy = replicate_policy

    def query(self, candidate: Mapping[str, Any] | pd.Series) -> OracleResponse:
        values = candidate.to_dict() if isinstance(candidate, pd.Series) else dict(candidate)

        use_candidate_id = bool(
            self.spec.candidate_id_column
            and self.spec.candidate_id_column in values
            and self.spec.candidate_id_column in self.df.columns
        )

        if use_candidate_id:
            cand_id_col = self.spec.candidate_id_column
            cand_id_val = values[cand_id_col]
            matches = self.df.loc[self.df[cand_id_col].astype(str) == str(cand_id_val)]

            if matches.empty:
                raise KeyError(f"No exact ground-truth candidate exists with {cand_id_col}={cand_id_val!r}")

            # Validate that if design coordinates are provided in query, they match stored ground truth
            for col in self.spec.candidate_columns:
                if col in values and col in matches.columns:
                    query_val = values[col]
                    stored_vals = matches[col].values
                    try:
                        q_float = float(query_val)
                        s_floats = stored_vals.astype(float)
                        if not np.all(np.isclose(s_floats, q_float, atol=1e-4, rtol=1e-4)):
                            raise ValueError(
                                f"Candidate design coordinate {col!r}={query_val} conflicts with ground truth for "
                                f"{cand_id_col}={cand_id_val!r} (expected {stored_vals[0]})."
                            )
                    except (ValueError, TypeError):
                        if not np.all(stored_vals == query_val):
                            raise ValueError(
                                f"Candidate design coordinate {col!r}={query_val} conflicts with ground truth for "
                                f"{cand_id_col}={cand_id_val!r} (expected {stored_vals[0]})."
                            )


            candidate_dict = {
                col: (values[col] if col in values else matches[col].iloc[0])
                for col in self.spec.candidate_columns
                if col in values or col in matches.columns
            }
            candidate_dict[cand_id_col] = cand_id_val

        else:
            # Fallback: match by candidate_columns
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

        # Handle replicates vs single match
        if len(matches) > 1:
            if self.replicate_policy == "error":
                raise ValueError(
                    f"Ground-truth candidate identity is ambiguous ({len(matches)} matching replicate records found in error mode)"
                )
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

