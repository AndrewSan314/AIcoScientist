from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset specification encoding schema, visibility boundaries, and provenance.

    Concepts:
    - supports_prediction: Whether the dataset supports supervised predictive modeling / regression.
    - supports_optimization: Whether the dataset supports closed-loop design / candidate optimization.
    - entity_id_column: Identifies the physical experimental entity (e.g., a specific battery cell
      or sample). Multiple cycles/measurements can belong to the same physical entity.
    - candidate_id_column: Identifies the candidate protocol or formulation (e.g., protocol ID or recipe ID).
    - candidate_columns: Design coordinates/parameters that define the search space and input features.
    - split_group_columns: Columns defining groups that must never cross train/test splits.
    - oracle_columns: Hidden ground truth/diagnostic columns strictly forbidden from model training.
    - observation_columns: Revealed post-query observations (experimental feedback).
    - feature_horizon: Maximum time or cycle cutoff index for early-prediction features.
    - source_dataset / source_version: Provenance metadata for reproducibility.
    """

    name: str
    id_column: str
    feature_columns: list[str]
    target_column: str
    objective: str = "maximize"
    candidate_columns: list[str] = field(default_factory=list)
    supports_prediction: bool = True
    supports_optimization: bool = True
    optional_columns: list[str] = field(default_factory=list)
    task_type: str = "regression"
    entity_id_column: str | None = None
    candidate_id_column: str | None = None
    split_group_columns: list[str] = field(default_factory=list)
    oracle_columns: list[str] = field(default_factory=list)
    observation_columns: list[str] = field(default_factory=list)
    feature_horizon: int | None = None
    source_dataset: str | None = None
    source_version: str | None = None

    def __post_init__(self) -> None:
        if self.objective not in {"maximize", "minimize"}:
            raise ValueError("objective must be 'maximize' or 'minimize'")
        if not self.name or not self.name.strip():
            raise ValueError("name is required and cannot be empty")
        if not self.id_column or not self.id_column.strip():
            raise ValueError("id_column is required and cannot be empty")
        if not self.target_column or not self.target_column.strip():
            raise ValueError("target_column is required and cannot be empty")
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")

        if self.supports_optimization:
            if not self.candidate_columns:
                raise ValueError("candidate_columns must not be empty for optimization-capable datasets")
        else:
            if self.candidate_id_column is not None and self.candidate_id_column == self.entity_id_column:
                pass  # entity identity is separate

        if self.entity_id_column is not None:
            if not isinstance(self.entity_id_column, str) or not self.entity_id_column.strip():
                raise ValueError("entity_id_column must be a non-empty string when provided")
        if self.candidate_id_column is not None:
            if not isinstance(self.candidate_id_column, str) or not self.candidate_id_column.strip():
                raise ValueError("candidate_id_column must be a non-empty string when provided")

        if self.feature_horizon is not None:
            if (
                isinstance(self.feature_horizon, bool)
                or not isinstance(self.feature_horizon, int)
                or self.feature_horizon <= 0
            ):
                raise ValueError("feature_horizon must be a positive integer when provided")

        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must not contain duplicates")
        if len(set(self.candidate_columns)) != len(self.candidate_columns):
            raise ValueError("candidate_columns must not contain duplicates")
        if len(set(self.split_group_columns)) != len(self.split_group_columns):
            raise ValueError("split_group_columns must not contain duplicates")
        if len(set(self.oracle_columns)) != len(self.oracle_columns):
            raise ValueError("oracle_columns must not contain duplicates")
        if len(set(self.observation_columns)) != len(self.observation_columns):
            raise ValueError("observation_columns must not contain duplicates")

        for col in self.split_group_columns:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("split_group_columns must not contain empty column names")
        for col in self.oracle_columns:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("oracle_columns must not contain empty column names")
        for col in self.observation_columns:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("observation_columns must not contain empty column names")

        if self.target_column in self.feature_columns:
            raise ValueError(f"target_column {self.target_column!r} cannot be in feature_columns")
        if self.target_column in self.observation_columns:
            raise ValueError(f"target_column {self.target_column!r} cannot be in observation_columns")

        feature_oracle_overlap = set(self.feature_columns) & set(self.oracle_columns)
        if feature_oracle_overlap:
            raise ValueError(
                f"feature_columns and oracle_columns must not overlap: {sorted(feature_oracle_overlap)}"
            )

        candidate_oracle_overlap = set(self.candidate_columns) & set(self.oracle_columns)
        if candidate_oracle_overlap:
            raise ValueError(
                f"candidate_columns and oracle_columns must not overlap: {sorted(candidate_oracle_overlap)}"
            )

        obs_oracle_overlap = set(self.observation_columns) & set(self.oracle_columns)
        if obs_oracle_overlap:
            raise ValueError(
                f"observation_columns and oracle_columns must not overlap: {sorted(obs_oracle_overlap)}"
            )


@dataclass
class DatasetBundle:
    candidates: pd.DataFrame
    observations: pd.DataFrame
    oracle: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    @property
    @abstractmethod
    def spec(self) -> DatasetSpec:
        raise NotImplementedError

    @abstractmethod
    def load(self, force_recompute: bool = False) -> pd.DataFrame:
        raise NotImplementedError

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def candidate_space(self, observed: pd.DataFrame) -> pd.DataFrame:
        if not self.spec.supports_optimization:
            raise NotImplementedError(
                f"Dataset {self.spec.name!r} is a prediction-only dataset and does not support candidate_space()."
            )
        raise NotImplementedError


    def validate_candidate(self, candidate: Mapping[str, Any]) -> tuple[bool, list[str]]:
        return True, []

    def candidate_metadata(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        return {}

    @property
    def static_feature_defaults(self) -> Mapping[str, Any]:
        return {}

    def build_candidate_features(
        self,
        candidates: pd.DataFrame,
        observed: pd.DataFrame,
        fill_values: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=candidates.index)
        static_defaults = self.static_feature_defaults
        for feature in self.spec.feature_columns:
            if feature in candidates.columns:
                result[feature] = candidates[feature]
            elif feature in static_defaults:
                result[feature] = static_defaults[feature]
            else:
                raise ValueError(
                    f"Cannot construct pre-experiment feature {feature!r} safely from candidate data or static defaults"
                )
        return result[self.spec.feature_columns]

    def build_observed_row(
        self,
        candidate: Mapping[str, Any],
        response: Any,
        step: int,
    ) -> dict[str, Any]:
        observations = (
            response.observations if hasattr(response, "observations") else response.get("observations", {})
        )
        target = response.target if hasattr(response, "target") else response.get("target")
        row = {
            self.spec.id_column: f"replay_step_{step}",
            **{k: v for k, v in candidate.items() if k in self.spec.candidate_columns},
            **observations,
            self.spec.target_column: target,
        }
        return row

    def distance_columns(self) -> list[str]:
        return list(self.spec.candidate_columns)

    def adjust_acquisition_score(
        self,
        candidates: pd.DataFrame,
        acquisition_score: pd.Series,
        predicted_mean: np.ndarray,
    ) -> pd.Series:
        return acquisition_score

    def format_recommendations(
        self,
        candidates: pd.DataFrame,
        observed: pd.DataFrame,
    ) -> pd.DataFrame:
        return candidates.reset_index(drop=True)
