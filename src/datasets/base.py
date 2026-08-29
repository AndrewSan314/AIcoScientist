from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    id_column: str
    feature_columns: list[str]
    target_column: str
    objective: str
    candidate_columns: list[str]
    optional_columns: list[str] = field(default_factory=list)
    task_type: str = "regression"

    def __post_init__(self) -> None:
        if self.objective not in {"maximize", "minimize"}:
            raise ValueError("objective must be 'maximize' or 'minimize'")
        if not self.name or not self.id_column or not self.target_column:
            raise ValueError("name, id_column, and target_column are required")
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        if not self.candidate_columns:
            raise ValueError("candidate_columns must not be empty")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must not contain duplicates")
        if len(set(self.candidate_columns)) != len(self.candidate_columns):
            raise ValueError("candidate_columns must not contain duplicates")


class DatasetAdapter(ABC):
    @property
    @abstractmethod
    def spec(self) -> DatasetSpec:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def candidate_space(self, observed: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def validate_candidate(self, candidate: Mapping[str, Any]) -> tuple[bool, list[str]]:
        return True, []

    def candidate_metadata(self, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        return {}

    def build_candidate_features(
        self,
        candidates: pd.DataFrame,
        observed: pd.DataFrame,
        fill_values: Mapping[str, Any],
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=candidates.index)
        for feature in self.spec.feature_columns:
            if feature in candidates:
                result[feature] = candidates[feature]
            elif feature in fill_values:
                result[feature] = fill_values[feature]
            else:
                raise ValueError(f"Missing candidate feature value for {feature!r}")
        return result[self.spec.feature_columns]

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
