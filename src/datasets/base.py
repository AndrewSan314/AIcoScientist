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
    pre_experiment_features: list[str] = field(default_factory=list)
    post_experiment_characterization: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    candidate_variables: list[str] = field(default_factory=list)
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

        # Initialize defaults for new scientific contract fields if not provided
        if not self.targets:
            object.__setattr__(self, "targets", [self.target_column])
        if not self.candidate_columns and self.candidate_variables:
            object.__setattr__(self, "candidate_columns", list(self.candidate_variables))
        if not self.candidate_variables and self.candidate_columns:
            object.__setattr__(self, "candidate_variables", list(self.candidate_columns))
        if not self.pre_experiment_features:
            if self.candidate_columns:
                object.__setattr__(self, "pre_experiment_features", list(self.candidate_columns))
            else:
                pre_feats = [c for c in self.feature_columns if c not in self.post_experiment_characterization]
                object.__setattr__(self, "pre_experiment_features", pre_feats)

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
        if len(set(self.pre_experiment_features)) != len(self.pre_experiment_features):
            raise ValueError("pre_experiment_features must not contain duplicates")
        if len(set(self.post_experiment_characterization)) != len(self.post_experiment_characterization):
            raise ValueError("post_experiment_characterization must not contain duplicates")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must not contain duplicates")
        if len(set(self.candidate_variables)) != len(self.candidate_variables):
            raise ValueError("candidate_variables must not contain duplicates")

        for col in self.split_group_columns:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("split_group_columns must not contain empty column names")
        for col in self.oracle_columns:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("oracle_columns must not contain empty column names")
        for col in self.observation_columns:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("observation_columns must not contain empty column names")
        for col in self.pre_experiment_features:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("pre_experiment_features must not contain empty column names")
        for col in self.post_experiment_characterization:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("post_experiment_characterization must not contain empty column names")
        for col in self.targets:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("targets must not contain empty column names")
        for col in self.candidate_variables:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("candidate_variables must not contain empty column names")

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

        # Pre/Post experiment characterization leakage checks
        pre_post_overlap = set(self.pre_experiment_features) & set(self.post_experiment_characterization)
        if pre_post_overlap:
            raise ValueError(
                f"pre_experiment_features and post_experiment_characterization must not overlap: {sorted(pre_post_overlap)}"
            )

        cand_post_overlap = set(self.candidate_columns) & set(self.post_experiment_characterization)
        if cand_post_overlap:
            raise ValueError(
                f"candidate_columns and post_experiment_characterization must not overlap: {sorted(cand_post_overlap)}"
            )

        cand_var_post_overlap = set(self.candidate_variables) & set(self.post_experiment_characterization)
        if cand_var_post_overlap:
            raise ValueError(
                f"candidate_variables and post_experiment_characterization must not overlap: {sorted(cand_var_post_overlap)}"
            )

        pre_oracle_overlap = set(self.pre_experiment_features) & set(self.oracle_columns)
        if pre_oracle_overlap:
            raise ValueError(
                f"pre_experiment_features and oracle_columns must not overlap: {sorted(pre_oracle_overlap)}"
            )

    def optimizer_visible_features(self, stage: str = "pre_experiment") -> list[str]:
        """Returns the list of features strictly visible to the optimizer at the given decision stage.

        At 'pre_experiment', only pre-experiment controllable variables are accessible.
        At 'post_observation', observed characterization measurements become visible for learning.
        """
        if stage == "pre_experiment":
            return list(self.candidate_variables or self.candidate_columns or self.pre_experiment_features)
        elif stage == "post_observation":
            return list(self.feature_columns)
        else:
            raise ValueError(f"Unknown workflow stage: {stage!r}. Supported stages: 'pre_experiment', 'post_observation'.")

    def learning_features(self, stage: str = "post_observation") -> list[str]:
        """Returns features available for supervised modeling after physical characterization."""
        if stage == "post_observation":
            return list(self.feature_columns)
        elif stage == "pre_experiment":
            return list(self.pre_experiment_features)
        else:
            raise ValueError(f"Unknown workflow stage: {stage!r}")


@dataclass(frozen=True)
class TwoStageModelSpec:
    """Specification for two-stage scientific models:

    Stage A (Process -> Structure/Characterization):
        Inputs: pre_experiment_features (e.g. synthesis temp, precursor ratios)
        Outputs: post_experiment_characterization (e.g. porosity, grain size, morphology)

    Stage B (Structure/Characterization + Process -> Property/Performance):
        Inputs: pre_experiment_features + post_experiment_characterization
        Outputs: targets (e.g. capacity retention, cycle life)
    """
    dataset_name: str
    process_features: list[str]
    characterization_targets: list[str]
    performance_targets: list[str]


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
