from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProcessSearchSpace:
    """Fail-closed finite process-recipe space with separate context/features."""

    candidates: pd.DataFrame
    id_column: str = "recipe_id"
    context_columns: tuple[str, ...] = ()
    context_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id_column not in self.candidates or self.candidates.empty:
            raise ValueError("process search space needs a non-empty recipe identity column")
        if self.candidates[self.id_column].isna().any() or self.candidates[self.id_column].astype(str).duplicated().any():
            raise ValueError("process recipe identities must be non-null and unique")
        if self.id_column in self.context_columns or len(set(self.context_columns)) != len(self.context_columns):
            raise ValueError("context columns must be unique and separate from recipe identity")
        missing = [column for column in self.context_columns if column not in self.candidates]
        if missing:
            raise ValueError(f"context columns missing from candidate pool: {missing}")
        if set(self.context_bounds) != set(self.context_columns):
            raise ValueError("context bounds must be supplied for every context column")
        for column in self.context_columns:
            try:
                lower, upper = (float(value) for value in self.context_bounds[column])
            except (TypeError, ValueError):
                raise ValueError(f"invalid bounds for context column {column!r}") from None
            if not np.isfinite((lower, upper)).all() or lower > upper:
                raise ValueError(f"invalid bounds for context column {column!r}")

    @classmethod
    def from_finite_pool(
        cls,
        candidates: pd.DataFrame,
        *,
        id_column: str = "recipe_id",
        context_columns: tuple[str, ...] = (),
        context_bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> "ProcessSearchSpace":
        return cls(
            candidates=candidates.copy().reset_index(drop=True), id_column=id_column,
            context_columns=context_columns, context_bounds=context_bounds or {},
        )

    @property
    def control_columns(self) -> list[str]:
        return [column for column in self.candidates if column not in {self.id_column, *self.context_columns}]

    @property
    def model_columns(self) -> list[str]:
        """Context first, then controllable features supplied to the optimizer."""
        return [*self.context_columns, *self.control_columns]

    def validate_recipe(self, controls: dict[str, Any]) -> bool:
        if set(controls) != set(self.control_columns):
            return False
        mask = pd.Series(True, index=self.candidates.index)
        for name, value in controls.items():
            mask &= self.candidates[name].eq(value)
        return bool(mask.any())

    def recipe(self, recipe_id: str) -> dict[str, Any]:
        match = self.candidates.loc[self.candidates[self.id_column].astype(str) == str(recipe_id)]
        if len(match) != 1:
            raise KeyError(f"unknown source recipe: {recipe_id}")
        return match.iloc[0][self.control_columns].to_dict()

    def official_botorch_view(self, observations: pd.DataFrame) -> tuple[pd.DataFrame, "ProcessSearchSpace"]:
        """Encode audited finite-pool controls and contextual state for official BoTorch models."""
        controls = self.control_columns
        context = list(self.context_columns)
        missing = [column for column in [self.id_column, *controls, *context] if column not in observations]
        if missing:
            raise ValueError(f"observations lack required identity/control/context columns: {missing}")
        if not controls:
            raise ValueError("official process BoTorch requires at least one control")
        encoded_pool = pd.DataFrame({self.id_column: self.candidates[self.id_column]})
        encoded_observations = observations.drop(columns=controls).copy()
        for column in context:
            pool_values = pd.to_numeric(self.candidates[column], errors="coerce")
            history_values = pd.to_numeric(observations[column], errors="coerce")
            lower, upper = (float(value) for value in self.context_bounds[column])
            if (
                pool_values.isna().any() or history_values.isna().any()
                or not np.isfinite(pool_values.to_numpy()).all() or not np.isfinite(history_values.to_numpy()).all()
                or ((history_values < lower) | (history_values > upper)).any()
                or ((pool_values < lower) | (pool_values > upper)).any()
            ):
                raise ValueError(f"context column {column!r} contains non-finite or out-of-support values")
            span = upper - lower or 1.0
            encoded_pool[column] = (pool_values - lower) / span
            encoded_observations[column] = (history_values - lower) / span
        for position, column in enumerate(controls):
            pool, history = self.candidates[column], observations[column]
            numeric_pool = pd.to_numeric(pool, errors="coerce")
            numeric_history = pd.to_numeric(history, errors="coerce")
            pool_present, history_present = pool.notna(), history.notna()
            numeric = (
                pool_present.any()
                and numeric_pool[pool_present].notna().all()
                and numeric_history[history_present].notna().all()
                and np.isfinite(numeric_pool[pool_present].to_numpy()).all()
                and np.isfinite(numeric_history[history_present].to_numpy()).all()
            )
            prefix = f"control_{position}"
            if numeric:
                lower, upper = numeric_pool[pool_present].min(), numeric_pool[pool_present].max()
                span = upper - lower or 1.0
                encoded_pool[f"{prefix}_value"] = ((numeric_pool - lower) / span).fillna(0.0)
                encoded_observations[f"{prefix}_value"] = ((numeric_history - lower) / span).fillna(0.0)
                if not pool_present.all() or not history_present.all():
                    encoded_pool[f"{prefix}_observed"] = pool_present.astype(float)
                    encoded_observations[f"{prefix}_observed"] = history_present.astype(float)
                if ((encoded_observations[f"{prefix}_value"] < -1e-12) | (encoded_observations[f"{prefix}_value"] > 1 + 1e-12)).any():
                    raise ValueError(f"observed control {column!r} falls outside the audited finite recipe pool")
                continue
            def category_key(value: object) -> tuple[str, str]:
                return ("missing", "") if pd.isna(value) else ("value", str(value))

            categories = sorted(set(pool.map(category_key)))
            history_categories = history.map(category_key)
            unknown = sorted(set(history_categories) - set(categories))
            if unknown:
                raise ValueError(f"observations include unaudited {column!r} categories: {unknown}")
            pool_categories = pool.map(category_key)
            for category_index, category in enumerate(categories):
                feature = f"{prefix}_category_{category_index}"
                encoded_pool[feature] = (pool_categories == category).astype(float)
                encoded_observations[feature] = (history_categories == category).astype(float)
        return encoded_observations, ProcessSearchSpace(
            encoded_pool, id_column=self.id_column, context_columns=self.context_columns,
            context_bounds={column: (0.0, 1.0) for column in self.context_columns},
        )
