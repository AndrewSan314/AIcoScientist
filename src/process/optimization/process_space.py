from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProcessSearchSpace:
    """Fail-closed finite process-recipe search space for initial source replay."""

    candidates: pd.DataFrame
    id_column: str = "recipe_id"

    def __post_init__(self) -> None:
        if self.id_column not in self.candidates or self.candidates.empty:
            raise ValueError("process search space needs a non-empty recipe identity column")
        if self.candidates[self.id_column].isna().any() or self.candidates[self.id_column].astype(str).duplicated().any():
            raise ValueError("process recipe identities must be non-null and unique")

    @classmethod
    def from_finite_pool(cls, candidates: pd.DataFrame, *, id_column: str = "recipe_id") -> "ProcessSearchSpace":
        return cls(candidates=candidates.copy().reset_index(drop=True), id_column=id_column)

    @property
    def control_columns(self) -> list[str]:
        return [column for column in self.candidates if column != self.id_column]

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
        """Encode only audited finite-pool controls for official BoTorch models."""
        controls = self.control_columns
        missing = [column for column in [self.id_column, *controls] if column not in observations]
        if missing:
            raise ValueError(f"observations lack required identity/control columns: {missing}")
        if not controls:
            raise ValueError("official process BoTorch requires at least one control")
        encoded_pool = pd.DataFrame({self.id_column: self.candidates[self.id_column]})
        encoded_observations = observations.drop(columns=controls).copy()
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
        return encoded_observations, ProcessSearchSpace(encoded_pool, id_column=self.id_column)
