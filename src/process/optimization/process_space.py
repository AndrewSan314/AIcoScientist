from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
