from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .optimization.process_space import ProcessSearchSpace


@dataclass(frozen=True)
class ReplayStep:
    step: int
    recipe_id: str
    revealed_target: float


def validate_no_lookahead(observed: pd.DataFrame, hidden: pd.DataFrame, *, id_column: str, target: str) -> None:
    if target in observed.columns and observed[target].isna().any():
        raise ValueError("observed target values must be revealed values, not placeholders")
    if set(observed[id_column]) & set(hidden[id_column]):
        raise ValueError("a recipe cannot be both observed and hidden")
    if target not in hidden:
        raise ValueError("hidden source outcomes are required for offline replay")


def reveal_one(observed: pd.DataFrame, hidden: pd.DataFrame, *, recipe_id: str, id_column: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame, ReplayStep]:
    validate_no_lookahead(observed, hidden, id_column=id_column, target=target)
    row = hidden.loc[hidden[id_column].astype(str) == str(recipe_id)]
    if len(row) != 1:
        raise KeyError("only one currently hidden source recipe may be revealed")
    revealed = row.iloc[[0]].copy()
    next_observed = pd.concat([observed, revealed], ignore_index=True)
    next_hidden = hidden.drop(row.index)
    return next_observed, next_hidden, ReplayStep(len(next_observed), str(recipe_id), float(revealed.iloc[0][target]))
