from __future__ import annotations

import numpy as np
import pandas as pd


def confidence(distance: float, std: float) -> str:
    if distance <= 0.7 and std <= 2.0:
        return "high"
    if distance <= 1.2 and std <= 5.0:
        return "medium"
    return "low"


def select_top_n(
    candidates: pd.DataFrame,
    n: int,
    *,
    score_column: str = "final_score",
) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be positive")
    if score_column not in candidates:
        raise ValueError(f"Missing score column: {score_column}")
    result = candidates.sort_values(score_column, ascending=False).head(n).copy()
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result.reset_index(drop=True)
