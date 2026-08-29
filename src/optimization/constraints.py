from __future__ import annotations

from typing import Any

import pandas as pd

from src.datasets.base import DatasetAdapter


def apply_constraints(
    candidates: pd.DataFrame,
    adapter: DatasetAdapter,
) -> pd.DataFrame:
    accepted: list[dict[str, Any]] = []
    for row in candidates.to_dict(orient="records"):
        validation = adapter.validate_candidate(row)
        if isinstance(validation, tuple):
            valid, _violations = validation
        else:
            valid = validation.valid
        if not valid:
            continue
        result = dict(row)
        result.update(adapter.candidate_metadata(row))
        accepted.append(result)
    return pd.DataFrame(accepted, columns=[*candidates.columns]) if not accepted else pd.DataFrame(accepted)
