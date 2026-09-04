from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domains.electrolyte.data import DEFAULT_COMPATIBLE_DERIVED_PATH, DEFAULT_VIRTUAL_1M_PATH, load_derived_historical_outcomes, load_lifsi_virtual_candidate_chunk
from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES
from src.domains.electrolyte.surrogate_worlds import evaluate_cross_surrogate_worlds


def main() -> None:
    historical = load_derived_historical_outcomes(DEFAULT_COMPATIBLE_DERIVED_PATH)
    features = list(ELECTROLYTE_SOLVENT_FEATURES)
    candidates = load_lifsi_virtual_candidate_chunk(DEFAULT_VIRTUAL_1M_PATH, nrows=333333, feature_cols=features, generate_ids=True)
    result = evaluate_cross_surrogate_worlds(
        candidates,
        historical[features].to_numpy(),
        historical["C_norm_20"].to_numpy(),
        features,
    )
    os.makedirs("outputs/electrolyte/benchmark", exist_ok=True)
    with open("outputs/electrolyte/benchmark/screening_cross_surrogate_robustness.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
