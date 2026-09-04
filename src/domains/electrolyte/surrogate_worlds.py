from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from src.domains.electrolyte.screening import ScreeningEvidenceMode, screen_large_pool_candidates
from src.science.hypothesis_backends.sklearn_backend import SklearnGaussianBackend


STAGE1_CONFIG = {
    "working_set_size": 200,
    "evidence_mode": ScreeningEvidenceMode.HISTORICAL_EVIDENCE,
    "discovery_scorer": "ensemble",
    "diversity_reservoir_size": 20000,
    "random_state": 42,
}


def _nonlinear_world(features: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(features, dtype=np.float64), nan=0.0)
    return (
        0.45 * np.sin(2.0 * np.pi * x[:, 0])
        + 0.35 * np.cos(np.pi * x[:, 1])
        + 0.20 * (x[:, 2] - 0.5) ** 2
        + 0.10 * np.sin(4.0 * np.pi * x[:, 0] * x[:, 1])
    )


def _world_truths(observed_features: np.ndarray, observed_targets: np.ndarray, candidate_features: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    extra = ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=seed, n_jobs=1)
    extra.fit(observed_features, observed_targets)
    backend = SklearnGaussianBackend(random_state=seed)
    sample = np.linspace(0, len(observed_features) - 1, min(256, len(observed_features)), dtype=int)
    backend.fit(observed_features[sample], observed_targets[sample])
    gp_mean, _ = backend.predict_distribution(candidate_features)
    return {
        "EXTRATREES": np.asarray(extra.predict(candidate_features), dtype=np.float64),
        "GP": np.asarray(gp_mean, dtype=np.float64),
        "NONLINEAR_SYNTHETIC": _nonlinear_world(candidate_features),
    }


def evaluate_cross_surrogate_worlds(
    candidates_df: pd.DataFrame,
    observed_features: np.ndarray,
    observed_targets: np.ndarray,
    feature_cols: Sequence[str],
    *,
    working_set_size: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluates one unchanged historical-evidence screen against independent frozen worlds."""
    feature_names = list(feature_cols)
    candidate_features = candidates_df[feature_names].to_numpy(dtype=np.float64, copy=False)
    truths = _world_truths(observed_features, observed_targets, candidate_features, random_state)
    results: dict[str, Any] = {"stage1_config": {**STAGE1_CONFIG, "evidence_mode": STAGE1_CONFIG["evidence_mode"].value}, "worlds": {}}
    for name, truth in truths.items():
        selected = screen_large_pool_candidates(
            candidates_df=candidates_df,
            observed_features=observed_features,
            observed_targets=observed_targets,
            working_set_size=working_set_size,
            feature_cols=feature_names,
            random_state=random_state,
            evidence_mode=ScreeningEvidenceMode.HISTORICAL_EVIDENCE,
            discovery_scorer="ensemble",
            diversity_reservoir_size=20000,
        )
        truth_by_id = dict(zip(candidates_df["candidate_id"].astype(str), truth))
        selected_values = np.asarray([truth_by_id[str(cid)] for cid in selected["candidate_id"]], dtype=np.float64)
        full_max = float(np.max(truth))
        selected_max = float(np.max(selected_values))
        results["worlds"][name] = {
            "candidate_count": int(len(candidates_df)),
            "working_set_size": int(len(selected)),
            "full_world_max": full_max,
            "working_set_max": selected_max,
            "screening_gap": max(0.0, full_max - selected_max),
            "working_set_p90_recovery": float(np.mean(selected_values >= np.percentile(truth, 90.0))),
            "screening_metadata": selected.attrs.get("screening_metadata", {}),
        }
    return results


__all__ = ["STAGE1_CONFIG", "evaluate_cross_surrogate_worlds"]
