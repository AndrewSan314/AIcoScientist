from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.build_dataset import build_dataset
from src.datasets.base import DatasetAdapter
from src.utils import OUTPUT_DIR

from .backend import OptimizerBackend
from .botorch_backend import BoTorchBackend
from .candidates import normalize_candidate_schema, remove_observed
from .constraints import apply_constraints
from .distance import nearest_neighbor_distances
from .objective import OptimizationObjective
from .proposal import CandidateProposal
from .selection import confidence, select_top_n


def recommend(
    adapter: DatasetAdapter,
    observed: pd.DataFrame | None = None,
    *,
    backend: OptimizerBackend | None = None,
    model_bundle: dict | None = None,
    n: int = 3,
    beta: float = 1.0,
    strategy: str = "gp_ucb",
    model_path: Path | None = None,
    output_path: Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates candidate recommendations by delegating surrogate/acquisition evaluation to OptimizerBackend."""
    if not adapter.spec.supports_optimization:
        raise ValueError(
            f"Dataset {adapter.spec.name!r} is a prediction-only dataset and does not support recommendation."
        )
    if observed is None:
        observed = build_dataset(adapter)

    candidates = normalize_candidate_schema(
        adapter.candidate_space(observed), adapter.spec
    )
    candidates = remove_observed(candidates, observed, adapter.spec)
    candidates = apply_constraints(candidates, adapter)

    if candidates.empty:
        raise RuntimeError("No valid unseen candidates are available")

    # Ensure a candidate_id column exists for 1-to-1 proposal mapping
    id_col = adapter.spec.candidate_id_column or "candidate_id"
    if id_col not in candidates.columns:
        candidates[id_col] = [f"CAND_{i:04d}" for i in range(len(candidates))]

    observed_df = observed.copy()
    if id_col not in observed_df.columns:
        for fallback_id in (adapter.spec.id_column, "sample_id", "experiment_id", "id"):
            if fallback_id and fallback_id in observed_df.columns:
                observed_df[id_col] = observed_df[fallback_id]
                break

    features = list(adapter.spec.feature_columns)
    fill_values = observed_df[features].median(numeric_only=True).to_dict()
    feature_matrix = adapter.build_candidate_features(candidates, observed_df, fill_values)
    if list(feature_matrix.columns) != features:
        raise ValueError("Adapter returned a candidate feature matrix with the wrong schema")

    # Combine candidates with full feature matrix
    candidates_with_features = candidates.copy()
    for feature in features:
        if feature not in candidates_with_features:
            candidates_with_features[feature] = feature_matrix[feature].to_numpy()

    # Use BoTorchBackend by default
    opt_backend = backend if backend is not None else BoTorchBackend(default_strategy=strategy)

    objective = OptimizationObjective(
        target_name=adapter.spec.target_column,
        minimize=adapter.spec.objective.strip().lower() == "minimize",
    )

    # Delegate proposal computation to the optimizer backend
    proposals = opt_backend.propose(
        observations=observed_df,
        candidate_pool=candidates_with_features,
        objective=objective,
        feature_columns=features,
        candidate_id_column=id_col,
        n=len(candidates_with_features),
        seed=seed,
        strategy=strategy,
        beta=beta,
    )

    # Map proposals back to candidates dataframe
    proposal_map = {p.candidate_id: p for p in proposals}
    candidates = candidates_with_features.copy()

    means = []
    stds = []
    acq_scores = []
    for cid in candidates[id_col]:
        p = proposal_map.get(str(cid))
        if p is not None:
            means.append(p.predicted_mean)
            stds.append(p.predicted_std)
            acq_scores.append(p.acquisition_value)
        else:
            means.append(0.0)
            stds.append(1.0)
            acq_scores.append(0.0)

    candidates["predicted_mean"] = np.array(means, dtype=float)
    candidates["predicted_std"] = np.array(stds, dtype=float)
    candidates["acquisition_score"] = np.array(acq_scores, dtype=float)

    candidates["final_score"] = adapter.adjust_acquisition_score(
        candidates,
        pd.Series(candidates["acquisition_score"].to_numpy(), index=candidates.index),
        candidates["predicted_mean"].to_numpy(),
    )
    distance_columns = adapter.distance_columns()
    candidates["nearest_distance"] = nearest_neighbor_distances(
        candidates, observed, distance_columns
    )
    candidates["confidence"] = [
        confidence(distance, uncertainty)
        for distance, uncertainty in zip(
            candidates["nearest_distance"], candidates["predicted_std"]
        )
    ]
    candidates[f"predicted_{adapter.spec.target_column}"] = candidates["predicted_mean"]
    candidates["recommendation_reason"] = [
        f"Recommended for high predicted {adapter.spec.target_column} (mean={m:.2f}) with epistemic uncertainty (std={s:.2f}) and novelty distance ({d:.2f})."
        for m, s, d in zip(
            candidates["predicted_mean"],
            candidates["predicted_std"],
            candidates["nearest_distance"],
        )
    ]
    top = select_top_n(candidates, n)
    result = adapter.format_recommendations(top, observed)
    if output_path is None:
        output_path = OUTPUT_DIR / adapter.spec.name / "recommendations.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


recommend_candidates = recommend
