from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.build_dataset import build_dataset
from src.datasets.base import DatasetAdapter
from src.train_model import train_model
from src.utils import OUTPUT_DIR

from .acquisition import ucb
from .candidates import normalize_candidate_schema, remove_observed
from .constraints import apply_constraints
from .distance import nearest_neighbor_distances
from .selection import confidence, select_top_n


def recommend(
    adapter: DatasetAdapter,
    observed: pd.DataFrame | None = None,
    *,
    model_bundle: dict | None = None,
    n: int = 3,
    beta: float = 1.0,
    model_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    if not adapter.spec.supports_optimization:
        raise ValueError(
            f"Dataset {adapter.spec.name!r} is a prediction-only dataset and does not support recommendation."
        )
    if observed is None:
        observed = build_dataset(adapter)

    if model_bundle is None:
        if model_path is None:
            model_path = OUTPUT_DIR / adapter.spec.name / "trained_model.pkl"
        model_path = Path(model_path)
        if not model_path.is_file():
            train_model(observed, adapter=adapter, output_path=model_path)
        model_bundle = joblib.load(model_path)

    features = list(model_bundle.get("features", adapter.spec.feature_columns))
    if features != adapter.spec.feature_columns:
        raise ValueError("Model feature metadata does not match the dataset specification")
    try:
        gp_model = model_bundle["gp_model"]
        scaler = model_bundle["scaler"]
    except KeyError as error:
        raise ValueError("Model artifact must contain gp_model and scaler") from error

    candidates = normalize_candidate_schema(
        adapter.candidate_space(observed), adapter.spec
    )
    candidates = remove_observed(candidates, observed, adapter.spec)
    candidates = apply_constraints(candidates, adapter)

    if candidates.empty:
        raise RuntimeError("No valid unseen candidates are available")

    fill_values = model_bundle.get(
        "fill_values", observed[features].median(numeric_only=True).to_dict()
    )
    feature_matrix = adapter.build_candidate_features(candidates, observed, fill_values)
    if list(feature_matrix.columns) != features:
        raise ValueError("Adapter returned a candidate feature matrix with the wrong schema")
    mean, std = gp_model.predict(scaler.transform(feature_matrix), return_std=True)
    candidates = candidates.copy()
    for feature in features:
        if feature not in candidates:
            candidates[feature] = feature_matrix[feature].to_numpy()
    candidates["predicted_mean"] = mean
    candidates["predicted_std"] = std
    candidates["acquisition_score"] = ucb(
        mean, std, beta=beta, objective=adapter.spec.objective
    )
    candidates["final_score"] = adapter.adjust_acquisition_score(
        candidates,
        pd.Series(candidates["acquisition_score"].to_numpy(), index=candidates.index),
        mean,
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
    candidates[f"predicted_{adapter.spec.target_column}"] = mean
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
