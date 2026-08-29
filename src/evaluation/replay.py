from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import pandas as pd

from src.datasets.base import DatasetAdapter
from src.evaluation.oracle import OfflineOracle
from src.optimization.recommender import recommend
from src.train_model import train_model


def replay(
    adapter: DatasetAdapter,
    oracle: OfflineOracle,
    initial_observed: pd.DataFrame,
    budget: int,
    *,
    beta: float = 1.0,
    model_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    observed = initial_observed.copy()
    history = []

    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(model_dir) if model_dir is not None else Path(temporary_dir)
        root.mkdir(parents=True, exist_ok=True)
        model_path = root / "trained_model.pkl"
        recommendation_path = root / "recommendations.csv"
        for step in range(1, budget + 1):
            train_model(observed, adapter=adapter, output_path=model_path)
            bundle = joblib.load(model_path)
            recommendations = recommend(
                adapter,
                observed,
                model_bundle=bundle,
                n=1,
                beta=beta,
                output_path=recommendation_path,
            )
            if recommendations.empty:
                raise RuntimeError("Replay recommender returned no candidate")
            candidate = recommendations.iloc[0].to_dict()
            response = oracle.query(candidate)
            observed = pd.concat(
                [observed, pd.DataFrame([response["row"]])], ignore_index=True
            )
            history.append(
                {
                    "step": step,
                    **{column: candidate[column] for column in adapter.spec.candidate_columns},
                    adapter.spec.target_column: response["target"],
                }
            )
    return {"observed": observed, "history": pd.DataFrame(history)}


run_replay = replay
