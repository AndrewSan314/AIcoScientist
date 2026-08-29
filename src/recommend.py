from __future__ import annotations

import joblib
import pandas as pd

from src.build_dataset import build_dataset, build_master_dataset
from src.datasets.base import DatasetAdapter
from src.datasets.registry import get_dataset_adapter
from src.optimization.recommender import recommend
from src.train_model import train_model
from src.utils import MASTER_FILE, MODEL_FILE, OUTPUT_DIR, RECOMMENDATIONS_FILE


def recommend_top(n: int = 3, *, adapter: DatasetAdapter | None = None) -> pd.DataFrame:
    legacy = adapter is None
    adapter = adapter or get_dataset_adapter("si_mxene")
    if legacy:
        if not MASTER_FILE.exists():
            build_master_dataset()
        observed = pd.read_csv(MASTER_FILE)
        model_path = MODEL_FILE
        output_path = RECOMMENDATIONS_FILE
    else:
        observed = build_dataset(adapter)
        model_path = OUTPUT_DIR / adapter.spec.name / "trained_model.pkl"
        output_path = OUTPUT_DIR / adapter.spec.name / "recommendations.csv"

    if not model_path.is_file():
        train_model(observed, adapter=adapter, output_path=model_path)
    bundle = joblib.load(model_path)
    if "gp_model" not in bundle or "scaler" not in bundle:
        train_model(observed, adapter=adapter, output_path=model_path)
        bundle = joblib.load(model_path)
    return recommend(
        adapter,
        observed,
        model_bundle=bundle,
        n=n,
        output_path=output_path,
    )


def main() -> None:
    top = recommend_top()
    print(f"Saved: {RECOMMENDATIONS_FILE}")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
