from __future__ import annotations

import argparse

import pandas as pd

from src.build_dataset import build_dataset, build_master_dataset
from src.datasets.registry import get_dataset_adapter
from src.recommend import recommend_top
from src.sem_features import extract_sem_features
from src.train_model import train_model
from src.utils import (
    MASTER_FILE,
    MODEL_FILE,
    OUTPUT_DIR,
    RECOMMENDATIONS_FILE,
    SEM_IMAGE_DIR,
)


def main(dataset: str = "si_mxene", mode: str = "full") -> None:
    adapter = get_dataset_adapter(dataset)
    legacy = dataset == "si_mxene"

    if legacy:
        model_path = MODEL_FILE
        if mode in {"full", "train"} or not MASTER_FILE.exists():
            print("[1/3] Building master dataset...")
            df = build_master_dataset()
            print("Saved: data/processed/master_dataset.csv")
        else:
            df = pd.read_csv(MASTER_FILE)
    else:
        model_path = OUTPUT_DIR / dataset / "trained_model.pkl"
        print(f"[1/3] Loading dataset for {dataset}...")
        df = adapter.load()
        print(f"Loaded {len(df)} records for {dataset}.")

    if mode in {"full", "train"}:
        print("\n[2/3] Training model...")
        train_model(df, adapter=adapter, output_path=model_path)
        if legacy:
            print("Saved: outputs/trained_model.pkl")
            print("Saved: outputs/model_metrics.json")
            print("Saved: outputs/feature_importance.csv")
        else:
            print(f"Saved: {OUTPUT_DIR / dataset / 'trained_model.pkl'}")
            print(f"Saved: {OUTPUT_DIR / dataset / 'model_metrics.json'}")
            print(f"Saved: {OUTPUT_DIR / dataset / 'feature_importance.csv'}")

    if mode in {"full", "recommend"}:
        if dataset == "severson":
            print("\n[3/3] Note: Severson 2019 is an early-life prediction benchmark (no recommendation loop).")
        else:
            if not model_path.is_file():
                train_model(df, adapter=adapter, output_path=model_path)
            print("\n[3/3] Generating recommendations...")
            if legacy:
                recommend_top()
                print("Saved: outputs/recommendations.csv")
            else:
                from src.optimization.recommender import recommend

                recommend(adapter, df, model_path=model_path)

    if legacy and mode == "full" and SEM_IMAGE_DIR.exists() and any(SEM_IMAGE_DIR.iterdir()):
        print("\n[extra] Extracting SEM image features...")
        sem_df = extract_sem_features()
        print(f"Saved: data/processed/sem_features_extracted.csv ({len(sem_df)} rows)")
    if mode == "full":
        print("\nPipeline completed successfully.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dataset pipeline")
    parser.add_argument("--dataset", choices=("si_mxene", "severson", "dynamic_cycling"), default="si_mxene")
    parser.add_argument("--mode", choices=("train", "recommend", "full"), default="full")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(dataset=args.dataset, mode=args.mode)

