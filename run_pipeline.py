from __future__ import annotations

import argparse

import pandas as pd

from src.build_dataset import build_dataset, build_master_dataset
from src.datasets.registry import get_dataset_adapter
from src.recommend import recommend_top
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

    # 1. Benchmark mode handling
    if mode == "continuous_benchmark":
        if dataset == "attia":
            print(f"Running continuous search-space benchmark suite for dataset: {dataset}...")
            from src.evaluation.attia_continuous_benchmark import run_attia_continuous_benchmark
            run_attia_continuous_benchmark(adapter)
            print(f"Continuous benchmark completed. Artifacts saved to: outputs/attia_continuous/")
            return
        else:
            raise ValueError(f"Dataset {dataset!r} does not currently support continuous_benchmark mode.")

    if mode == "benchmark" or (not legacy and mode == "full"):
        if dataset == "severson":
            print(f"Running benchmark suite for dataset: {dataset}...")
            from src.evaluation.severson_benchmark import run_severson_benchmark
            run_severson_benchmark(adapter)
            print(f"Benchmark completed. Artifacts saved to: outputs/severson/")
            return
        elif dataset == "dynamic_cycling":
            print(f"Running benchmark suite for dataset: {dataset}...")
            from src.evaluation.dynamic_cycling_benchmark import run_dynamic_cycling_benchmark
            run_dynamic_cycling_benchmark(adapter)
            print(f"Benchmark completed. Artifacts saved to: outputs/dynamic_cycling/")
            return
        elif dataset == "attia":
            print(f"Running benchmark suite for dataset: {dataset}...")
            from src.evaluation.attia_benchmark import run_attia_benchmark
            run_attia_benchmark(adapter)
            print(f"Benchmark completed. Artifacts saved to: outputs/attia/")
            return
        elif mode == "benchmark":
            print(f"Dataset {dataset!r} does not have a separate standalone benchmark runner. Running full pipeline.")
            main(dataset=dataset, mode="full")
            return

    # 2. Recommendation mode validation
    if mode == "recommend":
        if not adapter.spec.supports_optimization:
            raise ValueError(f"Dataset {dataset!r} is a prediction-only benchmark and does not support recommendation mode.")
        if dataset == "dynamic_cycling":
            raise ValueError(
                "Dynamic Cycling is an offline protocol optimization benchmark. "
                "Recommendation requires a partial observed subset or replay state; "
                "running recommendation on the full ground-truth dataset is forbidden to prevent data leakage. "
                "Use --mode benchmark instead."
            )
        if dataset == "attia":
            raise ValueError(
                "Attia is an offline protocol optimization benchmark. "
                "Recommendation requires a partial observed subset or replay state; "
                "running recommendation on the full candidate space without prior observations is forbidden. "
                "Use --mode benchmark instead."
            )

    # 3. Training and Legacy Execution
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
        if not adapter.spec.supports_optimization:
            print(f"\n[3/3] Note: Dataset {dataset!r} is a prediction-only benchmark (skipping recommendation).")
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
        try:
            from src.sem_features import extract_sem_features
            sem_df = extract_sem_features()
            print(f"Saved: data/processed/sem_features_extracted.csv ({len(sem_df)} rows)")
        except ImportError:
            print("Skipping optional SEM feature extraction: scikit-image is not installed.")
    if mode == "full":
        print("\nPipeline completed successfully.")



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dataset pipeline")
    parser.add_argument("--dataset", choices=("si_mxene", "severson", "dynamic_cycling", "attia"), default="si_mxene")
    parser.add_argument("--mode", choices=("train", "recommend", "full", "benchmark", "continuous_benchmark"), default="full")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(dataset=args.dataset, mode=args.mode)



