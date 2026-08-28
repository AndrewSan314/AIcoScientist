from src.build_dataset import build_master_dataset
from src.recommend import recommend_top
from src.sem_features import extract_sem_features
from src.train_model import train_model
from src.utils import SEM_IMAGE_DIR


def main():
    print("[1/3] Building master dataset...")
    build_master_dataset()
    print("Saved: data/processed/master_dataset.csv")

    print("\n[2/3] Training model...")
    train_model()
    print("Saved: outputs/trained_model.pkl")
    print("Saved: outputs/model_metrics.json")
    print("Saved: outputs/feature_importance.csv")

    print("\n[3/3] Generating recommendations...")
    recommend_top()
    print("Saved: outputs/recommendations.csv")

    if SEM_IMAGE_DIR.exists() and any(SEM_IMAGE_DIR.iterdir()):
        print("\n[extra] Extracting SEM image features...")
        sem_df = extract_sem_features()
        print(f"Saved: data/processed/sem_features_extracted.csv ({len(sem_df)} rows)")
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
