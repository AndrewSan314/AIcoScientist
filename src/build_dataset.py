from src.edx_features import add_edx_features
from src.experiment_store import ingest_csvs, load_source_tables
from src.utils import (
    MASTER_FILE,
    MODEL_FEATURES,
    PROCESSED_DIR,
    TARGET,
    ensure_sample_data,
)


def build_master_dataset():
    ensure_sample_data()
    ingest_csvs()
    sources = load_source_tables()
    process = sources["process_data"]
    sem = sources["sem_features"]
    edx = sources["edx_data"]
    electrochem = sources["electrochem_data"]

    df = process.merge(sem, on="sample_id", validate="one_to_one")
    df = df.merge(edx, on="sample_id", validate="one_to_one")
    df = df.merge(electrochem, on="sample_id", validate="one_to_one")

    df["si_mxene_ratio"] = df["si_content"] / df["mxene_content"]
    df = add_edx_features(df)
    df["capacity_fade"] = df["initial_capacity"] - df["capacity_100"]
    if df[[*MODEL_FEATURES, TARGET]].isna().any().any():
        raise ValueError("Master dataset contains missing model values")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_FILE, index=False)
    return df


def main():
    df = build_master_dataset()
    print(f"Saved: {MASTER_FILE.relative_to(MASTER_FILE.parents[2])} ({len(df)} rows)")


if __name__ == "__main__":
    main()
