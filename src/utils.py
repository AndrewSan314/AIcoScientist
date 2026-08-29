from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
SCREENSHOT_DIR = ROOT / "screenshots"

PROCESS_FILE = RAW_DIR / "process_data.csv"
SEM_IMAGE_DIR = RAW_DIR / "sem_images"
SEM_FILE = RAW_DIR / "sem_features.csv"
SEM_EXTRACTED_FILE = PROCESSED_DIR / "sem_features_extracted.csv"
EDX_FILE = RAW_DIR / "edx_data.csv"
ELECTROCHEM_FILE = RAW_DIR / "electrochem_data.csv"
MASTER_FILE = PROCESSED_DIR / "master_dataset.csv"
DATABASE_FILE = ROOT / "data" / "experiments.db"
SCHEMA_VERSION = 2
CHEMISTRY_RULES_VERSION = 1

MODEL_FILE = OUTPUT_DIR / "trained_model.pkl"
METRICS_FILE = OUTPUT_DIR / "model_metrics.json"
IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.csv"
RECOMMENDATIONS_FILE = OUTPUT_DIR / "recommendations.csv"

_LEGACY_DATASET_ATTRIBUTES = {
    "PROCESS_FEATURES",
    "SEM_FEATURES",
    "EDX_FEATURES",
    "ENGINEERED_FEATURES",
    "MODEL_FEATURES",
    "TARGET",
}


def __getattr__(name):
    if name in _LEGACY_DATASET_ATTRIBUTES:
        from src.datasets.si_mxene_spec import (
            EDX_FEATURES,
            ENGINEERED_FEATURES,
            MODEL_FEATURES,
            PROCESS_FEATURES,
            SEM_FEATURES,
            TARGET,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def ensure_dirs():
    for path in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, SEM_IMAGE_DIR, SCREENSHOT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ensure_sample_data(rows=50):
    """Create deterministic synthetic lab-style CSVs when real data is absent."""
    ensure_dirs()
    files = [PROCESS_FILE, SEM_FILE, EDX_FILE, ELECTROCHEM_FILE]
    if all(path.exists() for path in files):
        return
    if any(path.exists() for path in files):
        missing = ", ".join(path.name for path in files if not path.exists())
        raise FileNotFoundError(f"Raw dataset is incomplete; missing: {missing}")

    rng = np.random.default_rng(42)
    sample_id = [f"S{i:03d}" for i in range(1, rows + 1)]
    valid_compositions = np.array(
        [
            values
            for values in product(
                np.arange(55, 76, 5),
                np.arange(10, 31, 5),
                np.arange(5, 16, 5),
            )
            if 100 - sum(values) >= 5
        ]
    )
    si, mxene, alginate = valid_compositions[
        rng.integers(0, len(valid_compositions), rows)
    ].T
    carbon = 100 - si - mxene - alginate
    mixing = rng.choice([30, 45, 60], rows)
    drying = rng.choice([70, 80, 90, 100], rows)
    pressure = rng.choice([4, 5, 6, 7], rows)

    process = pd.DataFrame(
        {
            "sample_id": sample_id,
            "si_content": si,
            "mxene_content": mxene,
            "alginate_content": alginate,
            "carbon_content": carbon,
            "mixing_time": mixing,
            "drying_temp": drying,
            "pressing_pressure": pressure,
        }
    )

    porosity = np.clip(0.32 + (mxene - 10) * 0.008 + rng.normal(0, 0.04, rows), 0.2, 0.65)
    cracks = np.clip(0.04 + (si - 55) * 0.004 - alginate * 0.002 + rng.normal(0, 0.015, rows), 0.01, 0.2)
    uniformity = np.clip(0.72 + mxene * 0.004 - cracks + rng.normal(0, 0.04, rows), 0.45, 0.95)
    sem = pd.DataFrame(
        {
            "sample_id": sample_id,
            "particle_size_mean": np.round(170 - mxene * 1.8 + rng.normal(0, 12, rows), 2),
            "porosity_score": np.round(porosity, 3),
            "agglomeration_index": np.round(np.clip(0.55 - mxene * 0.01 + si * 0.002 + rng.normal(0, 0.04, rows), 0.12, 0.7), 3),
            "crack_density": np.round(cracks, 3),
            "surface_uniformity": np.round(uniformity, 3),
        }
    )

    edx = pd.DataFrame(
        {
            "sample_id": sample_id,
            "si_percent": np.round(si * 0.68 + rng.normal(0, 1.2, rows), 2),
            "ti_percent": np.round(mxene * 0.72 + rng.normal(0, 1.0, rows), 2),
            "c_percent": np.round(carbon * 1.8 + mxene * 0.3 + rng.normal(0, 1.5, rows), 2),
            "o_percent": np.round(7 + alginate * 0.45 + rng.normal(0, 0.7, rows), 2),
            "impurity_percent": np.round(np.clip(rng.normal(1.2, 0.5, rows), 0.1, 3.0), 2),
        }
    )

    initial = 980 + si * 5.5 + rng.normal(0, 45, rows)
    retention = np.clip(
        48
        + mxene * 0.65
        + alginate * 0.9
        + uniformity * 11
        - cracks * 55
        - np.abs(si - 65) * 0.35
        + rng.normal(0, 3.0, rows),
        45,
        92,
    )
    cap100 = initial * retention / 100
    electrochem = pd.DataFrame(
        {
            "sample_id": sample_id,
            "initial_capacity": np.round(initial, 2),
            "capacity_50": np.round(initial * (retention + rng.uniform(4, 9, rows)) / 100, 2),
            "capacity_100": np.round(cap100, 2),
            "retention_100": np.round(retention, 2),
            "coulombic_efficiency": np.round(np.clip(96 + retention * 0.03 + rng.normal(0, 0.25, rows), 95, 99.5), 2),
            "rct": np.round(np.clip(190 - mxene * 3 + cracks * 250 + rng.normal(0, 10, rows), 55, 210), 2),
        }
    )

    process.to_csv(PROCESS_FILE, index=False)
    sem.to_csv(SEM_FILE, index=False)
    edx.to_csv(EDX_FILE, index=False)
    electrochem.to_csv(ELECTROCHEM_FILE, index=False)
