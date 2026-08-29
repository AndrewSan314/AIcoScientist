from __future__ import annotations

import numpy as np
import pandas as pd

from src.datasets.base import DatasetAdapter, DatasetSpec
from src.datasets.registry import get_dataset_adapter
from src.utils import MASTER_FILE, PROCESSED_DIR


def validate_dataset(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("adapter.load() must return a pandas DataFrame")
    required = [spec.id_column, *spec.feature_columns, spec.target_column]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[spec.id_column].isna().any():
        raise ValueError(f"Dataset contains null {spec.id_column} values")
    if df[spec.id_column].astype(str).str.strip().eq("").any():
        raise ValueError(f"Dataset contains empty {spec.id_column} values")

    result = df.copy()
    numeric = [*spec.feature_columns, spec.target_column]
    try:
        result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Dataset model features and target must be numeric") from error
    values = result[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Dataset model features and target must be finite and non-null")
    return result


def build_dataset(adapter: DatasetAdapter) -> pd.DataFrame:
    return validate_dataset(adapter.build_features(adapter.load()), adapter.spec)


def build_master_dataset() -> pd.DataFrame:
    df = build_dataset(get_dataset_adapter("si_mxene"))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_FILE, index=False)
    return df


def main():
    df = build_master_dataset()
    print(f"Saved: {MASTER_FILE.relative_to(MASTER_FILE.parents[2])} ({len(df)} rows)")


if __name__ == "__main__":
    main()
