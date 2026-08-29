import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.build_dataset import build_dataset
from src.datasets.base import DatasetAdapter, DatasetSpec
from src.datasets.registry import get_dataset_adapter
from src.utils import (
    IMPORTANCE_FILE,
    METRICS_FILE,
    MODEL_FILE,
    OUTPUT_DIR,
    MASTER_FILE,
)


def _regression_metrics(y_true, pred, rows, target):
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
        "rows": int(rows),
        "target": target,
    }


def _gp_model():
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
        noise_level=1.0,
        noise_level_bounds=(1e-5, 1e1),
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=42,
    )


def train_model(
    df: pd.DataFrame | None = None,
    spec: DatasetSpec | None = None,
    *,
    adapter: DatasetAdapter | None = None,
    output_path: Path | None = None,
):
    legacy_defaults = adapter is None and df is None and spec is None and output_path is None
    if adapter is None:
        adapter = get_dataset_adapter("si_mxene")
    if spec is None:
        spec = adapter.spec
    if df is None:
        if legacy_defaults and MASTER_FILE.exists():
            df = pd.read_csv(MASTER_FILE)
        else:
            df = build_dataset(adapter)
            if legacy_defaults:
                MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(MASTER_FILE, index=False)

    if len(set(spec.feature_columns)) != len(spec.feature_columns):
        raise ValueError("feature_columns must not contain duplicates")
    if len(df) == 0:
        raise ValueError("Cannot train on a zero-row dataset")
    if len(df) < 4:
        raise ValueError("At least 4 rows are required for a train/test split")

    try:
        X = df[spec.feature_columns].apply(pd.to_numeric, errors="raise")
        y = pd.to_numeric(df[spec.target_column], errors="raise")
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("Model features and target must be numeric and present") from error
    if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Model features and target must be finite and non-null")
    if np.ptp(y.to_numpy(dtype=float)) == 0:
        raise ValueError("Target must not be constant")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_metrics = _regression_metrics(y_test, rf_pred, len(df), spec.target_column)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    gp_model = _gp_model()
    gp_model.fit(X_train_scaled, y_train)
    gp_pred = gp_model.predict(X_test_scaled)
    gp_metrics = _regression_metrics(y_test, gp_pred, len(df), spec.target_column)

    final_rf_model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42)
    final_rf_model.fit(X, y)
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_gp_model = _gp_model()
    final_gp_model.fit(X_scaled, y)

    if output_path is None:
        output_path = MODEL_FILE if legacy_defaults else OUTPUT_DIR / spec.name / "trained_model.pkl"
    output_path = Path(output_path)
    legacy_output = output_path.resolve() == MODEL_FILE.resolve()
    metrics_path = METRICS_FILE if legacy_output else output_path.with_name("model_metrics.json")
    importance_path = IMPORTANCE_FILE if legacy_output else output_path.with_name("feature_importance.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "dataset": spec.name,
            "features": spec.feature_columns,
            "target": spec.target_column,
            "objective": spec.objective,
            "model": final_rf_model,
            "fill_values": X.median(numeric_only=True).to_dict(),
            "gp_model": final_gp_model,
            "scaler": final_scaler,
            "rf_metrics": rf_metrics,
            "gp_metrics": gp_metrics,
            "model_versions": {
                "baseline": "RandomForestRegressor",
                "surrogate": "GaussianProcessRegressor",
                "acquisition": "UCB beta=1.0",
            },
        },
        output_path,
    )
    metrics = {**rf_metrics, "rf_metrics": rf_metrics, "gp_metrics": gp_metrics}
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    importance = pd.DataFrame(
        {"feature": spec.feature_columns, "importance": final_rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(importance_path, index=False)
    return metrics


def main():
    metrics = train_model()
    print(f"Saved: {MODEL_FILE}")
    print(f"Saved: {METRICS_FILE}")
    print(f"Saved: {IMPORTANCE_FILE}")
    print(f"Metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.2f}")


if __name__ == "__main__":
    main()
