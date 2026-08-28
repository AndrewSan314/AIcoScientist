import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.build_dataset import build_master_dataset
from src.utils import (
    IMPORTANCE_FILE,
    MASTER_FILE,
    METRICS_FILE,
    MODEL_FEATURES,
    MODEL_FILE,
    OUTPUT_DIR,
    TARGET,
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


def train_model():
    if not MASTER_FILE.exists():
        build_master_dataset()

    df = pd.read_csv(MASTER_FILE)
    X = df[MODEL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_metrics = _regression_metrics(y_test, rf_pred, len(df), TARGET)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    gp_model = _gp_model()
    gp_model.fit(X_train_scaled, y_train)
    gp_pred = gp_model.predict(X_test_scaled)
    gp_metrics = _regression_metrics(y_test, gp_pred, len(df), TARGET)

    final_rf_model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42)
    final_rf_model.fit(X, y)
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_gp_model = _gp_model()
    final_gp_model.fit(X_scaled, y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_rf_model,
            "features": MODEL_FEATURES,
            "target": TARGET,
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
        MODEL_FILE,
    )
    metrics = {**rf_metrics, "rf_metrics": rf_metrics, "gp_metrics": gp_metrics}
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    importance = pd.DataFrame(
        {"feature": MODEL_FEATURES, "importance": final_rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(IMPORTANCE_FILE, index=False)
    return metrics


def main():
    metrics = train_model()
    print(f"Saved: {MODEL_FILE}")
    print(f"Saved: {METRICS_FILE}")
    print(f"Saved: {IMPORTANCE_FILE}")
    print(f"Metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.2f}")


if __name__ == "__main__":
    main()
