from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.datasets.severson import SeversonAdapter


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    n = len(y_true)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if n >= 2 else None
    # Percentage error / MAPE
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6)))) * 100.0
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_percent": mape,
        "n_samples": int(n),
    }


def run_severson_benchmark(
    adapter: SeversonAdapter | None = None,
    output_dir: Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Runs early-life cycle life prediction benchmark on Severson 2019 dataset."""
    if adapter is None:
        adapter = SeversonAdapter()

    project_root = Path(__file__).resolve().parent.parent.parent
    if output_dir is None:
        output_dir = project_root / "outputs" / "severson"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = adapter.load()
    spec = adapter.spec
    if spec.feature_horizon is None:
        raise ValueError(f"DatasetSpec for {spec.name!r} must explicitly define feature_horizon")
    feature_horizon = spec.feature_horizon


    feature_cols = list(spec.feature_columns)
    target_col = spec.target_column

    # Partition by official published split
    train_mask = df["split"] == "train"
    prim_test_mask = df["split"] == "primary_test"
    sec_test_mask = df["split"] == "secondary_test"
    test_mask = prim_test_mask | sec_test_mask

    train_df = df[train_mask].copy()
    prim_test_df = df[prim_test_mask].copy()
    sec_test_df = df[sec_test_mask].copy()
    test_df = df[test_mask].copy()

    # Split count and data sanity assertions
    if len(train_df) != 41:
        raise ValueError(f"Expected 41 official train cells, got {len(train_df)}")
    if len(prim_test_df) != 43:
        raise ValueError(f"Expected 43 official primary test cells, got {len(prim_test_df)}")
    if len(sec_test_df) != 40:
        raise ValueError(f"Expected 40 official secondary test cells, got {len(sec_test_df)}")
    if len(df) != 124:
        raise ValueError(f"Expected 124 total physical cells, got {len(df)}")

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)

    X_prim = prim_test_df[feature_cols].to_numpy(dtype=float)
    y_prim = prim_test_df[target_col].to_numpy(dtype=float)

    X_sec = sec_test_df[feature_cols].to_numpy(dtype=float)
    y_sec = sec_test_df[target_col].to_numpy(dtype=float)

    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)

    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df[target_col].to_numpy(dtype=float)

    if not np.all(np.isfinite(X_all)):
        raise ValueError("Processed Severson features contain non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(y_all) & (y_all > 0)):
        raise ValueError("Processed Severson cycle_life targets contain non-finite or non-positive values")

    # 1. Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=random_state)
    rf_model.fit(X_train, y_train)

    rf_pred_train = rf_model.predict(X_train)
    rf_pred_prim = rf_model.predict(X_prim)
    rf_pred_sec = rf_model.predict(X_sec)
    rf_pred_test = rf_model.predict(X_test)
    rf_pred_all = rf_model.predict(X_all)

    # 2. Gaussian Process Regressor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_prim_scaled = scaler.transform(X_prim)
    X_sec_scaled = scaler.transform(X_sec)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X_all)

    kernel = ConstantKernel(1.0, (1e-2, 1e4)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
        noise_level=1.0,
        noise_level_bounds=(1e-5, 1e2),
    )
    gp_model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_state,
    )
    gp_model.fit(X_train_scaled, y_train)

    gp_pred_train = gp_model.predict(X_train_scaled)
    gp_pred_prim = gp_model.predict(X_prim_scaled)
    gp_pred_sec = gp_model.predict(X_sec_scaled)
    gp_pred_test = gp_model.predict(X_test_scaled)
    gp_pred_all = gp_model.predict(X_all_scaled)

    # Calculate metrics
    metrics = {
        "dataset": "severson_2019",
        "feature_horizon_cycles": feature_horizon,
        "features": feature_cols,
        "n_train": len(train_df),
        "n_primary_test": len(prim_test_df),
        "n_secondary_test": len(sec_test_df),
        "random_forest": {
            "train": compute_metrics(y_train, rf_pred_train),
            "primary_test": compute_metrics(y_prim, rf_pred_prim),
            "secondary_test": compute_metrics(y_sec, rf_pred_sec),
            "overall_test": compute_metrics(y_test, rf_pred_test),
        },
        "gaussian_process": {
            "train": compute_metrics(y_train, gp_pred_train),
            "primary_test": compute_metrics(y_prim, gp_pred_prim),
            "secondary_test": compute_metrics(y_sec, gp_pred_sec),
            "overall_test": compute_metrics(y_test, gp_pred_test),
        },
    }

    # Predictions DataFrame
    predictions_df = pd.DataFrame(
        {
            "physical_cell_id": df["physical_cell_id"],
            "split": df["split"],
            "batch_id": df["batch_id"],
            "charging_policy": df["charging_policy"],
            "actual_cycle_life": y_all,
            "predicted_cycle_life_rf": rf_pred_all,
            "predicted_cycle_life_gp": gp_pred_all,
        }
    )

    # Feature Importance DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    # Summary JSON
    summary = {
        "benchmark": "Severson 2019 Early-Life Cycle Life Prediction",
        "horizon_cycles": feature_horizon,
        "feature_note": "Includes paper-inspired Delta Q_100-10(V) curve features alongside early discharge, IR, and temperature statistics.",
        "splits": {
            "train": f"{len(train_df)} physical cells",
            "primary_test": f"{len(prim_test_df)} physical cells",
            "secondary_test": f"{len(sec_test_df)} physical cells",
            "total": f"{len(df)} physical cells",
        },
        "rf_overall_test_mae": metrics["random_forest"]["overall_test"]["mae"],
        "rf_overall_test_rmse": metrics["random_forest"]["overall_test"]["rmse"],
        "rf_overall_test_r2": metrics["random_forest"]["overall_test"]["r2"],
        "gp_overall_test_mae": metrics["gaussian_process"]["overall_test"]["mae"],
        "gp_overall_test_rmse": metrics["gaussian_process"]["overall_test"]["rmse"],
        "gp_overall_test_r2": metrics["gaussian_process"]["overall_test"]["r2"],
        "top_features": importance_df.head(5).to_dict(orient="records"),
    }

    # Save artifacts
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)


    # Save trained models
    joblib.dump(
        {
            "rf_model": rf_model,
            "gp_model": gp_model,
            "scaler": scaler,
            "features": feature_cols,
            "target": target_col,
            "metrics": metrics,
        },
        output_dir / "trained_model.pkl",
    )

    return summary


def main() -> None:
    summary = run_severson_benchmark()
    print("=" * 60)
    print("SEVERSON 2019 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Horizon: {summary['horizon_cycles']} cycles")
    print(f"Total Physical Cells: {summary['splits']['total']}")
    print(f"Random Forest  -> Test MAE: {summary['rf_overall_test_mae']:.1f}, RMSE: {summary['rf_overall_test_rmse']:.1f}, R2: {summary['rf_overall_test_r2']:.3f}")
    print(f"Gaussian Proc. -> Test MAE: {summary['gp_overall_test_mae']:.1f}, RMSE: {summary['gp_overall_test_rmse']:.1f}, R2: {summary['gp_overall_test_r2']:.3f}")
    print("\nTop 5 Features by Importance:")
    for feat in summary["top_features"]:
        print(f"  - {feat['feature']:<25} importance: {feat['importance']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
