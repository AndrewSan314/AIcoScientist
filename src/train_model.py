from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.build_dataset import build_dataset
from src.datasets.base import DatasetAdapter, DatasetSpec
from src.datasets.registry import get_dataset_adapter
from src.utils import (
    IMPORTANCE_FILE,
    MASTER_FILE,
    METRICS_FILE,
    MODEL_FILE,
    OUTPUT_DIR,
)


def _regression_metrics(y_true, pred, rows, target):
    if len(y_true) < 2:
        r2 = None
    else:
        try:
            r2 = float(r2_score(y_true, pred))
        except Exception:
            r2 = None
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": r2,
        "rows": int(rows),
        "target": target,
    }


def compute_uncertainty_calibration(
    y_true: np.ndarray | Sequence[float],
    pred_mean: np.ndarray | Sequence[float],
    pred_std: np.ndarray | Sequence[float],
) -> dict[str, Any]:
    """Computes Gaussian uncertainty calibration metrics on held-out test data.

    Evaluates:
    - Empirical coverage of 50%, 80%, 90%, 95% predictive intervals
    - Gaussian Negative Log-Likelihood (NLL)
    - Standardized residual statistics: z = (y_true - pred_mean) / pred_std
    """
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(pred_mean, dtype=float)
    sigma = np.maximum(np.asarray(pred_std, dtype=float), 1e-6)

    # Standard normal quantiles for symmetric intervals
    z_50 = 0.6744897501960817
    z_80 = 1.2815515655446004
    z_90 = 1.6448536269514722
    z_95 = 1.959963984540054

    coverage_50 = float(np.mean(np.abs(y - mu) <= z_50 * sigma))
    coverage_80 = float(np.mean(np.abs(y - mu) <= z_80 * sigma))
    coverage_90 = float(np.mean(np.abs(y - mu) <= z_90 * sigma))
    coverage_95 = float(np.mean(np.abs(y - mu) <= z_95 * sigma))

    # Gaussian negative log predictive density
    nll_per_sample = 0.5 * np.log(2.0 * np.pi * (sigma ** 2)) + 0.5 * (((y - mu) / sigma) ** 2)
    mean_nll = float(np.mean(nll_per_sample))

    # Standardized residuals: z = (y - mu) / sigma
    z_residuals = (y - mu) / sigma
    z_mean = float(np.mean(z_residuals))
    z_std = float(np.std(z_residuals, ddof=1)) if len(z_residuals) > 1 else 0.0

    return {
        "coverage_50_pct": coverage_50,
        "coverage_80_pct": coverage_80,
        "coverage_90_pct": coverage_90,
        "coverage_95_pct": coverage_95,
        "mean_gaussian_nll": mean_nll,
        "standardized_residuals_mean": z_mean,
        "standardized_residuals_std": z_std,
        "test_samples_count": len(y),
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


def _xgb_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )


def make_train_test_split(
    df: pd.DataFrame,
    spec: DatasetSpec,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(df)
    if n_rows < 4:
        raise ValueError("At least 4 rows are required for a train/test split")

    if not spec.split_group_columns:
        test_count = max(2, int(round(n_rows * test_size)))
        if n_rows - test_count < 2:
            test_count = n_rows - 2
        train_idx, test_idx = train_test_split(
            np.arange(n_rows), test_size=test_count, random_state=random_state
        )
        if len(train_idx) < 2 or len(test_idx) < 2:
            raise ValueError(
                f"Train/test split produced insufficient rows: train={len(train_idx)}, test={len(test_idx)}"
            )
        return train_idx, test_idx

    missing = sorted(set(spec.split_group_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing split_group_columns: {missing}")

    groups = df[spec.split_group_columns].astype(str).agg("||".join, axis=1)
    unique_groups = groups.unique()
    if len(unique_groups) < 2:
        raise ValueError(
            f"At least 2 unique groups are required for grouped split, found {len(unique_groups)}"
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    splits = list(splitter.split(df, groups=groups))
    if not splits:
        raise ValueError("Could not generate grouped split")
    train_idx, test_idx = splits[0]

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    if train_groups & test_groups:
        raise RuntimeError("Grouped split leaked groups across train and test partitions")

    if len(train_idx) < 2 or len(test_idx) < 2:
        raise ValueError(
            f"Grouped train/test split resulted in fewer than 2 samples (train={len(train_idx)}, test={len(test_idx)})"
        )

    return train_idx, test_idx


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

    train_idx, test_idx = make_train_test_split(df, spec)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf_model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_metrics = _regression_metrics(y_test, rf_pred, len(df), spec.target_column)

    xgb_model = _xgb_model()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_metrics = _regression_metrics(y_test, xgb_pred, len(df), spec.target_column)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    gp_model = _gp_model()
    gp_model.fit(X_train_scaled, y_train)
    gp_pred, gp_std = gp_model.predict(X_test_scaled, return_std=True)
    gp_metrics = _regression_metrics(y_test, gp_pred, len(df), spec.target_column)
    gp_calibration = compute_uncertainty_calibration(y_test, gp_pred, gp_std)
    gp_metrics["uncertainty_calibration"] = gp_calibration

    final_rf_model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42)
    final_rf_model.fit(X, y)
    final_xgb_model = _xgb_model()
    final_xgb_model.fit(X, y)
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
            "xgb_model": final_xgb_model,
            "fill_values": X.median(numeric_only=True).to_dict(),
            "gp_model": final_gp_model,
            "scaler": final_scaler,
            "rf_metrics": rf_metrics,
            "xgb_metrics": xgb_metrics,
            "gp_metrics": gp_metrics,
            "gp_uncertainty_calibration": gp_calibration,
            "model_versions": {
                "baseline": "RandomForestRegressor",
                "challenger": "XGBRegressor",
                "surrogate": "GaussianProcessRegressor",
                "acquisition": "UCB beta=1.0 / True NEI",
            },
        },
        output_path,
    )
    metrics = {
        **rf_metrics,
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "gp_metrics": gp_metrics,
        "gp_uncertainty_calibration": gp_calibration,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    importance = pd.DataFrame(
        {
            "feature": spec.feature_columns,
            "importance": final_rf_model.feature_importances_,
            "rf_importance": final_rf_model.feature_importances_,
            "xgb_importance": final_xgb_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(importance_path, index=False)
    return metrics


def main():
    metrics = train_model()
    print(f"Saved: {MODEL_FILE}")
    print(f"Saved: {METRICS_FILE}")
    print(f"Saved: {IMPORTANCE_FILE}")
    print(f"Metrics (RF):  MAE={metrics['rf_metrics']['mae']:.2f}, RMSE={metrics['rf_metrics']['rmse']:.2f}, R2={metrics['rf_metrics']['r2']}")
    print(f"Metrics (XGB): MAE={metrics['xgb_metrics']['mae']:.2f}, RMSE={metrics['xgb_metrics']['rmse']:.2f}, R2={metrics['xgb_metrics']['r2']}")
    print(f"Metrics (GP):  MAE={metrics['gp_metrics']['mae']:.2f}, RMSE={metrics['gp_metrics']['rmse']:.2f}, R2={metrics['gp_metrics']['r2']}")
    cal = metrics.get("gp_uncertainty_calibration", {})
    print(f"GP Calibration: 50% cov={cal.get('coverage_50_pct', 0):.2f}, 95% cov={cal.get('coverage_95_pct', 0):.2f}, NLL={cal.get('mean_gaussian_nll', 0):.2f}")


if __name__ == "__main__":
    main()
