from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.datasets.base import DatasetSpec
from src.datasets.registry import get_dataset_adapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INPUT_FILE = PROCESSED_DIR / "master_dataset.csv"
MODEL_FILE = OUTPUTS_DIR / "trained_model.pkl"
METRICS_FILE = OUTPUTS_DIR / "model_metrics.json"
IMPORTANCE_FILE = OUTPUTS_DIR / "feature_importance.csv"

FEATURE_COLS = [
    "si_loading_wt_pct",
    "mxene_loading_wt_pct",
    "slurry_mixing_speed_rpm",
    "slurry_mixing_time_min",
    "drying_temp_c",
    "drying_time_h",
    "binder_content_wt_pct",
    "mass_loading_mg_cm2",
    "sem_particle_area_fraction",
    "sem_crack_density",
    "edx_si_wt_pct",
    "edx_ti_wt_pct",
]
TARGET_COL = "retention_100"

logger = logging.getLogger(__name__)


def compute_uncertainty_calibration(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
) -> dict[str, float | int]:
    """Calculates empirical uncertainty calibration metrics for Gaussian Process predictive distributions.

    Scientific Interpretation Note:
    -------------------------------
    - If empirical 95% coverage is substantially lower than nominal 95% (e.g. ~66.7%) and the
      standardized residual std (z_std) is > 1.0 (e.g. ~1.63), the GP predictive intervals are
      under-dispersed / overconfident on this test sample.
    - Standardized residuals mean (z_mean) near 0 indicates unbiased mean predictions.
    - For small test splits (e.g. N_test = 12), coverage estimates carry high sampling variance
      and must not be over-interpreted as asymptotic calibration guarantees.
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
        "calibration_assessment": "under-dispersed / overconfident on small held-out test sample" if z_std > 1.2 else "moderately calibrated",
    }


def _rf_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
    )


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

    group_col = spec.split_group_columns[0]
    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(
            f"Group column {group_col!r} contains only {len(unique_groups)} unique group. At least 2 unique groups are required."
        )

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    if len(train_idx) < 2 or len(test_idx) < 2:
        n_splits = min(4, len(unique_groups))
        gkf = GroupKFold(n_splits=n_splits)
        train_idx, test_idx = next(gkf.split(df, groups=groups))

    if len(train_idx) < 2 or len(test_idx) < 2:
        raise ValueError(
            f"Group split produced insufficient rows: train={len(train_idx)}, test={len(test_idx)}"
        )
    return train_idx, test_idx


def train_model(
    df: pd.DataFrame | None = None,
    spec: DatasetSpec | None = None,
    output_path: Path | str | None = None,
    adapter: Any | None = None,
) -> dict[str, Any]:
    legacy_output = False
    if spec is None and adapter is not None:
        spec = adapter.spec
    elif spec is None:
        spec = get_dataset_adapter("si_mxene").spec
    if df is None:
        if not INPUT_FILE.exists():
            from src.build_master_dataset import build_master_dataset
            df = build_master_dataset()
        else:
            df = pd.read_csv(INPUT_FILE)
    if output_path is None:
        output_path = MODEL_FILE
        legacy_output = True
    else:
        output_path = Path(output_path)

    missing = [c for c in spec.feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if spec.target_column not in df.columns:
        raise ValueError(f"Missing target column: {spec.target_column}")

    X = df[spec.feature_columns].copy()
    y = df[spec.target_column].copy()

    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)

    train_idx, test_idx = make_train_test_split(df, spec)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf_model = _rf_model()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred_rf)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_rf))),
        "r2": float(r2_score(y_test, y_pred_rf)),
    }

    xgb_model = _xgb_model()
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred_xgb)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_xgb))),
        "r2": float(r2_score(y_test, y_pred_xgb)),
    }

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    gp_model = _gp_model()
    gp_model.fit(X_train_scaled, y_train)
    y_pred_gp, y_std_gp = gp_model.predict(X_test_scaled, return_std=True)
    gp_metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred_gp)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_gp))),
        "r2": float(r2_score(y_test, y_pred_gp)),
    }

    # Compute empirical uncertainty calibration metrics on held-out test split
    gp_calibration = compute_uncertainty_calibration(
        y_true=y_test.to_numpy(),
        pred_mean=y_pred_gp,
        pred_std=y_std_gp,
    )

    final_rf_model = _rf_model()
    final_rf_model.fit(X, y)

    final_xgb_model = _xgb_model()
    final_xgb_model.fit(X, y)

    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X)
    final_gp_model = _gp_model()
    final_gp_model.fit(X_all_scaled, y)

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
