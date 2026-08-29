from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.datasets.si_mxene_spec import SI_MXENE_SPEC
from src.train_model import (
    _xgb_model,
    compute_uncertainty_calibration,
    train_model,
)


def test_xgboost_model_hyperparameters() -> None:
    xgb = _xgb_model()
    params = xgb.get_params()

    assert params["n_estimators"] == 200
    assert params["max_depth"] == 2
    assert np.isclose(params["learning_rate"], 0.03)
    assert np.isclose(params["subsample"], 0.8)
    assert np.isclose(params["colsample_bytree"], 0.8)
    assert np.isclose(params["reg_alpha"], 0.1)
    assert np.isclose(params["reg_lambda"], 1.0)
    assert params["random_state"] == 42


def test_compute_uncertainty_calibration_metrics() -> None:
    y_true = np.array([10.0, 12.0, 9.0, 11.0, 10.5])
    pred_mean = np.array([10.1, 11.8, 9.2, 10.9, 10.4])
    pred_std = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    cal = compute_uncertainty_calibration(y_true, pred_mean, pred_std)

    assert "coverage_50_pct" in cal
    assert "coverage_80_pct" in cal
    assert "coverage_90_pct" in cal
    assert "coverage_95_pct" in cal
    assert "mean_gaussian_nll" in cal
    assert "standardized_residuals_mean" in cal
    assert "standardized_residuals_std" in cal

    assert 0.0 <= cal["coverage_50_pct"] <= 1.0
    assert 0.0 <= cal["coverage_95_pct"] <= 1.0
    assert np.isfinite(cal["mean_gaussian_nll"])
    assert np.isfinite(cal["standardized_residuals_mean"])


def test_train_model_xgboost_metrics_and_persistence(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n_rows = 20

    data = {feat: rng.uniform(10.0, 50.0, size=n_rows) for feat in SI_MXENE_SPEC.feature_columns}
    data["sample_id"] = [f"sample_{i}" for i in range(n_rows)]
    data[SI_MXENE_SPEC.target_column] = rng.uniform(70.0, 95.0, size=n_rows)
    df = pd.DataFrame(data)

    model_path = tmp_path / "test_model.pkl"
    metrics = train_model(df=df, spec=SI_MXENE_SPEC, output_path=model_path)

    assert "rf_metrics" in metrics
    assert "xgb_metrics" in metrics
    assert "gp_metrics" in metrics
    assert "gp_uncertainty_calibration" in metrics

    for model_key in ["rf_metrics", "xgb_metrics", "gp_metrics"]:
        m = metrics[model_key]
        assert "mae" in m and np.isfinite(m["mae"])
        assert "rmse" in m and np.isfinite(m["rmse"])
        assert "r2" in m

    assert model_path.is_file()
    assert (tmp_path / "model_metrics.json").is_file()
    assert (tmp_path / "feature_importance.csv").is_file()

    imp_df = pd.read_csv(tmp_path / "feature_importance.csv")
    assert "feature" in imp_df.columns
    assert "rf_importance" in imp_df.columns
    assert "xgb_importance" in imp_df.columns
