from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.datasets.base import DatasetSpec, TwoStageModelSpec
from src.science.direct_baseline import DirectPerformanceModel
from src.science.two_stage import TwoStageScientificModel


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes regression metrics safely."""
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot > 1e-12:
        ss_res = np.sum(diff**2)
        r2 = float(1.0 - ss_res / ss_tot)
    else:
        r2 = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def _compute_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> dict[str, Any]:
    """Evaluates empirical coverage and calibration metrics against true measured observations."""
    std = np.maximum(y_std, 1e-8)
    z_scores = (y_true - y_pred) / std

    # Empirical coverage fractions
    cov_50 = float(np.mean(np.abs(z_scores) <= 0.67449))
    cov_80 = float(np.mean(np.abs(z_scores) <= 1.28155))
    cov_90 = float(np.mean(np.abs(z_scores) <= 1.64485))
    cov_95 = float(np.mean(np.abs(z_scores) <= 1.95996))

    # Gaussian negative log likelihood (NLL)
    nll = float(np.mean(0.5 * np.log(2.0 * np.pi * (std**2)) + 0.5 * (z_scores**2)))

    return {
        "coverage_50_pct": cov_50,
        "coverage_80_pct": cov_80,
        "coverage_90_pct": cov_90,
        "coverage_95_pct": cov_95,
        "mean_nll": nll,
        "mean_standardized_residual": float(np.mean(z_scores)),
        "std_standardized_residual": float(np.std(z_scores)),
    }


def evaluate_two_stage_model(
    two_stage_model: TwoStageScientificModel,
    direct_model: DirectPerformanceModel,
    test_df: pd.DataFrame,
    spec: DatasetSpec,
    two_stage_spec: TwoStageModelSpec,
    target_column: str | None = None,
    n_mc_samples: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """Performs honest comparative evaluation between Direct Baseline and Two-Stage Scientific Model.

    CRITICAL RULE: True test characterization is NEVER fed into end-to-end two-stage evaluation.
    Stage B with observed characterization is strictly evaluated as a diagnostic theoretical upper-bound.
    """
    if target_column is None:
        target_column = spec.target_column

    process_cols = two_stage_spec.process_features
    char_cols = two_stage_spec.characterization_targets
    perf_col = target_column

    # 1. Direct Baseline Evaluation
    valid_direct = test_df[process_cols + [perf_col]].dropna()
    y_test_direct = valid_direct[perf_col].to_numpy(dtype=float)
    dir_mean, dir_latent_std, dir_obs_std = direct_model.predict_with_observation_std(valid_direct[process_cols])
    direct_metrics = _compute_metrics(y_test_direct, dir_mean)
    direct_obs_calib = _compute_uncertainty_calibration(y_test_direct, dir_mean, dir_obs_std)
    direct_latent_calib = _compute_uncertainty_calibration(y_test_direct, dir_mean, dir_latent_std)

    # 2. Stage A Evaluation per Characterization Channel (using robust boolean index masking)
    stage_a_metrics: dict[str, dict[str, float]] = {}
    for char_col in char_cols:
        if char_col in test_df.columns:
            valid_mask = test_df[process_cols + [char_col]].notna().all(axis=1)
            if valid_mask.sum() > 0:
                y_c_true = test_df.loc[valid_mask, char_col].to_numpy(dtype=float)
                sub_preds = two_stage_model.predict_characterization(test_df.loc[valid_mask, process_cols])
                if char_col in sub_preds:
                    y_c_pred = sub_preds[char_col]["mean"]
                    stage_a_metrics[char_col] = _compute_metrics(y_c_true, y_c_pred)

    # 3. Stage B Diagnostic Evaluation (Oracle Characterization Upper Bound)
    stage_b_diagnostic: dict[str, Any] = {}
    valid_stage_b = test_df[process_cols + char_cols + [perf_col]].dropna()
    if len(valid_stage_b) > 0:
        y_b_true = valid_stage_b[perf_col].to_numpy(dtype=float)
        b_mean, b_latent_std, b_noise_var = two_stage_model.predict_performance_with_observed_characterization(
            valid_stage_b[process_cols],
            valid_stage_b[char_cols],
            target_name=perf_col,
        )
        b_obs_std = np.sqrt(b_latent_std**2 + b_noise_var)
        stage_b_diagnostic = {
            "nature": "Diagnostic oracle-characterization upper bound (not achievable at proposal time)",
            "metrics": _compute_metrics(y_b_true, b_mean),
            "calibration": _compute_uncertainty_calibration(y_b_true, b_mean, b_obs_std),
            "sample_count": len(valid_stage_b),
        }

    # 4. Two-Stage End-to-End Evaluation (Process -> Predicted Characterization -> Performance)
    valid_e2e = test_df[process_cols + [perf_col]].dropna()
    y_test_e2e = valid_e2e[perf_col].to_numpy(dtype=float)
    e2e_pred = two_stage_model.predict_end_to_end(
        valid_e2e[process_cols],
        target_name=perf_col,
        n_mc_samples=n_mc_samples,
        seed=seed,
    )
    e2e_metrics = _compute_metrics(y_test_e2e, e2e_pred.performance_mean)

    # Observation predictive calibration evaluates measured observations against observation predictive std
    e2e_obs_calib = _compute_uncertainty_calibration(
        y_test_e2e,
        e2e_pred.performance_mean,
        e2e_pred.performance_observation_std,
    )
    e2e_latent_calib = _compute_uncertainty_calibration(
        y_test_e2e,
        e2e_pred.performance_mean,
        e2e_pred.performance_latent_std,
    )

    # 5. Model Disagreement Summary on Test Set
    disagreement = np.abs(dir_mean - e2e_pred.performance_mean)
    pooled_uncertainty = np.sqrt(dir_latent_std**2 + e2e_pred.performance_latent_std**2)
    rel_disagreement = disagreement / np.maximum(pooled_uncertainty, 1e-6)

    disagreement_summary = {
        "mean_absolute_disagreement": float(np.mean(disagreement)),
        "max_absolute_disagreement": float(np.max(disagreement)),
        "high_disagreement_count": int(np.sum(rel_disagreement > 2.0)),
        "high_disagreement_fraction": float(np.mean(rel_disagreement > 2.0)),
    }

    # 6. Uncertainty Decomposition Summary
    uncertainty_decomposition = {
        "mean_total_predictive_variance": float(np.mean(e2e_pred.total_predictive_variance)),
        "mean_total_latent_variance": float(np.mean(e2e_pred.total_latent_variance)),
        "mean_characterization_propagation_variance": float(np.mean(e2e_pred.characterization_propagation_variance)),
        "mean_performance_model_variance": float(np.mean(e2e_pred.performance_model_variance)),
        "mean_observation_noise_variance": float(np.mean(e2e_pred.observation_noise_variance)),
        "mean_characterization_variance_fraction": float(
            np.mean(e2e_pred.characterization_propagation_variance / np.maximum(e2e_pred.total_latent_variance, 1e-12))
        ),
    }

    return {
        "dataset_name": spec.name,
        "target_column": perf_col,
        "test_sample_count": len(valid_e2e),
        "direct_baseline": {
            "metrics": direct_metrics,
            "observation_predictive_calibration": direct_obs_calib,
            "latent_uncertainty_calibration_diagnostic": direct_latent_calib,
            # Backward compatible key
            "calibration": direct_obs_calib,
        },
        "stage_a_characterization": stage_a_metrics,
        "stage_b_diagnostic_upper_bound": stage_b_diagnostic,
        "two_stage_end_to_end": {
            "metrics": e2e_metrics,
            "observation_predictive_calibration": e2e_obs_calib,
            "latent_uncertainty_calibration_diagnostic": e2e_latent_calib,
            "uncertainty_decomposition": uncertainty_decomposition,
            # Backward compatible key
            "calibration": e2e_obs_calib,
        },
        "model_disagreement_summary": disagreement_summary,
    }
