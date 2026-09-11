from __future__ import annotations

import numpy as np


def conformal_interval(mean: np.ndarray, calibration_residuals: np.ndarray, coverage: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")
    residuals = np.abs(np.asarray(calibration_residuals, dtype=float))
    if not residuals.size:
        raise ValueError("calibration residuals are required")
    radius = float(np.quantile(residuals, coverage, method="higher"))
    values = np.asarray(mean, dtype=float)
    return values - radius, values + radius


def empirical_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual, lo, hi = np.asarray(y, dtype=float), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    return float(np.mean((actual >= lo) & (actual <= hi)))
