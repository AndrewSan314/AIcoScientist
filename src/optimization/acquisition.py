from __future__ import annotations

import numpy as np


def ucb(mean, std, beta: float = 1.0, objective: str = "maximize") -> np.ndarray:
    if objective not in {"maximize", "minimize"}:
        raise ValueError("objective must be 'maximize' or 'minimize'")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if mean.shape != std.shape:
        raise ValueError("mean and std must have the same shape")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("mean and std must contain finite values and non-negative std")
    return (mean if objective == "maximize" else -mean) + beta * std


upper_confidence_bound = ucb
