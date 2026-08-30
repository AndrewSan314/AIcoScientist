from __future__ import annotations

from typing import Any
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.gaussian_process.kernels import CompoundKernel, Kernel, Sum, WhiteKernel


def extract_signal_kernel(kernel: Kernel) -> Kernel | None:
    """Extracts signal covariance kernel from a compound kernel by pruning pure WhiteKernel noise components."""
    if isinstance(kernel, WhiteKernel):
        return None

    if isinstance(kernel, Sum):
        k1_sig = extract_signal_kernel(kernel.k1)
        k2_sig = extract_signal_kernel(kernel.k2)
        if k1_sig is not None and k2_sig is not None:
            return k1_sig + k2_sig
        return k1_sig or k2_sig

    if isinstance(kernel, CompoundKernel):
        filtered_kernels = [k for k in kernel.kernels if not isinstance(k, WhiteKernel)]
        if not filtered_kernels:
            return None
        if len(filtered_kernels) == 1:
            return filtered_kernels[0]
        return CompoundKernel(filtered_kernels)

    return kernel


def predict_latent_gp(
    gp: Any,
    X: np.ndarray,
    return_std: bool = False,
    return_cov: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Computes exact epistemic latent posterior mean and variance/covariance excluding observation noise."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    if not hasattr(gp, "kernel_") or not hasattr(gp, "X_train_") or not hasattr(gp, "L_") or not hasattr(gp, "alpha_"):
        return gp.predict(X_arr, return_std=return_std, return_cov=return_cov)

    signal_kernel = extract_signal_kernel(gp.kernel_)
    if signal_kernel is None or signal_kernel == gp.kernel_:
        return gp.predict(X_arr, return_std=return_std, return_cov=return_cov)

    # 1. K_trans = K_signal(X, X_train)
    K_trans = signal_kernel(X_arr, gp.X_train_)

    # 2. Target normalization scaling factors
    y_mean = getattr(gp, "_y_train_mean", 0.0)
    y_std = getattr(gp, "_y_train_std", 1.0)
    normalize_y = bool(getattr(gp, "normalize_y", False))

    # 3. Latent mean (normalized space -> original target units)
    mu_normalized = K_trans @ gp.alpha_
    if normalize_y:
        mu = y_mean + (y_std * mu_normalized)
    else:
        mu = y_mean + mu_normalized

    if return_cov:
        K_test = signal_kernel(X_arr, X_arr)
        V = solve_triangular(gp.L_, K_trans.T, lower=True, check_finite=False)
        cov_normalized = K_test - (V.T @ V)
        if normalize_y:
            cov = cov_normalized * (y_std ** 2)
        else:
            cov = cov_normalized
        return mu, cov
    elif return_std:
        K_diag = signal_kernel.diag(X_arr)
        V = solve_triangular(gp.L_, K_trans.T, lower=True, check_finite=False)
        var_normalized = K_diag - np.einsum("ij,ij->j", V, V)
        var_normalized = np.maximum(var_normalized, 0.0)
        std_normalized = np.sqrt(var_normalized)
        if normalize_y:
            std = std_normalized * y_std
        else:
            std = std_normalized
        return mu, std
    else:
        return mu
