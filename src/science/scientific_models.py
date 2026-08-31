from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from src.science.xrd_representation import XRDRepresentationExtractor

logger = logging.getLogger(__name__)


class StructureSurrogateModel:
    """Surrogate model predicting XRD structural embeddings and uncertainty from composition.

    Composition (Au, Ir, Rh) -> Predicted XRD embedding z_hat and structural uncertainty U_struct.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.is_fitted = False
        self._gprs: list[GaussianProcessRegressor] = []
        self._n_dims = 0

    def fit(self, compositions: np.ndarray, embeddings: np.ndarray) -> StructureSurrogateModel:
        """Fits Gaussian Process regressors for each structural embedding dimension.

        Args:
            compositions: (N, 3) or (N, 2) array of composition coordinates.
            embeddings: (N, D) array of XRD embeddings.
        """
        X = np.asarray(compositions, dtype=np.float64)
        Y = np.asarray(embeddings, dtype=np.float64)

        if len(X) == 0:
            self.is_fitted = False
            return self

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self._n_dims = Y.shape[1]
        self._gprs = []

        for d in range(self._n_dims):
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[10.0] * X.shape[1], length_scale_bounds=(1.0, 100.0)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1.0))
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=2,
                random_state=self.random_state + d,
                normalize_y=True,
            )
            gpr.fit(X, Y[:, d])
            self._gprs.append(gpr)

        self.is_fitted = True
        return self

    def predict(self, compositions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts mean structural embedding and structural uncertainty.

        Returns:
            (mean_embeddings (M, D), structural_uncertainties (M,))
        """
        X = np.asarray(compositions, dtype=np.float64)
        if not self.is_fitted or not self._gprs:
            # Prior / uniform uncertainty if no data
            return np.zeros((len(X), self._n_dims or 8)), np.ones(len(X), dtype=np.float64)

        means = []
        stds = []
        for gpr in self._gprs:
            m, s = gpr.predict(X, return_std=True)
            means.append(m)
            stds.append(s)

        mean_arr = np.column_stack(means)
        # Total structural uncertainty is root-mean-squared standard deviation across dimensions
        std_arr = np.sqrt(np.mean(np.column_stack(stds) ** 2, axis=1))
        return mean_arr, std_arr


class PropertySurrogateModel:
    """Surrogate model predicting electrochemical performance (k0) and uncertainty.

    Composition (Au, Ir, Rh) [+ structural features] -> Predicted k0 and property uncertainty U_prop.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.is_fitted = False
        self._gpr: GaussianProcessRegressor | None = None
        self._gpr_with_structure: GaussianProcessRegressor | None = None

    def fit(
        self,
        compositions: np.ndarray,
        targets: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> PropertySurrogateModel:
        """Fits GP surrogate on revealed property measurements."""
        X = np.asarray(compositions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)

        if len(X) == 0:
            self.is_fitted = False
            return self

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[10.0] * X.shape[1], nu=2.5, length_scale_bounds=(1.0, 100.0)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1.0))
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            random_state=self.random_state,
            normalize_y=True,
        )
        self._gpr.fit(X, y)

        if embeddings is not None and len(embeddings) == len(X):
            X_joint = np.hstack([X, embeddings])
            kernel_j = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[10.0] * X_joint.shape[1], nu=2.5, length_scale_bounds=(1.0, 100.0)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1.0))
            self._gpr_with_structure = GaussianProcessRegressor(
                kernel=kernel_j,
                n_restarts_optimizer=2,
                random_state=self.random_state,
                normalize_y=True,
            )
            self._gpr_with_structure.fit(X_joint, y)
        else:
            self._gpr_with_structure = None

        self.is_fitted = True
        return self

    def predict(self, compositions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts mean property and property uncertainty."""
        X = np.asarray(compositions, dtype=np.float64)
        if not self.is_fitted or self._gpr is None:
            return np.zeros(len(X)), np.ones(len(X), dtype=np.float64)

        m, s = self._gpr.predict(X, return_std=True)
        return m, s

    def evaluate_structure_predictive_advantage(
        self,
        compositions: np.ndarray,
        targets: np.ndarray,
        embeddings: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluates whether structural features improve predictive error over composition alone via Leave-One-Out Cross-Validation.

        GENUINE OUT-OF-SAMPLE CONTRACT:
        - Trains composition GP on N-1 samples and tests on held-out sample.
        - Trains structure-informed GP on N-1 samples and tests on held-out sample.
        - If N < 3, returns neutral advantage (0.0).
        """
        N = len(compositions)
        if N < 3:
            return {
                "composition_mse": 1.0,
                "structure_informed_mse": 1.0,
                "structure_advantage_ratio": 0.0,
                "note": "Insufficient joint observations (N < 3). Neutral evidence assigned.",
            }

        X = np.asarray(compositions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        Z = np.asarray(embeddings, dtype=np.float64)
        X_joint = np.hstack([X, Z])

        comp_errors: list[float] = []
        struct_errors: list[float] = []

        for i in range(N):
            train_idx = [j for j in range(N) if j != i]
            test_idx = [i]

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            Xj_tr = X_joint[train_idx]
            Xj_te = X_joint[test_idx]

            # 1. Composition-only GP on N-1
            k_comp = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[10.0] * X.shape[1], nu=2.5, length_scale_bounds=(1.0, 100.0)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1.0))
            gp_c = GaussianProcessRegressor(kernel=k_comp, n_restarts_optimizer=0, random_state=self.random_state + i, normalize_y=True)
            gp_c.fit(X_tr, y_tr)
            y_pred_c = float(gp_c.predict(X_te)[0])
            comp_errors.append((float(y_te[0]) - y_pred_c) ** 2)

            # 2. Structure-informed GP on N-1
            k_struct = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[10.0] * X_joint.shape[1], nu=2.5, length_scale_bounds=(1.0, 100.0)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1.0))
            gp_s = GaussianProcessRegressor(kernel=k_struct, n_restarts_optimizer=0, random_state=self.random_state + i, normalize_y=True)
            gp_s.fit(Xj_tr, y_tr)
            y_pred_s = float(gp_s.predict(Xj_te)[0])
            struct_errors.append((float(y_te[0]) - y_pred_s) ** 2)

        mse_comp = float(np.mean(comp_errors))
        mse_struct = float(np.mean(struct_errors))
        ratio = float((mse_comp - mse_struct) / (mse_comp + 1e-12))
        ratio = max(-1.0, min(1.0, ratio))

        return {
            "composition_mse": mse_comp,
            "structure_informed_mse": mse_struct,
            "structure_advantage_ratio": ratio,
            "note": f"Leave-One-Out CV across {N} joint observations.",
        }
