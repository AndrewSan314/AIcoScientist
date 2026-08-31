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
    ) -> dict[str, float]:
        """Evaluates whether structural features improve predictive error over composition alone."""
        if len(compositions) < 4:
            return {
                "composition_mse": 1.0,
                "structure_informed_mse": 1.0,
                "structure_advantage_ratio": 0.0,
            }

        X = np.asarray(compositions, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        X_joint = np.hstack([X, np.asarray(embeddings, dtype=np.float64)])

        # Leave-one-out or resubstitution proxy
        pred_comp = self._gpr.predict(X) if self._gpr is not None else np.zeros_like(y)
        mse_comp = float(mean_squared_error(y, pred_comp))

        if self._gpr_with_structure is not None:
            pred_struct = self._gpr_with_structure.predict(X_joint)
            mse_struct = float(mean_squared_error(y, pred_struct))
        else:
            mse_struct = mse_comp

        ratio = (mse_comp - mse_struct) / (mse_comp + 1e-12)
        return {
            "composition_mse": mse_comp,
            "structure_informed_mse": mse_struct,
            "structure_advantage_ratio": float(ratio),
        }
