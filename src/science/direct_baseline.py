from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from src.optimization.acquisition import predict_latent_gp


class DirectPerformanceModel:
    """Direct Baseline Model: maps pre-experiment process features directly to downstream performance targets.

    P(Y | X)
    Serves as an honest statistical baseline to evaluate whether mechanistic Stage-A characterization
    modeling provides predictive, interpretability, or uncertainty advantages.
    """

    def __init__(
        self,
        process_features: list[str],
        target_column: str,
        random_state: int = 42,
    ) -> None:
        self.process_features = list(process_features)
        self.target_column = target_column
        self.random_state = random_state
        self.gp: GaussianProcessRegressor | None = None
        self.scaler: StandardScaler | None = None
        self.training_sample_count: int = 0
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> DirectPerformanceModel:
        if df.empty:
            raise ValueError("Cannot fit DirectPerformanceModel on empty DataFrame")

        cols_needed = self.process_features + [self.target_column]
        valid_df = df[cols_needed].dropna()
        if len(valid_df) < 2:
            self.is_fitted = False
            return self

        X = valid_df[self.process_features].to_numpy(dtype=float)
        y = valid_df[self.target_column].to_numpy(dtype=float)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1.0))
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=self.random_state,
        )
        self.gp.fit(X_scaled, y)
        self.training_sample_count = len(valid_df)
        self.is_fitted = True
        return self

    def predict(
        self,
        X_process: pd.DataFrame | np.ndarray,
        return_std: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted or self.gp is None or self.scaler is None:
            raise RuntimeError("DirectPerformanceModel is not fitted yet.")

        if isinstance(X_process, pd.DataFrame):
            X_mat = X_process[self.process_features].to_numpy(dtype=float)
        else:
            X_mat = np.asarray(X_process, dtype=float)

        X_scaled = self.scaler.transform(X_mat)
        return predict_latent_gp(self.gp, X_scaled, return_std=return_std)
