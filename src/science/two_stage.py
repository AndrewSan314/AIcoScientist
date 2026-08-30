from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from src.datasets.base import TwoStageModelSpec
from src.optimization.acquisition import predict_latent_gp


@dataclass
class TwoStagePrediction:
    """End-to-end prediction output with decomposed uncertainty under the Law of Total Variance."""

    performance_mean: np.ndarray
    performance_latent_std: np.ndarray
    characterization_propagation_variance: np.ndarray
    performance_model_variance: np.ndarray
    characterization_predictions: dict[str, dict[str, np.ndarray]]
    n_mc_samples: int

    @property
    def total_variance(self) -> np.ndarray:
        return self.characterization_propagation_variance + self.performance_model_variance


class StageACharacterizationModel:
    """Stage A: Predicts physical/spectroscopic characterization channels from pre-experiment process features.

    P(C | X)
    Modeled via independent Gaussian Processes per characterization channel.
    Posterior cross-correlation between channels is ignored in v1.
    """

    def __init__(self, process_features: list[str], characterization_targets: list[str], random_state: int = 42) -> None:
        self.process_features = list(process_features)
        self.characterization_targets = list(characterization_targets)
        self.random_state = random_state
        self.models: dict[str, GaussianProcessRegressor] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.training_sample_counts: dict[str, int] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> StageACharacterizationModel:
        if df.empty:
            raise ValueError("Cannot fit Stage A on empty DataFrame")

        self.models.clear()
        self.scalers.clear()
        self.training_sample_counts.clear()

        for i, char_col in enumerate(self.characterization_targets):
            if char_col not in df.columns:
                continue

            # Filter rows where both process features and this characterization channel are present and non-null
            cols_needed = self.process_features + [char_col]
            valid_df = df[cols_needed].dropna()
            if len(valid_df) < 2:
                # Insufficient data to fit a reliable GP
                continue

            X = valid_df[self.process_features].to_numpy(dtype=float)
            y = valid_df[char_col].to_numpy(dtype=float)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1.0))
            )

            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=self.random_state + i * 13,
            )
            gp.fit(X_scaled, y)

            self.models[char_col] = gp
            self.scalers[char_col] = scaler
            self.training_sample_counts[char_col] = len(valid_df)

        self.is_fitted = len(self.models) > 0
        return self

    def predict(
        self,
        X_process: pd.DataFrame | np.ndarray,
        return_std: bool = True,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Predicts mean, latent_std, and noisy observation_std for all fitted characterization channels."""
        if not self.is_fitted:
            raise RuntimeError("Stage A model is not fitted yet.")

        if isinstance(X_process, pd.DataFrame):
            X_mat = X_process[self.process_features].to_numpy(dtype=float)
        else:
            X_mat = np.asarray(X_process, dtype=float)

        predictions: dict[str, dict[str, np.ndarray]] = {}

        for char_col in self.characterization_targets:
            if char_col not in self.models:
                continue

            gp = self.models[char_col]
            scaler = self.scalers[char_col]
            X_scaled = scaler.transform(X_mat)

            # Latent epistemic uncertainty (without observation noise)
            latent_mean, latent_std = predict_latent_gp(gp, X_scaled, return_std=True)
            # Standard predictive uncertainty (including observation noise)
            _, noisy_std = gp.predict(X_scaled, return_std=True)

            predictions[char_col] = {
                "mean": latent_mean,
                "latent_std": latent_std,
                "observation_std": noisy_std,
            }

        return predictions


class StageBPerformanceModel:
    """Stage B: Predicts downstream performance properties given both process and characterization features.

    P(Y | X, C)
    Modeled via independent Gaussian Processes per performance target.
    """

    def __init__(
        self,
        process_features: list[str],
        characterization_targets: list[str],
        performance_targets: list[str],
        random_state: int = 42,
    ) -> None:
        self.process_features = list(process_features)
        self.characterization_targets = list(characterization_targets)
        self.performance_targets = list(performance_targets)
        self.input_features = self.process_features + self.characterization_targets
        self.random_state = random_state
        self.models: dict[str, GaussianProcessRegressor] = {}
        self.scaler: StandardScaler | None = None
        self.training_sample_count: int = 0
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> StageBPerformanceModel:
        if df.empty:
            raise ValueError("Cannot fit Stage B on empty DataFrame")

        self.models.clear()
        cols_needed = self.input_features + self.performance_targets
        present_cols = [c for c in cols_needed if c in df.columns]
        valid_df = df[present_cols].dropna()

        if len(valid_df) < 2:
            self.is_fitted = False
            return self

        X = valid_df[self.input_features].to_numpy(dtype=float)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.training_sample_count = len(valid_df)

        for i, perf_col in enumerate(self.performance_targets):
            if perf_col not in valid_df.columns:
                continue

            y = valid_df[perf_col].to_numpy(dtype=float)
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1.0))
            )

            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=self.random_state + i * 17,
            )
            gp.fit(X_scaled, y)
            self.models[perf_col] = gp

        self.is_fitted = len(self.models) > 0
        return self

    def predict(
        self,
        X_full: pd.DataFrame | np.ndarray,
        return_std: bool = True,
        target_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predicts performance for concatenated [X_process, C_characterization]."""
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("Stage B model is not fitted yet.")

        if target_name is None:
            target_name = self.performance_targets[0]

        if target_name not in self.models:
            raise KeyError(f"Performance target {target_name!r} not in fitted Stage B models.")

        if isinstance(X_full, pd.DataFrame):
            X_mat = X_full[self.input_features].to_numpy(dtype=float)
        else:
            X_mat = np.asarray(X_full, dtype=float)

        X_scaled = self.scaler.transform(X_mat)
        gp = self.models[target_name]
        return predict_latent_gp(gp, X_scaled, return_std=return_std)


class TwoStageScientificModel:
    """Generic Two-Stage Process -> Structure/Characterization -> Property/Performance Model.

    Combines Stage A P(C | X) and Stage B P(Y | X, C) with Monte Carlo uncertainty propagation:
    E[Y | x] = E_C[ E[Y | x, C] ]
    Var[Y | x] = E_C[ Var[Y | x, C] ] + Var_C( E[Y | x, C] )
    """

    def __init__(self, spec: TwoStageModelSpec, random_state: int = 42) -> None:
        self.spec = spec
        self.random_state = random_state
        self.stage_a = StageACharacterizationModel(
            process_features=spec.process_features,
            characterization_targets=spec.characterization_targets,
            random_state=random_state,
        )
        self.stage_b = StageBPerformanceModel(
            process_features=spec.process_features,
            characterization_targets=spec.characterization_targets,
            performance_targets=spec.performance_targets,
            random_state=random_state + 100,
        )

    @property
    def is_fitted(self) -> bool:
        return self.stage_a.is_fitted and self.stage_b.is_fitted

    def fit(self, df: pd.DataFrame) -> TwoStageScientificModel:
        self.stage_a.fit(df)
        self.stage_b.fit(df)
        return self

    def predict_characterization(
        self,
        X_process: pd.DataFrame | np.ndarray,
    ) -> dict[str, dict[str, np.ndarray]]:
        return self.stage_a.predict(X_process)

    def predict_performance_with_observed_characterization(
        self,
        X_process: pd.DataFrame | np.ndarray,
        C_observed: pd.DataFrame | np.ndarray,
        target_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Diagnostic evaluation using actual observed characterization."""
        if isinstance(X_process, pd.DataFrame) and isinstance(C_observed, pd.DataFrame):
            X_full = pd.concat([X_process.reset_index(drop=True), C_observed.reset_index(drop=True)], axis=1)
        else:
            X_p = np.asarray(X_process, dtype=float)
            C_o = np.asarray(C_observed, dtype=float)
            X_full = np.hstack([X_p, C_o])

        return self.stage_b.predict(X_full, return_std=True, target_name=target_name)

    def predict_end_to_end(
        self,
        X_process: pd.DataFrame | np.ndarray,
        target_name: str | None = None,
        n_mc_samples: int = 64,
        seed: int | None = None,
    ) -> TwoStagePrediction:
        """Evaluates end-to-end Process -> Performance with Monte Carlo uncertainty propagation.

        Critical rule: Does NOT require or use ground-truth post-experiment characterization.
        """
        if not self.is_fitted:
            raise RuntimeError("TwoStageScientificModel is not fully fitted.")

        if target_name is None:
            target_name = self.spec.performance_targets[0]

        if isinstance(X_process, pd.DataFrame):
            X_mat = X_process[self.spec.process_features].to_numpy(dtype=float)
        else:
            X_mat = np.asarray(X_process, dtype=float)

        n_points = len(X_mat)
        mc_seed = seed if seed is not None else self.random_state
        rng = np.random.default_rng(mc_seed)

        # 1. Stage A predictions
        char_preds = self.stage_a.predict(X_mat)

        # 2. Draw Monte Carlo characterization samples per point
        # For each characterization channel j, sample K draws: C_j^(k) ~ N(mu_j, std_j^2)
        sampled_chars: dict[str, np.ndarray] = {}  # shape: (n_points, n_mc_samples)
        for char_col in self.spec.characterization_targets:
            pred = char_preds[char_col]
            mu = pred["mean"][:, None]  # shape (N, 1)
            std = np.maximum(pred["latent_std"][:, None], 1e-6)  # shape (N, 1)
            # Sample standard normal (N, K)
            z = rng.standard_normal(size=(n_points, n_mc_samples))
            sampled_chars[char_col] = mu + std * z

        # 3. Evaluate Stage B for all MC draws
        mc_means = np.zeros((n_points, n_mc_samples), dtype=float)
        mc_vars = np.zeros((n_points, n_mc_samples), dtype=float)

        for k in range(n_mc_samples):
            # Assemble X_full for sample k: [X_process, C_1^(k), C_2^(k), ...]
            char_k_cols = [sampled_chars[c][:, k : k + 1] for c in self.spec.characterization_targets]
            X_full_k = np.hstack([X_mat, *char_k_cols])

            m_k, s_k = self.stage_b.predict(X_full_k, return_std=True, target_name=target_name)
            mc_means[:, k] = m_k
            mc_vars[:, k] = s_k**2

        # 4. Law of Total Variance decomposition
        # E[Y | x] = mean_k(mu_y^(k))
        e_y = np.mean(mc_means, axis=1)

        # Var_perf = mean_k(sigma_y^(k)^2)
        performance_model_var = np.mean(mc_vars, axis=1)

        # Var_char_prop = Var_k(mu_y^(k))
        char_prop_var = np.var(mc_means, axis=1, ddof=1) if n_mc_samples > 1 else np.zeros(n_points)

        total_var = performance_model_var + char_prop_var
        total_std = np.sqrt(np.maximum(total_var, 1e-12))

        return TwoStagePrediction(
            performance_mean=e_y,
            performance_latent_std=total_std,
            characterization_propagation_variance=char_prop_var,
            performance_model_variance=performance_model_var,
            characterization_predictions=char_preds,
            n_mc_samples=n_mc_samples,
        )
