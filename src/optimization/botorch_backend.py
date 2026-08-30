from __future__ import annotations

import logging
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import botorch
import gpytorch

from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.optimization.backend import OptimizerBackend
from src.optimization.finite_pool import FiniteCandidatePool
from src.optimization.objective import OptimizationObjective
from src.optimization.proposal import CandidateProposal

logger = logging.getLogger(__name__)


class BoTorchBackend(OptimizerBackend):
    """Production Bayesian Optimization backend delegating surrogate modeling and acquisition math to BoTorch."""

    def __init__(self, *, default_strategy: str = "expected_improvement") -> None:
        self._default_strategy = default_strategy

    @property
    def name(self) -> str:
        return "botorch"

    @property
    def version(self) -> str:
        return str(botorch.__version__)

    def _prepare_data(
        self,
        observations: pd.DataFrame | Sequence[Mapping[str, Any]],
        candidate_pool: pd.DataFrame,
        objective: OptimizationObjective | str,
        feature_columns: Sequence[str] | None,
        candidate_id_column: str | None,
    ) -> tuple[FiniteCandidatePool, FiniteCandidatePool, np.ndarray, np.ndarray, OptimizationObjective, list[str]]:
        """Prepares and validates finite candidate pools, feature matrices, and target tensors."""
        obj = OptimizationObjective.create(objective) if isinstance(objective, str) else objective

        # Inferred feature columns
        if feature_columns is not None:
            feat_cols = list(feature_columns)
        else:
            # Numeric columns in candidate pool excluding identifiers and targets
            exclude = {
                obj.target_name,
                "candidate_id",
                "sample_id",
                "policy_id",
                "experiment_id",
                "id",
                "sample_index",
                "stage",
            }
            feat_cols = [
                c
                for c in candidate_pool.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(candidate_pool[c])
            ]

        if not feat_cols:
            raise ValueError("No numeric candidate feature columns identified.")

        id_col = candidate_id_column or "candidate_id"

        full_pool = FiniteCandidatePool(candidate_pool, feature_columns=feat_cols, id_column=id_col)
        unseen_pool = full_pool.filter_unseen(observations)

        # Parse observed X and y
        if isinstance(observations, pd.DataFrame):
            obs_df = observations.copy()
        else:
            obs_df = pd.DataFrame(list(observations))

        if obs_df.empty or obj.target_name not in obs_df.columns:
            X_obs = np.empty((0, len(feat_cols)), dtype=float)
            y_obs = np.empty((0,), dtype=float)
        else:
            valid_obs = obs_df.dropna(subset=[obj.target_name] + [c for c in feat_cols if c in obs_df.columns])
            if valid_obs.empty:
                X_obs = np.empty((0, len(feat_cols)), dtype=float)
                y_obs = np.empty((0,), dtype=float)
            else:
                X_obs = valid_obs[feat_cols].to_numpy(dtype=float, copy=True)
                y_obs = valid_obs[obj.target_name].to_numpy(dtype=float, copy=True)

        return full_pool, unseen_pool, X_obs, y_obs, obj, feat_cols

    def propose(
        self,
        observations: pd.DataFrame | Sequence[Mapping[str, Any]],
        candidate_pool: pd.DataFrame,
        objective: OptimizationObjective | str,
        *,
        feature_columns: Sequence[str] | None = None,
        candidate_id_column: str | None = None,
        n: int = 1,
        seed: int | None = None,
        strategy: str | None = None,
        **kwargs: Any,
    ) -> list[CandidateProposal]:
        """Proposes next candidate(s) from finite pool using BoTorch surrogate and acquisition functions."""
        full_pool, unseen_pool, X_obs, y_obs, obj, feat_cols = self._prepare_data(
            observations, candidate_pool, objective, feature_columns, candidate_id_column
        )

        strat = (strategy or self._default_strategy).strip().lower()

        # Handle purely random strategy or cold start without observations
        if strat in {"random", "uniform"} or len(y_obs) == 0:
            return self._propose_random(unseen_pool, obj, n=n, seed=seed, strat_name=strat)

        # Build PyTorch tensors with double precision (ensuring writable contiguous buffers)
        train_X = torch.as_tensor(np.ascontiguousarray(X_obs), dtype=torch.float64)
        raw_y = torch.as_tensor(np.ascontiguousarray(y_obs), dtype=torch.float64).unsqueeze(-1)

        # BoTorch natively assumes maximization. For minimization, invert sign internally.
        train_Y = -raw_y if obj.minimize else raw_y

        # Construct SingleTaskGP with standard input & outcome transforms
        input_transform = Normalize(d=train_X.shape[-1])
        outcome_transform = Standardize(m=1)

        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

        # Fit model hyperparameters using Marginal Log Likelihood
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(mll)
            except Exception as exc:
                logger.warning(f"BoTorch MLL fitting warning: {exc}; falling back to default priors.")

        model.eval()

        # Unseen candidates feature tensor
        X_unseen_np = unseen_pool.get_feature_matrix()
        X_unseen = torch.as_tensor(np.ascontiguousarray(X_unseen_np), dtype=torch.float64)

        # 1. Compute posterior over unseen candidate pool
        with torch.no_grad():
            posterior = model.posterior(X_unseen)
            # Posterior mean and variance in internally maximized space
            post_mean_int = posterior.mean.squeeze(-1).cpu().numpy()
            post_var_int = posterior.variance.squeeze(-1).cpu().numpy()
            post_std = np.sqrt(np.maximum(post_var_int, 1e-12))

            # Convert mean back to user's real-world target orientation
            user_post_mean = -post_mean_int if obj.minimize else post_mean_int

        # 2. Evaluate strategy acquisition values over discrete candidates
        scores = self._compute_acquisition_scores(
            model=model,
            train_X=train_X,
            train_Y=train_Y,
            X_unseen=X_unseen,
            strategy=strat,
            seed=seed,
            **kwargs,
        )

        # 3. Select top n candidates (argmax of acquisition score)
        # Stable sort breaking ties deterministically
        ranked_indices = np.argsort(-scores, kind="stable")
        selected_indices = ranked_indices[:n]

        proposals: list[CandidateProposal] = []
        for rank, idx in enumerate(selected_indices):
            cid = unseen_pool.get_candidate_id(idx)
            d_vars = unseen_pool.get_design_variables(idx)
            meta = unseen_pool.get_metadata(idx)

            # Compute nearest neighbor distance to observed points in normalized feature space
            if len(X_obs) > 0:
                cand_vec = X_unseen_np[idx]
                dists = np.linalg.norm(X_obs - cand_vec, axis=1)
                min_dist = float(np.min(dists))
            else:
                min_dist = 0.0

            prop = CandidateProposal(
                candidate_id=cid,
                design_variables=d_vars,
                predicted_mean=float(user_post_mean[idx]),
                predicted_std=float(post_std[idx]),
                acquisition_name=strat,
                acquisition_value=float(scores[idx]),
                backend_name=self.name,
                backend_version=self.version,
                seed=seed,
                reason_code="BALANCED_EXPLORATION_EXPLOITATION",
                recommendation_reason=f"BoTorch {strat} acquisition score = {scores[idx]:.4f}",
                distance_to_nearest_observed=min_dist,
                step=len(y_obs) + rank + 1,
                metadata={"rank": rank, "full_metadata": meta},
            )
            proposals.append(prop)

        return proposals

    def _propose_random(
        self,
        unseen_pool: FiniteCandidatePool,
        obj: OptimizationObjective,
        n: int,
        seed: int | None,
        strat_name: str,
    ) -> list[CandidateProposal]:
        """Proposes random candidates uniformly from unseen pool."""
        rng = np.random.default_rng(seed)
        n_cand = len(unseen_pool)
        n_select = min(n, n_cand)
        chosen_indices = rng.choice(n_cand, size=n_select, replace=False)

        proposals: list[CandidateProposal] = []
        for i, idx in enumerate(chosen_indices):
            cid = unseen_pool.get_candidate_id(idx)
            d_vars = unseen_pool.get_design_variables(idx)
            meta = unseen_pool.get_metadata(idx)
            prop = CandidateProposal(
                candidate_id=cid,
                design_variables=d_vars,
                predicted_mean=0.0,
                predicted_std=1.0,
                acquisition_name=strat_name,
                acquisition_value=0.0,
                backend_name=self.name,
                backend_version=self.version,
                seed=seed,
                reason_code="UNIFORM_RANDOM_EXPLORATION",
                recommendation_reason="Uniform stochastic selection from candidate pool",
                distance_to_nearest_observed=0.0,
                step=i + 1,
                metadata={"full_metadata": meta},
            )
            proposals.append(prop)
        return proposals

    def _compute_acquisition_scores(
        self,
        model: SingleTaskGP,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        X_unseen: torch.Tensor,
        strategy: str,
        seed: int | None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Computes discrete acquisition function scores across candidate tensor using BoTorch primitives."""
        # 1. Greedy / Posterior Mean
        if strategy in {"greedy", "posterior_mean"}:
            acq_func = PosteriorMean(model=model)
            with torch.no_grad():
                scores = acq_func(X_unseen.unsqueeze(1))
            return scores.cpu().numpy().astype(float)

        # 2. GP-UCB
        elif strategy in {"gp_ucb", "ucb", "upper_confidence_bound"}:
            beta = float(kwargs.get("beta", 2.0))
            acq_func = UpperConfidenceBound(model=model, beta=beta)
            with torch.no_grad():
                scores = acq_func(X_unseen.unsqueeze(1))
            return scores.cpu().numpy().astype(float)

        # 3. Expected Improvement (LogEI or analytic EI)
        elif strategy in {"ei", "expected_improvement", "log_ei", "log_expected_improvement"}:
            best_f = train_Y.max()
            try:
                acq_func = LogExpectedImprovement(model=model, best_f=best_f)
                with torch.no_grad():
                    scores = acq_func(X_unseen.unsqueeze(1))
            except Exception:
                acq_func = ExpectedImprovement(model=model, best_f=best_f)
                with torch.no_grad():
                    scores = acq_func(X_unseen.unsqueeze(1))
            return scores.cpu().numpy().astype(float)

        # 4. Noisy Expected Improvement (LogNEI / qLogNEI)
        elif strategy in {"nei", "noisy_expected_improvement", "log_nei", "log_noisy_expected_improvement", "turbo_nei", "turbo_ei"}:
            with torch.no_grad():
                try:
                    if seed is not None:
                        torch.manual_seed(seed)
                    acq_func = qLogNoisyExpectedImprovement(
                        model=model,
                        X_baseline=train_X,
                        prune_baseline=True,
                    )
                    scores_list = []
                    for chunk in torch.split(X_unseen.unsqueeze(1), 128):
                        scores_list.append(acq_func(chunk))
                    scores = torch.cat(scores_list, dim=0)
                except Exception as exc:
                    logger.warning(f"qLogNEI fallback: {exc}; evaluating standard EI")
                    best_f = train_Y.max()
                    acq_func = LogExpectedImprovement(model=model, best_f=best_f)
                    scores = acq_func(X_unseen.unsqueeze(1))

            return scores.cpu().numpy().astype(float)

        # 5. Joint Thompson Sampling (discrete posterior joint draws)
        elif strategy in {"thompson", "ts", "thompson_sampling"}:
            if seed is not None:
                torch.manual_seed(seed)
            with torch.no_grad():
                posterior = model.posterior(X_unseen)
                joint_sample = posterior.rsample(sample_shape=torch.Size([1]))
                scores = joint_sample.squeeze(0).squeeze(-1)
            return scores.cpu().numpy().astype(float)

        else:
            raise ValueError(f"Unsupported optimization strategy {strategy!r} for BoTorch backend.")
