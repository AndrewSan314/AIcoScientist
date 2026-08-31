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

from src.optimization.backend import (
    AcquisitionEvaluationError,
    OptimizerBackend,
    STRATEGY_ALIASES,
    SUPPORTED_STRATEGIES,
    UnsupportedStrategyError,
    resolve_strategy,
)
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
        strict_identity: bool = True,
    ) -> tuple[FiniteCandidatePool, FiniteCandidatePool, np.ndarray, np.ndarray, OptimizationObjective, list[str]]:
        """Prepares and validates finite candidate pools, feature matrices, and target tensors."""
        obj = OptimizationObjective.create(objective) if isinstance(objective, str) else objective

        # Validate objective capabilities
        if obj.constraints:
            raise NotImplementedError("Objective constraints are not currently supported by BoTorchBackend.")
        if obj.is_multiobjective:
            raise NotImplementedError("Multi-objective optimization is not currently supported by BoTorchBackend.")
        if obj.threshold is not None:
            raise NotImplementedError("Objective threshold semantics are not currently supported by BoTorchBackend.")

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

        full_pool = FiniteCandidatePool(
            candidate_pool,
            feature_columns=feat_cols,
            id_column=id_col,
            strict_identity=strict_identity,
        )
        unseen_pool = full_pool.filter_unseen(observations)

        # Parse observed X and y
        if isinstance(observations, pd.DataFrame):
            obs_df = observations.copy()
            if id_col not in obs_df.columns:
                for fallback_id in ("candidate_id", "sample_id", "policy_id", "experiment_id", "id"):
                    if fallback_id in obs_df.columns:
                        obs_df[id_col] = obs_df[fallback_id]
                        break
            if strict_identity and not obs_df.empty:
                if id_col not in obs_df.columns or obs_df[id_col].isna().any():
                    raise ValueError(
                        f"Observations dataframe must contain non-null candidate ID column {id_col!r} "
                        "under strict candidate identity mode."
                    )
        else:
            obs_list = list(observations)
            if strict_identity and obs_list:
                for r in obs_list:
                    if isinstance(r, (dict, Mapping)):
                        cid = r.get(id_col) or r.get("candidate_id") or r.get("sample_id") or r.get("policy_id")
                        if cid is None or pd.isna(cid) or str(cid).strip() == "":
                            raise ValueError(
                                f"Observation mapping is missing non-null candidate ID ({id_col!r}) "
                                f"under strict identity mode: {r}"
                            )
            obs_df = pd.DataFrame(obs_list)
            if id_col not in obs_df.columns and "candidate_id" in obs_df.columns:
                obs_df[id_col] = obs_df["candidate_id"]

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
        strict_identity: bool = True,
        **kwargs: Any,
    ) -> list[CandidateProposal]:
        """Proposes next candidate(s) from finite pool using BoTorch surrogate and acquisition functions."""
        requested_strat = strategy or self._default_strategy
        canonical_strat = resolve_strategy(requested_strat)

        full_pool, unseen_pool, X_obs, y_obs, obj, feat_cols = self._prepare_data(
            observations,
            candidate_pool,
            objective,
            feature_columns,
            candidate_id_column,
            strict_identity=strict_identity,
        )

        # Handle purely random strategy or cold start without observations
        if canonical_strat == "random" or len(y_obs) == 0:
            return self._propose_random(
                unseen_pool,
                obj,
                n=n,
                seed=seed,
                requested_strat=requested_strat,
                canonical_strat=canonical_strat,
            )

        # Build PyTorch tensors with double precision (ensuring writable contiguous buffers)
        train_X = torch.as_tensor(np.ascontiguousarray(X_obs).copy(), dtype=torch.float64)
        raw_y = torch.as_tensor(np.ascontiguousarray(y_obs).copy(), dtype=torch.float64).unsqueeze(-1)

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
        X_unseen = torch.as_tensor(np.ascontiguousarray(X_unseen_np).copy(), dtype=torch.float64)

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
        scores, actual_strat_name, acq_class_name = self._compute_acquisition_scores(
            model=model,
            train_X=train_X,
            train_Y=train_Y,
            X_unseen=X_unseen,
            strategy=canonical_strat,
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

            prop_meta = {
                "rank": rank,
                "requested_strategy": requested_strat,
                "canonical_strategy": canonical_strat,
                "actual_strategy": actual_strat_name,
                "backend_name": self.name,
                "backend_version": self.version,
                "model_class": "SingleTaskGP",
                "acquisition_class": acq_class_name,
                "batch_semantics": "top_n_individual_scores",
                "batch_requested": n,
                "seed": seed,
                "full_metadata": meta,
            }

            prop = CandidateProposal(
                candidate_id=cid,
                design_variables=d_vars,
                predicted_mean=float(user_post_mean[idx]),
                predicted_std=float(post_std[idx]),
                acquisition_name=requested_strat,
                acquisition_value=float(scores[idx]),
                backend_name=self.name,
                backend_version=self.version,
                seed=seed,
                reason_code="BALANCED_EXPLORATION_EXPLOITATION",
                recommendation_reason=f"BoTorch {actual_strat_name} acquisition score = {scores[idx]:.4f}",
                distance_to_nearest_observed=min_dist,
                step=len(y_obs) + rank + 1,
                metadata=prop_meta,
            )
            proposals.append(prop)

        return proposals

    def _propose_random(
        self,
        unseen_pool: FiniteCandidatePool,
        obj: OptimizationObjective,
        n: int,
        seed: int | None,
        requested_strat: str,
        canonical_strat: str = "random",
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
            prop_meta = {
                "rank": i,
                "requested_strategy": requested_strat,
                "canonical_strategy": canonical_strat,
                "actual_strategy": "uniform_random",
                "backend_name": self.name,
                "backend_version": self.version,
                "model_class": "None",
                "acquisition_class": "UniformRandom",
                "batch_semantics": "top_n_individual_scores",
                "batch_requested": n,
                "seed": seed,
                "full_metadata": meta,
            }
            prop = CandidateProposal(
                candidate_id=cid,
                design_variables=d_vars,
                predicted_mean=0.0,
                predicted_std=1.0,
                acquisition_name=requested_strat,
                acquisition_value=0.0,
                backend_name=self.name,
                backend_version=self.version,
                seed=seed,
                reason_code="PURE_EXPLORATION",
                recommendation_reason="Uniform random sampling from finite candidate space",
                distance_to_nearest_observed=0.0,
                step=i + 1,
                metadata=prop_meta,
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
    ) -> tuple[np.ndarray, str, str]:
        """Computes discrete acquisition function scores across candidate tensor using BoTorch primitives.

        Returns:
            scores: numpy array of acquisition scores
            actual_strategy: canonical algorithm name
            acquisition_class: exact BoTorch acquisition class evaluated
        """
        canonical = resolve_strategy(strategy)

        # 1. Greedy / Posterior Mean
        if canonical == "greedy":
            acq_func = PosteriorMean(model=model)
            with torch.no_grad():
                scores = acq_func(X_unseen.unsqueeze(1))
            return scores.cpu().numpy().astype(float), "posterior_mean", "PosteriorMean"

        # 2. GP-UCB
        elif canonical == "gp_ucb":
            beta = float(kwargs.get("beta", 2.0))
            acq_func = UpperConfidenceBound(model=model, beta=beta)
            with torch.no_grad():
                scores = acq_func(X_unseen.unsqueeze(1))
            return scores.cpu().numpy().astype(float), "gp_ucb", "UpperConfidenceBound"

        # 3. Expected Improvement (LogEI or analytic EI)
        elif canonical == "expected_improvement":
            best_f = train_Y.max()
            try:
                acq_func = LogExpectedImprovement(model=model, best_f=best_f)
                with torch.no_grad():
                    scores = acq_func(X_unseen.unsqueeze(1))
                return scores.cpu().numpy().astype(float), "log_expected_improvement", "LogExpectedImprovement"
            except Exception:
                acq_func = ExpectedImprovement(model=model, best_f=best_f)
                with torch.no_grad():
                    scores = acq_func(X_unseen.unsqueeze(1))
                return scores.cpu().numpy().astype(float), "expected_improvement", "ExpectedImprovement"

        # 4. Noisy Expected Improvement (qLogNEI) - FAILS CLOSED ON ERROR
        elif canonical == "noisy_expected_improvement":
            try:
                with torch.no_grad():
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
                return scores.cpu().numpy().astype(float), "qLogNoisyExpectedImprovement", "qLogNoisyExpectedImprovement"
            except Exception as exc:
                raise AcquisitionEvaluationError(
                    f"Failed to evaluate NEI acquisition function {strategy!r} on BoTorch backend: {exc}. "
                    f"Context: backend=botorch, botorch_version={botorch.__version__}, torch_version={torch.__version__}, "
                    f"n_candidates={len(X_unseen)}, n_observations={len(train_X)}."
                ) from exc

        # 5. Joint Thompson Sampling (discrete posterior joint draws)
        elif canonical == "thompson":
            if seed is not None:
                torch.manual_seed(seed)
            with torch.no_grad():
                posterior = model.posterior(X_unseen)
                joint_sample = posterior.rsample(sample_shape=torch.Size([1]))
                scores = joint_sample.squeeze(0).squeeze(-1)
            return scores.cpu().numpy().astype(float), "thompson_sampling", "JointPosteriorSampling"

        else:
            raise UnsupportedStrategyError(f"Unsupported optimization strategy {strategy!r} for BoTorch backend.")
