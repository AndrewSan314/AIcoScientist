from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from .process_objective import ConstraintSpec, ProcessOptimizationObjective
from .proposal import Prediction, ProcessControlProposal
from .process_space import ProcessSearchSpace


class UnsupportedProcessOptimizationError(NotImplementedError):
    pass


class OfficialMultiObjectiveBoTorch:
    """Official BoTorch qNEHVI over an audited finite process-recipe pool.

    Constraints are fail-closed: a hard constraint must name a modeled outcome or
    existing candidate column. Unsupported semantics raise instead of scalarizing.
    """

    def propose(
        self,
        observations: pd.DataFrame,
        space: ProcessSearchSpace,
        objective: ProcessOptimizationObjective,
        *,
        n: int = 1,
        seed: int | None = None,
    ) -> list[ProcessControlProposal]:
        if not objective.is_multiobjective:
            raise UnsupportedProcessOptimizationError("qNEHVI requires at least two objectives")
        target_names = [item.target for item in objective.objectives]
        self._validate_constraint_semantics(objective.constraints, space.control_columns, target_names)
        required = [space.id_column, *space.control_columns, *target_names]
        missing = [column for column in required if column not in observations]
        if missing:
            raise ValueError(f"observations lack required identity/control/target columns: {missing}")
        observed = observations.dropna(subset=required).copy()
        if len(observed) < 2:
            raise ValueError("qNEHVI needs at least two fully observed source recipes")
        unseen = space.candidates.loc[~space.candidates[space.id_column].isin(observed[space.id_column])].copy()
        unseen = self._filter_hard_control_constraints(unseen, objective.constraints)
        if unseen.empty:
            raise ValueError("no unobserved feasible process recipe remains")

        try:
            from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import ModelListGP, SingleTaskGP
            from botorch.models.transforms.outcome import Standardize
            from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
        except ImportError as exc:
            raise UnsupportedProcessOptimizationError("multi-objective process optimization requires the official botorch dependency") from exc
        X_values, candidate_values = self._scaled_inputs(observed, unseen, space)
        X = torch.as_tensor(X_values, dtype=torch.double)
        candidate_X = torch.as_tensor(candidate_values, dtype=torch.double)
        models = []
        transformed_targets = []
        for item in objective.objectives:
            y = observed[item.target].to_numpy(dtype=float)
            if item.sense == "minimize":
                y = -y
            transformed_targets.append(y)
            model = SingleTaskGP(X, torch.as_tensor(y[:, None], dtype=torch.double), outcome_transform=Standardize(m=1))
            models.append(model)
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)
        model.eval()
        Y = torch.as_tensor(np.column_stack(transformed_targets), dtype=torch.double)
        reference = self._reference_point(Y, objective)
        constraints = self._outcome_constraints(objective.constraints, target_names, objective.objectives)
        if seed is not None:
            torch.manual_seed(seed)
        try:
            acquisition = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=reference.tolist(),
                X_baseline=X,
                prune_baseline=True,
                constraints=constraints or None,
            )
            with torch.no_grad():
                scores = acquisition(candidate_X.unsqueeze(-2)).detach().cpu().numpy()
        except Exception as exc:  # no fake fallback: scientifically different behavior is rejected
            raise UnsupportedProcessOptimizationError(f"official qNEHVI evaluation failed: {exc}") from exc
        posterior = model.posterior(candidate_X)
        means = posterior.mean.detach().cpu().numpy()
        stds = posterior.variance.clamp_min(0).sqrt().detach().cpu().numpy()
        ranks = self._pareto_ranks(means)
        selected = np.argsort(-scores, kind="stable")[: min(n, len(unseen))]
        result: list[ProcessControlProposal] = []
        for index in selected:
            outputs = {
                item.target: Prediction(float(-means[index, pos] if item.sense == "minimize" else means[index, pos]), float(stds[index, pos]), item.units)
                for pos, item in enumerate(objective.objectives)
            }
            controls = unseen.iloc[index][space.control_columns].to_dict()
            result.append(ProcessControlProposal(
                proposal_id=f"process:{unseen.iloc[index][space.id_column]}", stage=None, controls=controls,
                predicted_outputs=outputs, feasibility_probability=self._feasibility_probability(outputs, objective.constraints),
                acquisition_value=float(scores[index]), pareto_rank=int(ranks[index]), model_version="botorch-qNEHVI",
                data_fingerprint=self._fingerprint(observed), source_recipe_id=str(unseen.iloc[index][space.id_column]),
                provenance={"acquisition": "qNoisyExpectedHypervolumeImprovement", "reference_point": reference.tolist(), "seed": seed},
            ))
        return result

    @staticmethod
    def _reference_point(Y: torch.Tensor, objective: ProcessOptimizationObjective) -> np.ndarray:
        if objective.reference_point is not None:
            reference = np.asarray(objective.reference_point, dtype=float)
            return np.asarray([(-value if item.sense == "minimize" else value) for value, item in zip(reference, objective.objectives)], dtype=float)
        return (Y.min(dim=0).values - 0.01 * Y.std(dim=0).clamp_min(1e-8)).cpu().numpy()

    @staticmethod
    def _filter_hard_control_constraints(pool: pd.DataFrame, constraints: Sequence[ConstraintSpec]) -> pd.DataFrame:
        filtered = pool.copy()
        for constraint in constraints:
            if constraint.hard and constraint.name in filtered:
                values = pd.to_numeric(filtered[constraint.name], errors="coerce")
                if constraint.type == "lower":
                    filtered = filtered.loc[values >= float(constraint.threshold)]
                elif constraint.type == "upper":
                    filtered = filtered.loc[values <= float(constraint.threshold)]
                elif constraint.type == "range":
                    low, high = constraint.threshold
                    filtered = filtered.loc[(values >= low) & (values <= high)]
        return filtered

    @staticmethod
    def _validate_constraint_semantics(
        constraints: Sequence[ConstraintSpec], control_names: Sequence[str], target_names: Sequence[str],
    ) -> None:
        controls, targets = set(control_names), set(target_names)
        for constraint in constraints:
            if not constraint.hard:
                raise UnsupportedProcessOptimizationError("soft process constraints are not implemented; use an explicit objective")
            if constraint.name not in controls | targets:
                raise UnsupportedProcessOptimizationError(
                    f"constraint {constraint.name!r} must name a modeled outcome or finite-pool control"
                )
            if constraint.name in controls and constraint.type == "feasibility":
                raise UnsupportedProcessOptimizationError("feasibility constraints require a modeled probability outcome")

    @staticmethod
    def _scaled_inputs(
        observed: pd.DataFrame, unseen: pd.DataFrame, space: ProcessSearchSpace,
    ) -> tuple[np.ndarray, np.ndarray]:
        controls = space.control_columns
        pool = space.candidates[controls].apply(pd.to_numeric, errors="coerce")
        history = observed[controls].apply(pd.to_numeric, errors="coerce")
        candidates = unseen[controls].apply(pd.to_numeric, errors="coerce")
        if any(frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all() for frame in (pool, history, candidates)):
            raise ValueError("official process qNEHVI requires finite numeric recipe controls")
        lower, span = pool.min(), pool.max() - pool.min()
        span = span.mask(span == 0, 1.0)
        scaled_history = (history - lower) / span
        scaled_candidates = (candidates - lower) / span
        if ((scaled_history < -1e-12) | (scaled_history > 1 + 1e-12)).any().any():
            raise ValueError("observed controls fall outside the audited finite recipe pool")
        return scaled_history.to_numpy(), scaled_candidates.to_numpy()

    @staticmethod
    def _outcome_constraints(constraints: Sequence[ConstraintSpec], names: list[str], objectives: Sequence[object]):
        functions = []
        for constraint in constraints:
            if constraint.name not in names:
                continue
            index = names.index(constraint.name)
            direction = objectives[index].sense
            threshold = float(constraint.threshold) if not isinstance(constraint.threshold, tuple) else None
            if constraint.type == "lower":
                functions.append(
                    (lambda Y, i=index, t=threshold: t - Y[..., i]) if direction == "maximize"
                    else (lambda Y, i=index, t=threshold: Y[..., i] + t)
                )
            elif constraint.type == "upper":
                functions.append(
                    (lambda Y, i=index, t=threshold: Y[..., i] - t) if direction == "maximize"
                    else (lambda Y, i=index, t=threshold: -t - Y[..., i])
                )
            elif constraint.type == "range":
                low, high = constraint.threshold
                if direction == "maximize":
                    functions.extend([lambda Y, i=index, v=low: v - Y[..., i], lambda Y, i=index, v=high: Y[..., i] - v])
                else:
                    functions.extend([lambda Y, i=index, v=low: Y[..., i] + v, lambda Y, i=index, v=high: -v - Y[..., i]])
            elif constraint.type == "feasibility":
                if direction != "maximize":
                    raise UnsupportedProcessOptimizationError("feasibility outcomes must be maximized probabilities")
                functions.append(lambda Y, i=index, t=threshold: t - Y[..., i])
        return functions

    @staticmethod
    def _pareto_ranks(means: np.ndarray) -> np.ndarray:
        from botorch.utils.multi_objective.pareto import is_non_dominated

        values = torch.as_tensor(means)
        ranks = np.ones(len(means), dtype=int)
        ranks[is_non_dominated(values).cpu().numpy()] = 0
        return ranks

    @staticmethod
    def _feasibility_probability(outputs: dict[str, Prediction], constraints: Sequence[ConstraintSpec]) -> float | None:
        relevant = [constraint for constraint in constraints if constraint.name in outputs]
        if not relevant:
            return None
        return float(all(constraint.satisfied(outputs[constraint.name].mean) for constraint in relevant))

    @staticmethod
    def _fingerprint(observed: pd.DataFrame) -> str:
        return str(pd.util.hash_pandas_object(observed, index=True).sum())
