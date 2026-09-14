from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import pandas as pd
from src.optimization.objective import OptimizationObjective as OfficialOptimizationObjective

if TYPE_CHECKING:
    from src.optimization.objective import OptimizationObjective

from .contracts import BatteryProcessRun
from .information_horizon import InformationHorizon
from .optimization.botorch import OfficialMultiObjectiveBoTorch, UnsupportedProcessOptimizationError
from .optimization.process_objective import ConstraintSpec, ProcessOptimizationObjective
from .optimization.process_space import ProcessSearchSpace
from .optimization.proposal import Prediction, ProcessControlProposal
from .stages import ProcessStage
from .validation import validate_process_run


@dataclass
class ProcessOptimizationCoordinator:
    """Composition root for the default battery process path, without HIG imports."""

    scalar_backend: Any | None = None
    multiobjective_backend: OfficialMultiObjectiveBoTorch | None = None
    manufacturability_model: Any | None = None

    def available_state(self, run: BatteryProcessRun, decision_stage: ProcessStage):
        report = validate_process_run(run)
        if not report.valid:
            raise ValueError("invalid process run: " + "; ".join(report.errors))
        return InformationHorizon(decision_stage).project(run)

    @staticmethod
    def _scale_official_botorch_inputs(
        observations: pd.DataFrame,
        space: ProcessSearchSpace,
    ) -> tuple[pd.DataFrame, ProcessSearchSpace]:
        """Encode audited finite-pool controls and contextual state features."""
        return space.official_botorch_view(observations)

    def _scalar_constraint_space(self, space: ProcessSearchSpace, constraints: list[ConstraintSpec]) -> tuple[ProcessSearchSpace, dict[str, Any]]:
        """Apply only deterministic finite-pool control constraints; reject all else."""
        controls = set(space.control_columns)
        candidates, probabilities = space.candidates.copy(), {}
        for constraint in constraints:
            if not constraint.hard:
                raise UnsupportedProcessOptimizationError("soft scalar process constraints are not implemented; use an explicit objective")
            if constraint.name in controls:
                if constraint.type == "feasibility":
                    raise UnsupportedProcessOptimizationError("feasibility constraints require a learned manufacturability model")
                continue
            if constraint.type != "feasibility":
                raise UnsupportedProcessOptimizationError(
                    f"scalar backend cannot model outcome constraint {constraint.name!r}; configure a constrained outcome backend"
                )
            model = self.manufacturability_model
            if model is None or constraint.name != model.label_name:
                raise UnsupportedProcessOptimizationError("feasibility constraints require a matching learned manufacturability model")
            predictions = model.predict(candidates)
            probabilities = dict(zip(candidates[space.id_column].astype(str), predictions))
            candidates = candidates.loc[[item.probability >= float(constraint.threshold) for item in predictions]].copy()
        candidates = OfficialMultiObjectiveBoTorch._filter_hard_control_constraints(candidates, constraints)
        if candidates.empty:
            raise ValueError("no feasible process recipe remains after hard control constraints")
        return ProcessSearchSpace.from_finite_pool(
            candidates, id_column=space.id_column, context_columns=space.context_columns,
            context_bounds=space.context_bounds, source_recipe_id_column=space.source_recipe_id_column,
            control_action_id_column=space.control_action_id_column, metadata_columns=space.metadata_columns,
        ), probabilities

    def propose_recipes(
        self,
        observations: pd.DataFrame,
        space: ProcessSearchSpace,
        objective: ProcessOptimizationObjective,
        *,
        n: int = 1,
        seed: int | None = None,
        strategy: str | None = None,
    ) -> list[ProcessControlProposal]:
        if objective.is_multiobjective:
            if strategy is not None:
                raise UnsupportedProcessOptimizationError("multi-objective process strategy selection is unsupported; use the official qNEHVI path")
            backend = self.multiobjective_backend or OfficialMultiObjectiveBoTorch()
            return backend.propose(observations, space, objective, n=n, seed=seed)
        target = objective.objectives[0]
        constrained_space, feasibility = self._scalar_constraint_space(space, objective.constraints)
        if self.scalar_backend is None:
            try:
                from src.optimization.botorch_backend import BoTorchBackend
            except ImportError as exc:
                raise RuntimeError("scalar process optimization requires the official botorch dependency") from exc
            backend = BoTorchBackend()
            backend_observations, backend_space = self._scale_official_botorch_inputs(observations, constrained_space)
        else:
            backend = self.scalar_backend
            backend_observations, backend_space = observations, constrained_space
        proposals = backend.propose(
            observations=backend_observations,
            candidate_pool=backend_space.candidates,
            objective=OfficialOptimizationObjective(target_name=target.target, minimize=target.sense == "minimize"),
            feature_columns=backend_space.model_columns,
            candidate_id_column=backend_space.id_column,
            n=n,
            seed=seed,
            strategy=strategy or "noisy_expected_improvement",
        )
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(observations, index=True).values.tobytes()).hexdigest()
        source_candidates = constrained_space.candidates.set_index(constrained_space.id_column)
        result: list[ProcessControlProposal] = []
        for item in proposals:
            row = source_candidates.loc[item.candidate_id]
            provenance = dict(item.metadata)
            provenance.update({"candidate_instance_id": str(item.candidate_id)})
            if "context_provenance_fingerprint" in row:
                provenance["context_provenance_fingerprint"] = row["context_provenance_fingerprint"]
            if "source_recipe_ids" in row:
                provenance["source_recipe_ids"] = list(row["source_recipe_ids"])
            source_recipe_id = row[constrained_space.source_id_column] if constrained_space.source_id_column in row.index else item.candidate_id
            control_action_id = row[constrained_space.control_action_id_column] if constrained_space.control_action_id_column else None
            result.append(ProcessControlProposal(
                proposal_id=f"process:{item.candidate_id}", stage=None, controls=row[constrained_space.control_columns].to_dict(),
                predicted_outputs={target.target: Prediction(item.predicted_mean, item.predicted_std, target.units)},
                feasibility_probability=feasibility.get(str(item.candidate_id)).probability if str(item.candidate_id) in feasibility else None, acquisition_value=item.acquisition_value, pareto_rank=0,
                model_version=f"{item.backend_name}:{item.acquisition_class}", data_fingerprint=fingerprint,
                source_recipe_id=str(source_recipe_id), provenance={**provenance, "hard_control_constraints": [constraint.__dict__ for constraint in objective.constraints], "feasibility_model_fingerprint": getattr(self.manufacturability_model, "fingerprint", None)},
                control_action_id=str(control_action_id) if control_action_id is not None else None,
                candidate_instance_id=str(item.candidate_id),
            ))
        return result
