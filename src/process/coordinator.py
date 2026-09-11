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
from .optimization.botorch import OfficialMultiObjectiveBoTorch
from .optimization.process_objective import ProcessOptimizationObjective
from .optimization.process_space import ProcessSearchSpace
from .optimization.proposal import Prediction, ProcessControlProposal
from .stages import ProcessStage
from .validation import validate_process_run


@dataclass
class ProcessOptimizationCoordinator:
    """Composition root for the default battery process path, without HIG imports."""

    scalar_backend: Any | None = None
    multiobjective_backend: OfficialMultiObjectiveBoTorch | None = None

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
        """Encode only audited finite-pool controls for the official backend."""
        return space.official_botorch_view(observations)

    def propose_recipes(
        self,
        observations: pd.DataFrame,
        space: ProcessSearchSpace,
        objective: ProcessOptimizationObjective,
        *,
        n: int = 1,
        seed: int | None = None,
    ) -> list[ProcessControlProposal]:
        if objective.is_multiobjective:
            backend = self.multiobjective_backend or OfficialMultiObjectiveBoTorch()
            return backend.propose(observations, space, objective, n=n, seed=seed)
        target = objective.objectives[0]
        if self.scalar_backend is None:
            try:
                from src.optimization.botorch_backend import BoTorchBackend
            except ImportError as exc:
                raise RuntimeError("scalar process optimization requires the official botorch dependency") from exc
            backend = BoTorchBackend()
            backend_observations, backend_space = self._scale_official_botorch_inputs(observations, space)
        else:
            backend = self.scalar_backend
            backend_observations, backend_space = observations, space
        proposals = backend.propose(
            observations=backend_observations,
            candidate_pool=backend_space.candidates,
            objective=OfficialOptimizationObjective(target_name=target.target, minimize=target.sense == "minimize"),
            feature_columns=backend_space.control_columns,
            candidate_id_column=backend_space.id_column,
            n=n,
            seed=seed,
            strategy="noisy_expected_improvement",
        )
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(observations, index=True).values.tobytes()).hexdigest()
        source_controls = space.candidates.set_index(space.id_column)[space.control_columns]
        return [
            ProcessControlProposal(
                proposal_id=f"process:{item.candidate_id}", stage=None, controls=source_controls.loc[item.candidate_id].to_dict(),
                predicted_outputs={target.target: Prediction(item.predicted_mean, item.predicted_std, target.units)},
                feasibility_probability=None, acquisition_value=item.acquisition_value, pareto_rank=0,
                model_version=f"{item.backend_name}:{item.acquisition_class}", data_fingerprint=fingerprint,
                source_recipe_id=item.candidate_id, provenance=item.metadata,
            )
            for item in proposals
        ]
