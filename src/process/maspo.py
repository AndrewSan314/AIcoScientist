from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from numbers import Real
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .contracts import BatteryProcessRun
from .coordinator import ProcessOptimizationCoordinator
from .information_horizon import HorizonView
from .optimization.process_objective import ProcessOptimizationObjective
from .optimization.process_space import ProcessSearchSpace
from .optimization.proposal import ProcessControlProposal
from .stages import ProcessStage, STAGE_ORDER


@dataclass(frozen=True)
class MASPOPlan:
    current_stage: ProcessStage
    next_stage: ProcessStage
    next_control: ProcessControlProposal
    downstream_stages: tuple[ProcessStage, ...]
    legal_state: HorizonView
    decision_latency_seconds: float


@dataclass(frozen=True)
class ContextualProcessState:
    """Numeric process state encoded from one legal InformationHorizon view."""

    feature_names: tuple[str, ...]
    feature_values: dict[str, float]
    source_stage_ids: tuple[str, ...]
    decision_stage: ProcessStage
    provenance_fingerprint: str

    @classmethod
    def from_horizon(cls, horizon: HorizonView) -> "ContextualProcessState":
        values: dict[str, float] = {}
        for prefix, source in (("context.control.", horizon.controls), ("context.intermediate.", horizon.intermediate_properties)):
            for name, item in source.items():
                value = item.value
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                    continue
                values[prefix + name] = float(value)
        if not values:
            raise ValueError("contextual MASPO requires a finite numeric feature in the legal HorizonView; no zero-state fallback")
        payload = {"decision_stage": horizon.decision_stage.value, "source_stage_ids": horizon.source_stage_ids, "features": values}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(tuple(values), values, horizon.source_stage_ids, horizon.decision_stage, fingerprint)


class MASPOProcessOptimizationCoordinator:
    """Stage-wise/receding-horizon wrapper around the existing official optimizer."""

    def __init__(self, optimizer: ProcessOptimizationCoordinator | None = None) -> None:
        self.optimizer = optimizer or ProcessOptimizationCoordinator()

    def optimize_remaining_process(
        self,
        *,
        current_state: BatteryProcessRun | HorizonView,
        current_stage: ProcessStage,
        remaining_control_spaces: dict[ProcessStage, ProcessSearchSpace],
        observations: pd.DataFrame | dict[ProcessStage, pd.DataFrame],
        objective: ProcessOptimizationObjective,
        seed: int | None = None,
    ) -> MASPOPlan:
        started = perf_counter()
        legal_state = (
            self.optimizer.available_state(current_state, current_stage)
            if isinstance(current_state, BatteryProcessRun)
            else current_state
        )
        if legal_state.decision_stage != current_stage:
            raise ValueError("current HorizonView decision_stage must match current_stage")
        stages = tuple(sorted((stage for stage in remaining_control_spaces if STAGE_ORDER[stage] > STAGE_ORDER[current_stage]), key=STAGE_ORDER.__getitem__))
        if not stages:
            raise ValueError("no remaining process control stage exists after current_stage")
        next_stage = stages[0]
        stage_observations = observations[next_stage] if isinstance(observations, dict) else observations
        _, contextual_observations, contextual_space = self._contextual_inputs(
            legal_state, remaining_control_spaces[next_stage], stage_observations, objective,
        )
        proposals = self.optimizer.propose_recipes(contextual_observations, contextual_space, objective, n=1, seed=seed)
        if not proposals:
            raise ValueError("official process optimizer returned no next-stage action")
        action = replace(proposals[0], stage=next_stage)
        return MASPOPlan(
            current_stage=current_stage,
            next_stage=next_stage,
            next_control=action,
            downstream_stages=stages[1:],
            legal_state=legal_state,
            decision_latency_seconds=perf_counter() - started,
        )

    @staticmethod
    def _contextual_inputs(
        legal_state: HorizonView,
        space: ProcessSearchSpace,
        observations: pd.DataFrame,
        objective: ProcessOptimizationObjective,
    ) -> tuple[ContextualProcessState, pd.DataFrame, ProcessSearchSpace]:
        state = ContextualProcessState.from_horizon(legal_state)
        if space.context_columns:
            raise ValueError("MASPO expects an uncontextualized finite control space")
        target_names = tuple(item.target for item in objective.objectives)
        required = [space.id_column, *space.control_columns, *state.feature_names, *target_names]
        missing = [column for column in required if column not in observations]
        if missing:
            raise ValueError(f"contextual MASPO requires source-backed historical context columns: {missing}")
        if observations.empty or observations[space.id_column].isna().any():
            raise ValueError("contextual MASPO requires non-null historical recipe identities")
        known_ids = set(space.candidates[space.id_column].astype(str))
        unknown_ids = sorted(set(observations[space.id_column].astype(str)) - known_ids)
        if unknown_ids:
            raise ValueError(f"historical observations contain unknown source recipes: {unknown_ids}")
        clean = observations.loc[:, required].copy()
        bounds: dict[str, tuple[float, float]] = {}
        for name in state.feature_names:
            values = pd.to_numeric(clean[name], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy()).all():
                raise ValueError(f"historical context column {name!r} is missing or non-finite")
            lower, upper = float(values.min()), float(values.max())
            current = state.feature_values[name]
            if current < lower or current > upper:
                raise ValueError(f"current legal context {name!r} lies outside historical support")
            bounds[name] = (lower, upper)
        for name in target_names:
            values = pd.to_numeric(clean[name], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy()).all():
                raise ValueError(f"historical target column {name!r} is missing or non-finite")
        candidate_pool = space.candidates.copy()
        for name, value in state.feature_values.items():
            if name in candidate_pool:
                raise ValueError(f"context column collides with source control column: {name}")
            candidate_pool[name] = value
        contextual_space = ProcessSearchSpace.from_finite_pool(
            candidate_pool, id_column=space.id_column, context_columns=state.feature_names, context_bounds=bounds,
        )
        return state, clean, contextual_space
