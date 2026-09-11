from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

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
        stages = tuple(sorted((stage for stage in remaining_control_spaces if STAGE_ORDER[stage] > STAGE_ORDER[current_stage]), key=STAGE_ORDER.__getitem__))
        if not stages:
            raise ValueError("no remaining process control stage exists after current_stage")
        next_stage = stages[0]
        stage_observations = observations[next_stage] if isinstance(observations, dict) else observations
        proposals = self.optimizer.propose_recipes(stage_observations, remaining_control_spaces[next_stage], objective, n=1, seed=seed)
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
