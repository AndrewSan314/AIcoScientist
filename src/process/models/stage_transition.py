from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from src.process.stages import ProcessStage


class StageAwareProcessModel(nn.Module):
    """Compact learned state transition, intentionally suitable for small datasets."""

    def __init__(self, state_dim: int, control_dim: int, observation_dim: int) -> None:
        super().__init__()
        if min(state_dim, control_dim, observation_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        self.control_dim = control_dim
        self.observation_dim = observation_dim
        self.stage_embedding = nn.Embedding(len(ProcessStage), state_dim)
        self.transition = nn.Sequential(nn.Linear(state_dim + control_dim + observation_dim + state_dim, state_dim), nn.Tanh())
        self.final_heads = nn.ModuleDict()
        self.intermediate_heads = nn.ModuleDict()

    def add_final_head(self, target: str) -> None:
        self.final_heads[target] = nn.Linear(self.stage_embedding.embedding_dim, 1)

    def add_intermediate_head(self, stage: ProcessStage, target: str) -> None:
        self.intermediate_heads[f"{stage.value}::{target}"] = nn.Linear(self.stage_embedding.embedding_dim, 1)

    def transition_stage(self, state: torch.Tensor, stage: ProcessStage, controls: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        single = state.ndim == 1
        state_batch = _batch(state, self.stage_embedding.embedding_dim, "state")
        controls_batch = _batch(controls, self.control_dim, "controls")
        observations_batch = _batch(observations, self.observation_dim, "observations")
        if state_batch.shape[0] != controls_batch.shape[0] or state_batch.shape[0] != observations_batch.shape[0]:
            raise ValueError("state, controls, and observations must share a batch dimension")
        indices = torch.full((state_batch.shape[0],), list(ProcessStage).index(stage), dtype=torch.long, device=state_batch.device)
        embedding = self.stage_embedding(indices)
        next_state = self.transition(torch.cat([state_batch, controls_batch, observations_batch, embedding], dim=-1))
        return next_state[0] if single else next_state

    def forward(self, initial_state: torch.Tensor, transitions: Iterable[tuple[ProcessStage, torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        single = initial_state.ndim == 1
        state = initial_state
        states: dict[ProcessStage, torch.Tensor] = {}
        for stage, controls, observations in transitions:
            state = self.transition_stage(state, stage, controls, observations)
            states[stage] = state
        outputs = {target: _prediction(head(state), single) for target, head in self.final_heads.items()}
        for key, head in self.intermediate_heads.items():
            stage_name, target = key.split("::", 1)
            stage = ProcessStage(stage_name)
            if stage in states:
                outputs[f"{stage.value}.{target}"] = _prediction(head(states[stage]), single)
        return outputs


def _batch(value: torch.Tensor, expected_dim: int, name: str) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[1] != expected_dim:
        raise ValueError(f"{name} must have shape [features] or [batch, {expected_dim}]")
    return value


def _prediction(value: torch.Tensor, single: bool) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    value = value.squeeze(-1)
    return value[0] if single else value
