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
        self.stage_embedding = nn.Embedding(len(ProcessStage), state_dim)
        self.transition = nn.Sequential(nn.Linear(state_dim + control_dim + observation_dim + state_dim, state_dim), nn.Tanh())
        self.final_heads = nn.ModuleDict()

    def add_final_head(self, target: str) -> None:
        self.final_heads[target] = nn.Linear(self.stage_embedding.embedding_dim, 1)

    def transition_stage(self, state: torch.Tensor, stage: ProcessStage, controls: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        embedding = self.stage_embedding(torch.tensor(list(ProcessStage).index(stage), device=state.device))
        return self.transition(torch.cat([state, controls, observations, embedding], dim=-1))

    def forward(self, initial_state: torch.Tensor, transitions: Iterable[tuple[ProcessStage, torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        state = initial_state
        for stage, controls, observations in transitions:
            state = self.transition_stage(state, stage, controls, observations)
        return {target: head(state).squeeze(-1) for target, head in self.final_heads.items()}
