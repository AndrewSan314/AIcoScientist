from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from src.process.fusion import GatedMaskedFusion
from src.process.stages import ProcessStage

from .stage_transition import StageAwareProcessModel


class MASPOProcessStateModel(nn.Module):
    """Small trainable process-state model composed from the existing fusion/transition blocks."""

    def __init__(self, state_dim: int, control_dim: int, observation_dim: int, modality_input_dims: Mapping[str, int], embedding_dim: int = 16) -> None:
        super().__init__()
        if not modality_input_dims or min(modality_input_dims.values()) <= 0 or embedding_dim <= 0:
            raise ValueError("at least one positive-dimension modality and embedding_dim are required")
        self.modality_encoders = nn.ModuleDict({name: nn.Sequential(nn.Linear(size, embedding_dim), nn.Tanh()) for name, size in modality_input_dims.items()})
        self.fusion = GatedMaskedFusion(embedding_dim)
        self.stage_model = StageAwareProcessModel(state_dim, control_dim, observation_dim + embedding_dim)

    def add_final_head(self, target: str) -> None:
        self.stage_model.add_final_head(target)

    def add_intermediate_head(self, stage: ProcessStage, target: str) -> None:
        self.stage_model.add_intermediate_head(stage, target)

    def fuse_observations(self, tokens: Mapping[str, torch.Tensor], available: Mapping[str, bool | torch.Tensor]) -> torch.Tensor:
        unknown = (set(tokens) | set(available)) - set(self.modality_encoders)
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")
        encoded = {name: self.modality_encoders[name](value.float()) for name, value in tokens.items()}
        return self.fusion(encoded, available)

    def forward(
        self,
        initial_state: torch.Tensor,
        transitions: Iterable[tuple[ProcessStage, torch.Tensor, torch.Tensor, Mapping[str, torch.Tensor], Mapping[str, bool | torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        encoded_transitions = []
        for stage, controls, observations, tokens, available in transitions:
            fused = self.fuse_observations(tokens, available)
            if fused.ndim == 1 and controls.ndim == 2:
                fused = fused.unsqueeze(0)
            encoded_transitions.append((stage, controls, torch.cat([observations, fused], dim=-1)))
        return self.stage_model(initial_state, encoded_transitions)
