from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from src.process.fusion import GatedMaskedFusion
from src.process.stages import ProcessStage

from .stage_transition import StageAwareProcessModel
from .transitions import LegalStageTransition


class MASPOProcessStateModel(nn.Module):
    """Small trainable process-state model composed from the existing fusion/transition blocks."""

    def __init__(self, state_dim: int, control_dim: int, observation_dim: int, modality_input_dims: Mapping[str, int], embedding_dim: int = 16) -> None:
        super().__init__()
        if not modality_input_dims or min(modality_input_dims.values()) <= 0 or embedding_dim <= 0:
            raise ValueError("at least one positive-dimension modality and embedding_dim are required")
        self.embedding_dim = embedding_dim
        self.modality_encoders = nn.ModuleDict({name: nn.Sequential(nn.Linear(size, embedding_dim), nn.Tanh()) for name, size in modality_input_dims.items()})
        self.fusion = GatedMaskedFusion(embedding_dim)
        self.empty_modality_state = nn.Parameter(torch.empty(embedding_dim))
        nn.init.normal_(self.empty_modality_state, mean=0.0, std=0.02)
        self.stage_model = StageAwareProcessModel(state_dim, control_dim, observation_dim + embedding_dim)

    def add_final_head(self, target: str) -> None:
        self.stage_model.add_final_head(target)

    def add_intermediate_head(self, stage: ProcessStage, target: str) -> None:
        self.stage_model.add_intermediate_head(stage, target)

    def fuse_observations(self, tokens: Mapping[str, torch.Tensor], available: Mapping[str, bool | torch.Tensor]) -> torch.Tensor:
        unknown = (set(tokens) | set(available)) - set(self.modality_encoders)
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")
        if not tokens and (not available or not any(bool(torch.as_tensor(mask).any()) for mask in available.values())):
            return self.empty_modality_state
        encoded = {name: self.modality_encoders[name](value.float()) for name, value in tokens.items()}
        return self.fusion(encoded, available)

    def state_from_transitions(
        self,
        initial_state: torch.Tensor,
        transitions: Iterable[LegalStageTransition],
    ) -> torch.Tensor:
        """Return the stage-aware latent state after the supplied legal transitions."""
        state, _ = self._states_from_transitions(initial_state, transitions)
        return state

    def _states_from_transitions(
        self,
        initial_state: torch.Tensor,
        transitions: Iterable[LegalStageTransition],
    ) -> tuple[torch.Tensor, dict[ProcessStage, torch.Tensor]]:
        state = initial_state
        states: dict[ProcessStage, torch.Tensor] = {}
        for transition in transitions:
            if not isinstance(transition, LegalStageTransition):
                raise TypeError("state transitions must be source-bound LegalStageTransition values")
            fused = self.fuse_observations(transition.modality_inputs, transition.availability)
            if fused.ndim == 1 and transition.controls.ndim == 2:
                fused = fused.unsqueeze(0)
            state = self.stage_model.transition_stage(
                state, transition.stage, transition.controls,
                torch.cat([transition.scalar_observations, fused], dim=-1),
            )
            states[transition.stage] = state
        return state, states

    def forward(
        self,
        initial_state: torch.Tensor,
        transitions: Iterable[LegalStageTransition],
    ) -> dict[str, torch.Tensor]:
        state, states = self._states_from_transitions(initial_state, transitions)
        single = initial_state.ndim == 1
        outputs = {
            target: _prediction(head(state), single)
            for target, head in self.stage_model.final_heads.items()
        }
        for key, head in self.stage_model.intermediate_heads.items():
            stage_name, target = key.split("::", 1)
            stage = ProcessStage(stage_name)
            if stage in states:
                outputs[f"{stage.value}.{target}"] = _prediction(head(states[stage]), single)
        return outputs


def _prediction(value: torch.Tensor, single: bool) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    value = value.squeeze(-1)
    return value[0] if single else value
