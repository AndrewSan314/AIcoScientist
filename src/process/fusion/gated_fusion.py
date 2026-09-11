from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class GatedMaskedFusion(nn.Module):
    """Learned attention pooling over only actually available modality tokens."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.gate = nn.Linear(embedding_dim, 1)

    def forward(self, tokens: Mapping[str, torch.Tensor], available: Mapping[str, bool]) -> torch.Tensor:
        declared = set(available)
        if (
            set(tokens) - declared
            or any(available[name] and name not in tokens for name in declared)
            or any(not available[name] and name in tokens for name in declared)
        ):
            raise ValueError("available modalities require tokens; unavailable modalities must not be zero-filled tokens")
        active = [tokens[name] for name in available if available[name]]
        if not active:
            raise ValueError("at least one observed modality is required for fusion")
        matrix = torch.stack(active)
        if matrix.ndim != 2 or matrix.shape[1] != self.gate.in_features:
            raise ValueError("every modality token must be a 1D common-dimension embedding")
        weights = torch.softmax(self.gate(matrix).squeeze(-1), dim=0)
        return (weights.unsqueeze(-1) * matrix).sum(dim=0)
