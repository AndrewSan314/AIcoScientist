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

    def forward(self, tokens: Mapping[str, torch.Tensor], available: Mapping[str, bool | torch.Tensor]) -> torch.Tensor:
        declared = set(available)
        if set(tokens) - declared:
            raise ValueError("tokens must be declared in the availability mask")
        matrices: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        single = True
        batch_size: int | None = None
        device = next(iter(tokens.values())).device if tokens else self.gate.weight.device
        for name, declared_mask in available.items():
            mask = torch.as_tensor(declared_mask, device=device)
            if mask.ndim == 0:
                mask = mask.reshape(1)
            elif mask.ndim != 1:
                raise ValueError("modality availability masks must be scalar or 1D batch masks")
            valid_mask = torch.ones_like(mask, dtype=torch.bool) if mask.dtype == torch.bool else (mask == 0) | (mask == 1)
            if not valid_mask.all():
                raise ValueError("modality availability masks must contain only 0/1 values")
            token = tokens.get(name)
            if token is None:
                if not mask.bool().any():
                    continue
                raise ValueError(f"available modality {name!r} requires a token")
            if not mask.bool().any():
                raise ValueError(f"unavailable modality {name!r} must not provide a token")
            token = token.to(device=device)
            if token.ndim == 1:
                token = token.unsqueeze(0)
            elif token.ndim != 2:
                raise ValueError("every modality token must be a 1D or batched 2D embedding")
            else:
                single = False
            if token.shape[1] != self.gate.in_features:
                raise ValueError("every modality token must use the common embedding dimension")
            current_batch = token.shape[0]
            if batch_size is None:
                batch_size = current_batch
            if current_batch != batch_size:
                raise ValueError("modality tokens must share a batch dimension")
            if mask.numel() == 1:
                mask = mask.expand(batch_size)
            elif mask.numel() != batch_size:
                raise ValueError("modality availability masks must match token batch size")
            matrices.append(token)
            masks.append(mask.bool())
        if not matrices:
            raise ValueError("at least one observed modality is required for fusion")
        matrix = torch.stack(matrices, dim=1).to(dtype=self.gate.weight.dtype)
        mask_matrix = torch.stack(masks, dim=1)
        if not mask_matrix.any(dim=1).all():
            raise ValueError("each batch item needs at least one observed modality")
        logits = self.gate(matrix).squeeze(-1).masked_fill(~mask_matrix, torch.finfo(matrix.dtype).min)
        weights = torch.softmax(logits, dim=1)
        fused = (weights.unsqueeze(-1) * matrix).sum(dim=1)
        return fused[0] if single and fused.shape[0] == 1 else fused
