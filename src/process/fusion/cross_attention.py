from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


class CrossAttentionSetFusion(nn.Module):
    """Experimental learned-query set fusion over declared, observed modality tokens."""

    def __init__(self, embedding_dim: int, modality_names: Sequence[str], *, num_heads: int = 1) -> None:
        super().__init__()
        names = tuple(modality_names)
        if embedding_dim <= 0 or not names or len(set(names)) != len(names) or embedding_dim % num_heads:
            raise ValueError("positive embedding_dim, unique modality names, and divisible num_heads are required")
        self.modality_names = names
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.modality_type = nn.Parameter(torch.empty(len(names), embedding_dim))
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        nn.init.normal_(self.modality_type, mean=0.0, std=0.02)

    def forward(self, tokens: Mapping[str, torch.Tensor], available: Mapping[str, bool | torch.Tensor]) -> torch.Tensor:
        if set(available) != set(self.modality_names) or set(tokens) - set(available):
            raise ValueError("tokens and availability must use exactly the declared modality names")
        matrices: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        batch_size: int | None = None
        single = True
        device = self.query.device
        for name in self.modality_names:
            mask = torch.as_tensor(available[name], device=device)
            if mask.ndim == 0:
                mask = mask.reshape(1)
            elif mask.ndim != 1 or not ((mask == 0) | (mask == 1)).all():
                raise ValueError("modality availability masks must be scalar or binary 1D batch masks")
            token = tokens.get(name)
            if token is None:
                if mask.bool().any():
                    raise ValueError(f"available modality {name!r} requires a token")
                continue
            if not mask.bool().any():
                raise ValueError(f"unavailable modality {name!r} must not provide a token")
            token = token.to(device=device, dtype=self.query.dtype)
            if token.ndim == 1:
                token = token.unsqueeze(0)
            elif token.ndim != 2:
                raise ValueError("every modality token must be a 1D or batched 2D embedding")
            else:
                single = False
            if token.shape[1] != self.query.shape[-1]:
                raise ValueError("every modality token must use the common embedding dimension")
            batch_size = batch_size or token.shape[0]
            if token.shape[0] != batch_size or (mask.numel() not in (1, batch_size)):
                raise ValueError("modality tokens and masks must share a batch dimension")
            matrices.append(token)
            masks.append(mask.bool().expand(batch_size))
        if not matrices:
            raise ValueError("at least one observed modality is required for fusion")
        matrix = torch.stack(matrices, dim=1)
        mask_matrix = torch.stack(masks, dim=1)
        if not mask_matrix.any(dim=1).all():
            raise ValueError("each batch item needs at least one observed modality")
        type_indices = torch.tensor([self.modality_names.index(name) for name in self.modality_names if name in tokens], device=device)
        matrix = matrix + self.modality_type[type_indices].unsqueeze(0)
        fused, _ = self.attention(self.query.expand(batch_size, -1, -1), matrix, matrix, key_padding_mask=~mask_matrix, need_weights=False)
        return fused[0, 0] if single and batch_size == 1 else fused[:, 0]
