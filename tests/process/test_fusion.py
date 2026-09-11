from __future__ import annotations

import pytest
import torch

from src.process.fusion.gated_fusion import GatedMaskedFusion


def test_gated_fusion_uses_only_available_tokens() -> None:
    fusion = GatedMaskedFusion(2)
    assert fusion({"tabular": torch.ones(2)}, {"tabular": True, "sem": False}).shape == (2,)
    with pytest.raises(ValueError):
        fusion({"tabular": torch.ones(2), "sem": torch.zeros(2)}, {"tabular": True, "sem": False})


def test_gated_fusion_supports_per_item_batch_masks() -> None:
    fusion = GatedMaskedFusion(2)
    output = fusion(
        {"tabular": torch.ones(2, 2), "sem": torch.full((2, 2), 3.0)},
        {"tabular": torch.tensor([1, 1]), "sem": torch.tensor([0, 1])},
    )
    assert output.shape == (2, 2)
    assert torch.allclose(output[0], torch.ones(2))
