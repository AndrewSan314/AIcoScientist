from __future__ import annotations

import pytest
import torch

from src.process.fusion.gated_fusion import GatedMaskedFusion


def test_gated_fusion_uses_only_available_tokens() -> None:
    fusion = GatedMaskedFusion(2)
    assert fusion({"tabular": torch.ones(2)}, {"tabular": True, "sem": False}).shape == (2,)
    with pytest.raises(ValueError):
        fusion({"tabular": torch.ones(2), "sem": torch.zeros(2)}, {"tabular": True, "sem": False})
