from __future__ import annotations

import torch

from src.process.models.stage_transition import StageAwareProcessModel
from src.process.stages import ProcessStage


def test_stage_model_predicts_registered_final_head() -> None:
    model = StageAwareProcessModel(3, 2, 2)
    model.add_final_head("capacity")
    output = model(torch.zeros(3), [(ProcessStage.MIXING, torch.ones(2), torch.ones(2))])
    assert set(output) == {"capacity"}
