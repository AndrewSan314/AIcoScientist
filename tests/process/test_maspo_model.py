from __future__ import annotations

import torch

from src.process.models import MASPOProcessStateModel
from src.process.stages import ProcessStage


def test_maspo_model_encodes_masks_and_transitions_a_batch() -> None:
    model = MASPOProcessStateModel(3, 2, 2, {"tabular": 2, "signal": 3}, embedding_dim=4)
    model.add_final_head("capacity")
    model.add_intermediate_head(ProcessStage.MIXING, "viscosity")
    output = model(
        torch.zeros(2, 3),
        [(
            ProcessStage.MIXING,
            torch.ones(2, 2),
            torch.ones(2, 2),
            {"tabular": torch.ones(2, 2), "signal": torch.ones(2, 3)},
            {"tabular": torch.tensor([1, 1]), "signal": torch.tensor([0, 1])},
        )],
    )
    assert output["capacity"].shape == (2,)
    assert output["MIXING.viscosity"].shape == (2,)
