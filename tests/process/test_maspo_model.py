from __future__ import annotations

import torch

from src.process.contracts import MeasurementValue, ParameterValue, StageRecord
from src.process.modalities import ModalityObservation, ModalityType
from src.process.models import MASPOProcessStateModel
from src.process.stages import ProcessStage
from src.process.models.transitions import LegalStageTransition


def test_maspo_model_encodes_masks_and_transitions_a_batch() -> None:
    model = MASPOProcessStateModel(3, 2, 2, {"tabular": 2, "signal": 3}, embedding_dim=4)
    model.add_final_head("capacity")
    model.add_intermediate_head(ProcessStage.MIXING, "viscosity")
    record = StageRecord(
        "mix", ProcessStage.MIXING, 1,
        {"speed": ParameterValue(100)},
        {"a": MeasurementValue(1), "b": MeasurementValue(2)},
        [
            ModalityObservation("tab", ModalityType.PROCESS_TABULAR, ProcessStage.MIXING, values=[1, 2]),
            ModalityObservation("signal", ModalityType.MACHINE_TIME_SERIES, ProcessStage.MIXING, values=[1, 2, 3]),
        ],
    )
    transition = LegalStageTransition.from_source_stage(
        record, controls=torch.ones(2, 2), scalar_observations=torch.ones(2, 2),
        modality_inputs={"tab": torch.ones(2, 2), "signal": torch.ones(2, 3)},
        modality_bindings={"tab": "tabular", "signal": "signal"},
    )
    output = model(
        torch.zeros(2, 3),
        [transition],
    )
    assert output["capacity"].shape == (2,)
    assert output["MIXING.viscosity"].shape == (2,)
