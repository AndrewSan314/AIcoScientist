from __future__ import annotations

import pytest

from src.process.modalities import ModalityObservation, ModalityType
from src.process.stages import ProcessStage


def test_missing_modality_is_explicit_and_never_synthetic() -> None:
    missing = ModalityObservation("sem", ModalityType.SEM_IMAGE, ProcessStage.CALENDERING, missing_reason="not_collected")
    assert missing.is_missing
    with pytest.raises(ValueError):
        ModalityObservation("bad", ModalityType.SEM_IMAGE, ProcessStage.CALENDERING, values=None)
