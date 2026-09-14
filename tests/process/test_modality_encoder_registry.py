import pytest

from src.process.fusion import BaselineModalityEncoderRegistry
from src.process.modalities import ModalityObservation, ModalityType
from src.process.stages import ProcessStage


@pytest.mark.parametrize("kind,values,dimension", [
    (ModalityType.PROCESS_TABULAR, {"speed": 1.0, "gap": 2.0}, 2),
    (ModalityType.ULTRASOUND_SIGNAL, [0.0, 1.0, 0.5], 6),
    (ModalityType.SEM_IMAGE, [[0.0, 1.0], [2.0, 3.0]], 4),
    (ModalityType.CYCLING_CURVE, [1.0, 0.9, 0.8], 6),
])
def test_registry_covers_baseline_modality_families(kind, values, dimension):
    encoded = BaselineModalityEncoderRegistry().encode(ModalityObservation("source", kind, ProcessStage.MIXING, values=values))
    assert encoded.available and encoded.output_dim == dimension and encoded.preprocessing_fingerprint


def test_registry_preserves_missingness_without_zero_fill():
    encoded = BaselineModalityEncoderRegistry().encode(ModalityObservation("missing", ModalityType.ULTRASOUND_SIGNAL, ProcessStage.MIXING, missing_reason="not collected"))
    assert not encoded.available and encoded.values is None and encoded.output_dim == 0
