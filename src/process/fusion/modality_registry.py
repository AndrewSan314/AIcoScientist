"""Baseline-first modality encoder registry with explicit source missingness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.process.modalities import ModalityObservation, ModalityType

from .encoders import ImageFeatureEncoder, SignalFeatureEncoder, TabularEncoder


_TABULAR = {ModalityType.PROCESS_TABULAR, ModalityType.SCALAR_METROLOGY, ModalityType.ENVIRONMENT_METADATA}
_SIGNAL = {ModalityType.MACHINE_TIME_SERIES, ModalityType.THERMAL_TIME_SERIES, ModalityType.ULTRASOUND_SIGNAL, ModalityType.ULTRASOUND_SPECTRUM, ModalityType.EIS_SPECTRUM, ModalityType.XRD_SPECTRUM, ModalityType.FORMATION_CURVE, ModalityType.CYCLING_CURVE}
_IMAGE = {ModalityType.OPTICAL_IMAGE, ModalityType.SEM_IMAGE, ModalityType.EDS_MAP, ModalityType.XCT_VOLUME}


@dataclass(frozen=True)
class EncodedModality:
    modality_id: str
    modality_type: ModalityType
    available: bool
    values: np.ndarray | None
    feature_names: tuple[str, ...]
    preprocessing_fingerprint: str

    @property
    def output_dim(self) -> int:
        return 0 if self.values is None else int(self.values.size)


class BaselineModalityEncoderRegistry:
    """One auditable descriptor encoder per declared modality family."""

    VERSION = "baseline-modality-registry-v1"

    def __init__(self) -> None:
        self.tabular, self.signal, self.image = TabularEncoder(), SignalFeatureEncoder(), ImageFeatureEncoder()

    def encode(self, observation: ModalityObservation) -> EncodedModality:
        if observation.is_missing:
            return EncodedModality(observation.modality_id, observation.modality_type, False, None, (), self._fingerprint(observation.modality_type, ()))
        values, names = self._encode_present(observation)
        if not np.isfinite(values).all():
            raise ValueError("baseline modality encoder produced non-finite values")
        return EncodedModality(observation.modality_id, observation.modality_type, True, values, names, self._fingerprint(observation.modality_type, names))

    def _encode_present(self, observation: ModalityObservation) -> tuple[np.ndarray, tuple[str, ...]]:
        kind, value = observation.modality_type, observation.values
        if kind in _TABULAR:
            mapping = value if isinstance(value, dict) else {"value": value}
            columns = tuple(sorted(mapping))
            return self.tabular.encode(mapping, columns=columns), columns
        if kind in _SIGNAL:
            encoded = self.signal.encode(np.asarray(value, dtype=float).reshape(-1))
            return encoded, ("mean", "std", "min", "max", "rms", "dominant_fft_amplitude")
        if kind in _IMAGE:
            encoded = self.image.encode(np.asarray(value, dtype=float))
            return encoded, ("mean", "std", "min", "max")
        raise ValueError(f"unsupported modality type: {kind.value}")

    def _fingerprint(self, kind: ModalityType, names: tuple[str, ...]) -> str:
        payload: dict[str, Any] = {"version": self.VERSION, "modality_type": kind.value, "feature_names": names}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
