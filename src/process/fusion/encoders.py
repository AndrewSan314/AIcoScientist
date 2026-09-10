from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


class TabularEncoder:
    """Small, auditable baseline encoder for scalar process/context values."""

    def encode(self, values: Mapping[str, float], *, columns: Sequence[str] | None = None) -> np.ndarray:
        names = list(columns) if columns is not None else sorted(values)
        if not names:
            raise ValueError("tabular encoder requires at least one feature")
        result = np.asarray([values[name] for name in names], dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("tabular features must be finite; missingness belongs in a modality mask")
        return result


class SignalFeatureEncoder:
    """Physical/statistical descriptors before any large learned signal model."""

    def encode(self, signal: Sequence[float]) -> np.ndarray:
        values = np.asarray(signal, dtype=float).reshape(-1)
        if values.size < 2 or not np.isfinite(values).all():
            raise ValueError("signal must contain at least two finite observations")
        fft = np.abs(np.fft.rfft(values))
        return np.asarray([values.mean(), values.std(), values.min(), values.max(), np.sqrt(np.mean(values ** 2)), fft[1:].max(initial=0.0)], dtype=float)


class ImageFeatureEncoder:
    """Minimal image-statistics fallback; SEM morphology remains an optional adapter."""

    def encode(self, image: np.ndarray) -> np.ndarray:
        pixels = np.asarray(image, dtype=float)
        if pixels.size == 0 or not np.isfinite(pixels).all():
            raise ValueError("image must be non-empty and finite")
        return np.asarray([pixels.mean(), pixels.std(), pixels.min(), pixels.max()], dtype=float)
