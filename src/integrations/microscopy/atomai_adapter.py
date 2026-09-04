from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.science.multimodal.extractors import DeterministicExtractor, ExtractionError
from src.science.multimodal.provenance import build_observable_provenance
from src.science.multimodal.schemas import ScientificObservable


def _as_image(raw: Any) -> np.ndarray:
    if isinstance(raw, (str, Path)):
        raw = Path(raw).read_bytes()
    if isinstance(raw, (bytes, bytearray)):
        try:
            from PIL import Image

            with Image.open(io.BytesIO(bytes(raw))) as image:
                raw = np.asarray(image.convert("L"))
        except Exception as exc:
            raise ExtractionError(f"SEM image decoding failed: {exc}") from exc
    image = np.asarray(raw, dtype=np.float64)
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    if image.ndim != 2 or image.size < 9 or not np.all(np.isfinite(image)):
        raise ExtractionError("SEM input must be a finite two-dimensional image")
    lo, hi = float(np.min(image)), float(np.max(image))
    return (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)


def _observables(common: dict[str, Any], values: Mapping[str, float], uncertainties: Mapping[str, float]) -> list[ScientificObservable]:
    return [
        ScientificObservable(
            observable_id=f"{common['modality']}:{common['candidate_id']}:{name}",
            name=name, value=float(value), uncertainty=float(uncertainties.get(name, 0.1)),
            observable_type="scalar", **common,
        )
        for name, value in values.items()
    ]


class AtomAISEMExtractor(DeterministicExtractor):
    """AtomAI-compatible boundary; classical CPU descriptors are the tested fallback."""

    name = "atomai_optional_sem_cpu_fallback"
    version = "1.0.0"

    def extract(
        self,
        raw_measurement: Any,
        candidate_id: str | Mapping[str, Any] = "unknown",
        metadata: Mapping[str, Any] | None = None,
    ) -> Sequence[ScientificObservable]:
        if isinstance(candidate_id, Mapping):
            metadata = dict(candidate_id)
            candidate_id = str(metadata.get("candidate_id", "unknown"))
        meta = dict(metadata or {})
        image = _as_image(raw_measurement)
        gradient = np.mean(np.abs(np.diff(image, axis=0))) + np.mean(np.abs(np.diff(image, axis=1)))
        threshold = float(np.median(image))
        foreground = image > threshold
        texture = float(np.std(image))
        porosity = float(np.mean(~foreground))
        agglomeration = float(np.clip(np.mean(foreground[:, 1:] & foreground[:, :-1]), 0.0, 1.0))
        provenance = build_observable_provenance(
            meta.get("raw_artifact_ref"), self.name, self.version,
            raw_artifact_hash=hashlib.sha256(np.asarray(image, dtype=np.float64).tobytes()).hexdigest(),
            configuration={"backend": "classical_cpu", "threshold": "median"},
        )
        common = {
            "candidate_id": str(candidate_id), "modality": "SEM",
            "raw_artifact_ref": meta.get("raw_artifact_ref"),
            "extractor_name": self.name, "extractor_version": self.version,
            "provenance": {**provenance, **meta}, "timestamp": meta.get("timestamp"),
        }
        values = {"grain_size_mean": float(1.0 / max(gradient, 1e-6)), "grain_size_std": texture, "porosity": porosity, "agglomeration": agglomeration}
        return _observables(common, values, {"grain_size_mean": 0.2, "grain_size_std": 0.1, "porosity": 0.05, "agglomeration": 0.05})


class AtomAIEDSExtractor(DeterministicExtractor):
    """AtomAI-compatible EDS boundary with a dependency-light numeric table fallback."""

    name = "atomai_optional_eds_cpu_fallback"
    version = "1.0.0"

    def _values(self, raw: Any) -> np.ndarray:
        if isinstance(raw, (str, Path)):
            raw = Path(raw).read_bytes()
        if isinstance(raw, (bytes, bytearray)):
            try:
                import pandas as pd

                frame = pd.read_csv(io.BytesIO(bytes(raw)))
                values = frame.select_dtypes(include=["number"]).to_numpy(dtype=np.float64)
            except Exception as exc:
                raise ExtractionError(f"EDS table parsing failed: {exc}") from exc
        elif isinstance(raw, Mapping):
            values = np.asarray(raw.get("values", raw.get("composition", [])), dtype=np.float64)
        else:
            values = np.asarray(raw, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ExtractionError("EDS input must contain a finite numeric table")
        return values

    def extract(
        self,
        raw_measurement: Any,
        candidate_id: str | Mapping[str, Any] = "unknown",
        metadata: Mapping[str, Any] | None = None,
    ) -> Sequence[ScientificObservable]:
        if isinstance(candidate_id, Mapping):
            metadata = dict(candidate_id)
            candidate_id = str(metadata.get("candidate_id", "unknown"))
        meta = dict(metadata or {})
        values = self._values(raw_measurement)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        composition_error = float(abs(np.sum(mean) - 1.0)) if np.sum(mean) <= 1.5 else float(np.mean(std))
        spatial_variance = float(np.mean(std**2))
        segregation = float(np.clip(np.mean(std / np.maximum(np.abs(mean), 1e-9)), 0.0, 1.0))
        provenance = build_observable_provenance(
            meta.get("raw_artifact_ref"), self.name, self.version,
            raw_artifact_hash=hashlib.sha256(values.tobytes()).hexdigest(),
            configuration={"backend": "classical_cpu", "rows": int(values.shape[0]), "columns": int(values.shape[1])},
        )
        common = {
            "candidate_id": str(candidate_id), "modality": "EDS",
            "raw_artifact_ref": meta.get("raw_artifact_ref"),
            "extractor_name": self.name, "extractor_version": self.version,
            "provenance": {**provenance, **meta}, "timestamp": meta.get("timestamp"),
        }
        obs = {"composition_error": composition_error, "spatial_variance": spatial_variance, "segregation_index": segregation, "element_colocalization": 1.0 - segregation}
        return _observables(common, obs, {name: 0.05 for name in obs})


__all__ = ["AtomAIEDSExtractor", "AtomAISEMExtractor"]
