from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.science.multimodal.extractors import DeterministicExtractor, ExtractionError
from src.science.multimodal.ontology import observable_names_for_modality
from src.science.multimodal.provenance import build_observable_provenance
from src.science.multimodal.schemas import ScientificObservable


def _raw_hash(raw: Any) -> str | None:
    if isinstance(raw, (bytes, bytearray)):
        return hashlib.sha256(bytes(raw)).hexdigest()
    if isinstance(raw, (str, Path)) and Path(raw).exists():
        return hashlib.sha256(Path(raw).read_bytes()).hexdigest()
    if isinstance(raw, np.ndarray):
        return hashlib.sha256(np.asarray(raw, dtype=np.float64).tobytes()).hexdigest()
    return None


class DeterministicXRDSpectralDescriptorExtractor(DeterministicExtractor):
    """Dependency-light descriptors; names deliberately do not claim phase physics."""

    name = "aicoscientist_xrd_descriptor_fallback"
    version = "1.0.0"

    def _pattern(self, pattern: Any, metadata: Mapping[str, Any]) -> np.ndarray:
        if isinstance(pattern, Mapping):
            pattern = pattern.get("normalized_intensity", pattern.get("intensity"))
        if isinstance(pattern, (str, Path)):
            raw = Path(pattern).read_bytes()
            pattern = raw
        if isinstance(pattern, (bytes, bytearray)):
            try:
                from src.domains.alab.xrd_io import parse_alab_xrd

                return np.asarray(parse_alab_xrd(bytes(pattern), scan_metadata=metadata).normalized_intensity, dtype=np.float64)
            except Exception as exc:
                raise ExtractionError(f"unable to parse XRD artifact: {exc}") from exc
        values = np.asarray(pattern, dtype=np.float64).reshape(-1)
        if values.size < 3 or not np.all(np.isfinite(values)):
            raise ExtractionError("XRD pattern must contain at least three finite values")
        values = values - float(np.min(values))
        peak = float(np.max(values))
        if peak <= 0:
            raise ExtractionError("XRD pattern has no positive intensity")
        return values / peak

    def extract(
        self,
        pattern: Any,
        candidate_id: str | Mapping[str, Any] = "unknown",
        metadata: Mapping[str, Any] | None = None,
    ) -> Sequence[ScientificObservable]:
        if isinstance(candidate_id, Mapping):
            metadata = dict(candidate_id)
            candidate_id = str(metadata.get("candidate_id", "unknown"))
        meta = dict(metadata or {})
        values = self._pattern(pattern, meta)
        peak_index = int(np.argmax(values))
        half_max = 0.5 * float(np.max(values))
        above = np.flatnonzero(values >= half_max)
        halfmax_span = float(above[-1] - above[0]) / max(len(values) - 1, 1) if len(above) else 1.0
        mass = values / max(float(np.sum(values)), 1e-12)
        spectral_entropy = float(-np.sum(mass * np.log(np.maximum(mass, 1e-12))) / np.log(len(values)))
        threshold = 0.1 * float(np.max(values))
        local_maxima = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]) & (values[1:-1] >= threshold)
        peak_count = float(np.sum(local_maxima) + (1 if values.size >= 1 and values[0] >= threshold else 0))
        provenance = build_observable_provenance(
            meta.get("raw_artifact_ref"), self.name, self.version,
            raw_artifact_hash=_raw_hash(pattern),
            configuration={"normalization": "min_shift_then_peak_scale"},
        )
        common = {
            "candidate_id": str(candidate_id), "modality": "XRD",
            "raw_artifact_ref": meta.get("raw_artifact_ref"),
            "extractor_name": self.name, "extractor_version": self.version,
            "provenance": {**provenance, **meta}, "timestamp": meta.get("timestamp"),
        }
        names = observable_names_for_modality("XRD")
        values_by_name = {
            names[0]: float(np.std(values)),
            names[1]: peak_index / max(len(values) - 1, 1),
            names[2]: halfmax_span,
            names[3]: spectral_entropy,
            names[4]: peak_count,
        }
        uncertainty = {name: 0.05 for name in names}
        uncertainty[names[1]] = 1.0 / max(len(values), 1)
        uncertainty[names[4]] = 1.0
        return [
            ScientificObservable(
                observable_id=f"XRD:{candidate_id}:{name}",
                name=name,
                value=value,
                uncertainty=uncertainty[name],
                observable_type="scalar",
                **common,
            )
            for name, value in values_by_name.items()
        ]


class AutoXRDPhaseExtractor(DeterministicExtractor):
    """Real AutoXRD boundary; it never silently falls back to descriptors."""

    name = "autoxrd_phase_model"
    version = "1.0.0"

    def __init__(self, checkpoint: str | Path | None = None, references_dir: str | Path | None = None) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.references_dir = Path(references_dir) if references_dir else None

    @property
    def integration_status(self) -> str:
        try:
            import autoXRD  # type: ignore
        except ImportError:
            return "NOT_AVAILABLE"
        if autoXRD is None or self.checkpoint is None or not self.checkpoint.is_file() or self.references_dir is None or not self.references_dir.is_dir():
            return "NOT_AVAILABLE"
        return "REAL_INTEGRATION"

    @property
    def auto_xrd_available(self) -> bool:
        return self.integration_status == "REAL_INTEGRATION"

    def extract(self, pattern: Any, candidate_id: str = "unknown", metadata: Mapping[str, Any] | None = None) -> Sequence[ScientificObservable]:
        if not self.auto_xrd_available:
            raise ExtractionError(
                "AutoXRD inference is NOT_AVAILABLE: configure the pinned upstream model checkpoint and reference directory"
            )
        # The pinned upstream API requires a spectrum directory, reference CIFs, and a trained Keras checkpoint.
        # Keep this boundary explicit until all three are configured; no descriptor is mislabeled as model output.
        raise ExtractionError("AutoXRD inference configuration is present but this adapter requires a file-backed spectrum path")


# Compatibility names retain import compatibility without using the upstream brand for the fallback.
XRDObservableExtractor = DeterministicXRDSpectralDescriptorExtractor
AutoXRDObservableExtractor = AutoXRDPhaseExtractor


__all__ = [
    "AutoXRDObservableExtractor",
    "AutoXRDPhaseExtractor",
    "DeterministicXRDSpectralDescriptorExtractor",
    "XRDObservableExtractor",
]
