from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.science.multimodal.extractors import DeterministicExtractor, ExtractionError
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


class XRDObservableExtractor(DeterministicExtractor):
    """Validated CPU fallback for XRD descriptors; AutoXRD is optional and never assumed truth."""

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
        fwhm = float(above[-1] - above[0]) / max(len(values) - 1, 1) if len(above) else 1.0
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
        return [
            ScientificObservable(observable_id=f"XRD:{candidate_id}:crystallinity", name="crystallinity", value=float(np.std(values)), uncertainty=0.05, observable_type="scalar", **common),
            ScientificObservable(observable_id=f"XRD:{candidate_id}:peak_position_index", name="peak_position_index", value=peak_index / max(len(values) - 1, 1), uncertainty=1.0 / max(len(values), 1), observable_type="scalar", **common),
            ScientificObservable(observable_id=f"XRD:{candidate_id}:FWHM", name="FWHM", value=fwhm, uncertainty=0.05, observable_type="scalar", **common),
        ]


class AutoXRDObservableExtractor(XRDObservableExtractor):
    """Optional AutoXRD boundary with deterministic fallback and explicit backend metadata."""

    name = "autoxrd_optional_with_cpu_fallback"
    version = "1.0.0"

    @property
    def auto_xrd_available(self) -> bool:
        try:
            import autoXRD  # type: ignore
        except ImportError:
            return False
        return autoXRD is not None


__all__ = ["AutoXRDObservableExtractor", "XRDObservableExtractor"]
