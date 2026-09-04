from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from src.science.multimodal.schemas import ScientificObservable


class ExtractionError(ValueError):
    """Raised when a raw artifact cannot be interpreted safely."""


class ObservableExtractor(Protocol):
    name: str
    version: str

    def extract(
        self,
        raw_measurement: Any,
        candidate_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Sequence[ScientificObservable]:
        ...


class DeterministicExtractor:
    """Base class for CPU-safe extractors with explicit provenance."""

    name = "deterministic_extractor"
    version = "1.0.0"

    def _metadata(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        return dict(metadata or {})


__all__ = ["DeterministicExtractor", "ExtractionError", "ObservableExtractor"]
