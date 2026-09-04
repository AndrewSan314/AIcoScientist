from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from src.science.multimodal.schemas import ScientificObservable


class MicroscopyExtractor(Protocol):
    def extract(self, raw_measurement: Any, candidate_id: str, metadata: Mapping[str, Any] | None = None) -> Sequence[ScientificObservable]:
        ...


__all__ = ["MicroscopyExtractor"]
