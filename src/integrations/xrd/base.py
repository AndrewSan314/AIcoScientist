from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from src.science.multimodal.schemas import ScientificObservable


class XRDExtractor(Protocol):
    def extract(self, pattern: Any, candidate_id: str, metadata: Mapping[str, Any] | None = None) -> Sequence[ScientificObservable]:
        ...


__all__ = ["XRDExtractor"]
