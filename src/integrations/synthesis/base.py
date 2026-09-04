from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SynthesisFeasibility:
    feasible: bool
    score: float
    source: str
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "score": self.score,
            "source": self.source,
            "limitations": list(self.limitations),
        }


class SynthesisFeasibilityAdapter:
    def evaluate(self, candidate_features: Mapping[str, Any]) -> SynthesisFeasibility:
        raise NotImplementedError
