from __future__ import annotations

from typing import Any, Mapping

from src.integrations.synthesis.base import SynthesisFeasibility, SynthesisFeasibilityAdapter


class S4FeasibilityAdapter(SynthesisFeasibilityAdapter):
    """S4-inspired feasibility prior; it is not a trained synthesis predictor."""

    source = "s4_reference_only_feasibility_prior"

    def evaluate(self, candidate_features: Mapping[str, Any]) -> SynthesisFeasibility:
        present = sum(value is not None for value in candidate_features.values())
        score = min(1.0, present / max(1, len(candidate_features)))
        return SynthesisFeasibility(
            feasible=bool(score > 0.0),
            score=float(score),
            source=self.source,
            limitations=("reference-only prior", "not a calibrated synthesis-success probability"),
        )


__all__ = ["S4FeasibilityAdapter"]
