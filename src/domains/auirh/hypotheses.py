from __future__ import annotations

from typing import Mapping

from src.science.domain import HypothesisProvider
from src.science.hypothesis_models import (
    CompositionSufficientHypothesis,
    LocalStructuralRegimeHypothesis,
    ScientificHypothesisModel,
    StructureInformedHypothesis,
)


class AuIrRhHypothesisProvider:
    """Constructs competing scientific hypotheses for the Au-Ir-Rh catalyst domain."""

    def build_hypotheses(self) -> Mapping[str, ScientificHypothesisModel]:
        """Returns the canonical H1, H2, H3 hypothesis models for Au-Ir-Rh."""
        return {
            "H1": CompositionSufficientHypothesis(),
            "H2": StructureInformedHypothesis(),
            "H3": LocalStructuralRegimeHypothesis(),
        }
