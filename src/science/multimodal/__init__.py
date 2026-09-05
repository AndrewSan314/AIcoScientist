"""Small, dependency-light contracts for multimodal scientific decisions."""

from src.science.multimodal.decision import (
    MultimodalDecisionEngine,
    MultimodalRecommendation,
    MultimodalScientificDecisionEngine,
)
from src.science.multimodal.hypotheses import (
    CompositionHomogeneityLimitedHypothesis,
    MorphologyKineticsLimitedHypothesis,
    MultimodalScientificHypothesis,
    PhasePurityLimitedHypothesis,
    build_alab_multimodal_hypotheses,
)
from src.science.multimodal.measurement_models import PredictiveObservableDistribution
from src.science.multimodal.ontology import MODALITY_OBSERVABLE_NAMES, OBSERVABLE_REGISTRY
from src.science.multimodal.schemas import ScientificObservable, ScientificObservableBundle

__all__ = [
    "CompositionHomogeneityLimitedHypothesis",
    "MorphologyKineticsLimitedHypothesis",
    "MultimodalDecisionEngine",
    "MultimodalRecommendation",
    "MultimodalScientificDecisionEngine",
    "MultimodalScientificHypothesis",
    "PhasePurityLimitedHypothesis",
    "PredictiveObservableDistribution",
    "ScientificObservable",
    "ScientificObservableBundle",
    "MODALITY_OBSERVABLE_NAMES",
    "OBSERVABLE_REGISTRY",
    "build_alab_multimodal_hypotheses",
]
