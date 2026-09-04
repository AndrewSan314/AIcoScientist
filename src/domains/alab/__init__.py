from __future__ import annotations

from src.domains.alab.adapter import ALabDomainAdapter
from src.domains.alab.artifact_index import ALabArtifactIndex, ArtifactRef
from src.domains.alab.chemistry import (
    are_chemically_equivalent,
    get_fractional_composition,
    parse_chemical_formula,
    parse_refinement_phases,
)
from src.domains.alab.config import (
    ALAB_CANONICAL_PRECURSORS,
    ALAB_CANDIDATE_FEATURE_NAMES,
    ALAB_DOMAIN_CONFIG,
    ALAB_MODALITY_OUTCOME_TEST,
    ALAB_MODALITY_EDS,
    ALAB_MODALITY_REFINEMENT,
    ALAB_MODALITY_SEM,
    ALAB_MODALITY_XRD,
    ALAB_OBJECTIVE_REACTION_OUTCOME,
)
from src.domains.alab.feature_encoder import ALabFeatureEncoder
from src.domains.alab.hypotheses import (
    ALabHypothesisProvider,
    PrecursorThermodynamicsHypothesis,
    ProcessKineticsHypothesis,
    StructurePhaseInformedHypothesis,
)

__all__ = [
    "ALAB_CANONICAL_PRECURSORS",
    "ALAB_CANDIDATE_FEATURE_NAMES",
    "ALAB_DOMAIN_CONFIG",
    "ALAB_MODALITY_OUTCOME_TEST",
    "ALAB_MODALITY_EDS",
    "ALAB_MODALITY_REFINEMENT",
    "ALAB_MODALITY_SEM",
    "ALAB_MODALITY_XRD",
    "ALAB_OBJECTIVE_REACTION_OUTCOME",
    "ALabArtifactIndex",
    "ALabDomainAdapter",
    "ALabFeatureEncoder",
    "ALabHypothesisProvider",
    "ArtifactRef",
    "PrecursorThermodynamicsHypothesis",
    "ProcessKineticsHypothesis",
    "StructurePhaseInformedHypothesis",
    "are_chemically_equivalent",
    "get_fractional_composition",
    "parse_chemical_formula",
    "parse_refinement_phases",
]
