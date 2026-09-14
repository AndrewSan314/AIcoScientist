"""Dataset-pluggable, leakage-safe process surrogate utilities."""

from .core import (
    DatasetManifest,
    DatasetValidationReport,
    ArtisticRunDirectoryAdapter,
    GenericTabularAdapter,
    ProcessSurrogate,
    ProcessSurrogateSample,
    SurrogateDecisionContext,
    SurrogateInputSchema,
    SplitManifest,
    SurrogateArtifact,
    split_groups,
)
from .multifidelity import MultiFidelityFitReport, PairedFidelityResidualCorrection
from .rollout import OneStageLookaheadRollout, one_stage_lookahead

__all__ = [
    "DatasetManifest", "DatasetValidationReport", "ArtisticRunDirectoryAdapter", "GenericTabularAdapter", "ProcessSurrogate",
    "ProcessSurrogateSample", "SurrogateDecisionContext", "SurrogateInputSchema", "SplitManifest", "SurrogateArtifact", "split_groups",
    "MultiFidelityFitReport", "PairedFidelityResidualCorrection",
    "OneStageLookaheadRollout", "one_stage_lookahead",
]
