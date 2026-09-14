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

__all__ = [
    "DatasetManifest", "DatasetValidationReport", "ArtisticRunDirectoryAdapter", "GenericTabularAdapter", "ProcessSurrogate",
    "ProcessSurrogateSample", "SurrogateDecisionContext", "SurrogateInputSchema", "SplitManifest", "SurrogateArtifact", "split_groups",
    "MultiFidelityFitReport", "PairedFidelityResidualCorrection",
]
