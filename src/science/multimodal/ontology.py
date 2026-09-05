from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ObservableDefinition:
    """Machine-readable meaning and validation rules for one observable."""

    name: str
    modality: str
    semantic_definition: str
    units: str | None
    value_range: tuple[float, float] | None
    source_type: str
    extractor_requirements: tuple[str, ...]
    uncertainty_semantics: str
    observable_type: str = "scalar"

    def validate(self, value: Any, uncertainty: Any | None = None) -> None:
        if self.observable_type == "categorical":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{self.name} requires a non-empty categorical value")
        else:
            values = np.asarray(value, dtype=np.float64)
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"{self.name} requires finite values")
            if self.value_range is not None and (
                np.any(values < self.value_range[0]) or np.any(values > self.value_range[1])
            ):
                raise ValueError(f"{self.name} is outside its declared range {self.value_range}")
        if uncertainty is not None:
            errors = np.asarray(uncertainty, dtype=np.float64)
            if not np.all(np.isfinite(errors)) or np.any(errors < 0):
                raise ValueError(f"{self.name} uncertainty must be finite and non-negative")


def _definition(
    name: str,
    modality: str,
    semantic_definition: str,
    units: str | None,
    value_range: tuple[float, float] | None,
    source_type: str,
    extractor_requirements: tuple[str, ...],
    uncertainty_semantics: str,
    observable_type: str = "scalar",
) -> ObservableDefinition:
    return ObservableDefinition(
        name,
        modality,
        semantic_definition,
        units,
        value_range,
        source_type,
        extractor_requirements,
        uncertainty_semantics,
        observable_type,
    )


OBSERVABLE_REGISTRY: dict[str, ObservableDefinition] = {
    definition.name: definition
    for definition in (
        _definition("XRD.normalized_intensity_std_proxy", "XRD", "Standard deviation of min-shifted, peak-scaled intensity; a descriptor, not crystallinity.", "normalized intensity", (0.0, 1.0), "raw_xrd", ("finite intensity vector",), "heuristic descriptor uncertainty"),
        _definition("XRD.dominant_peak_index_fraction", "XRD", "Location of the dominant intensity sample as a fraction of the supplied grid.", "fraction of grid", (0.0, 1.0), "raw_xrd", ("finite intensity vector",), "heuristic descriptor uncertainty"),
        _definition("XRD.global_halfmax_span_proxy", "XRD", "Fractional span of samples above half the maximum intensity; not a calibrated FWHM.", "fraction of grid", (0.0, 1.0), "raw_xrd", ("finite intensity vector",), "heuristic descriptor uncertainty"),
        _definition("XRD.spectral_entropy", "XRD", "Normalized Shannon entropy of non-negative intensity mass.", "normalized entropy", (0.0, 1.0), "raw_xrd", ("finite intensity vector",), "heuristic descriptor uncertainty"),
        _definition("XRD.peak_count_proxy", "XRD", "Count of separated local maxima above a fixed 10% relative threshold.", "count", None, "raw_xrd", ("finite intensity vector",), "heuristic descriptor uncertainty"),
        _definition("REFINEMENT.target_phase_fraction", "REFINEMENT", "Normalized refined phase weight chemically equivalent to the target compound.", "fraction", (0.0, 1.0), "canonical_refinement", ("phase weights", "target formula"), "refinement/model uncertainty"),
        _definition("REFINEMENT.precursor_phase_fraction", "REFINEMENT", "Normalized refined phase weight chemically equivalent to an unreacted precursor.", "fraction", (0.0, 1.0), "canonical_refinement", ("phase weights", "precursor formulas"), "refinement/model uncertainty"),
        _definition("REFINEMENT.other_identified_phase_fraction", "REFINEMENT", "Normalized refined phase weight not assigned to target or precursor phases.", "fraction", (0.0, 1.0), "canonical_refinement", ("phase weights",), "refinement/model uncertainty"),
        _definition("REFINEMENT.rwp_scaled", "REFINEMENT", "Stored Rietveld Rwp divided by 10 for a bounded comparison descriptor.", "scaled Rwp", (0.0, 2.0), "canonical_refinement", ("Rwp",), "refinement/model uncertainty"),
        _definition("SEM.inverse_gradient_scale_proxy", "SEM", "Inverse mean absolute image gradient; no pixel-size calibration implied.", "inverse normalized-pixel gradient", None, "raw_sem", ("finite 2-D image",), "heuristic descriptor uncertainty"),
        _definition("SEM.intensity_texture_std", "SEM", "Standard deviation of normalized grayscale intensity.", "normalized intensity", (0.0, 1.0), "raw_sem", ("finite 2-D image",), "heuristic descriptor uncertainty"),
        _definition("SEM.median_threshold_background_fraction", "SEM", "Fraction below or equal to the image median threshold.", "fraction", (0.0, 1.0), "raw_sem", ("finite 2-D image",), "heuristic descriptor uncertainty"),
        _definition("SEM.foreground_adjacency_fraction", "SEM", "Fraction of horizontal adjacent foreground pixels under a median threshold.", "fraction", (0.0, 1.0), "raw_sem", ("finite 2-D image",), "heuristic descriptor uncertainty"),
        _definition("EDS.composition_error", "EDS", "Absolute deviation of mean tabulated composition from unit sum, or mean row spread when values are not fractions.", "composition units", None, "raw_eds", ("finite numeric table",), "heuristic descriptor uncertainty"),
        _definition("EDS.spatial_variance", "EDS", "Mean per-column variance across tabulated EDS rows.", "squared composition units", None, "raw_eds", ("finite numeric table",), "heuristic descriptor uncertainty"),
        _definition("EDS.segregation_index", "EDS", "Clipped relative row-spread descriptor.", "index", (0.0, 1.0), "raw_eds", ("finite numeric table",), "heuristic descriptor uncertainty"),
        _definition("EDS.element_colocalization", "EDS", "One minus the clipped segregation descriptor.", "index", (0.0, 1.0), "raw_eds", ("finite numeric table",), "heuristic descriptor uncertainty"),
        _definition("OUTCOME.reaction_outcome_utility", "OUTCOME_TEST", "Ordinal decision utility mapped from the categorical reaction outcome; not a conversion fraction.", "ordinal decision utility", (0.0, 1.0), "ledger_outcome", ("reaction category",), "categorical observation mapped to decision utility"),
        _definition("OUTCOME.reaction_category", "OUTCOME_TEST", "Observed synthesis reaction category.", None, None, "ledger_outcome", ("reaction category",), "categorical probability", "categorical"),
    )
}


MODALITY_OBSERVABLE_NAMES: dict[str, tuple[str, ...]] = {
    "XRD": tuple(name for name in OBSERVABLE_REGISTRY if name.startswith("XRD.")),
    "REFINEMENT": tuple(name for name in OBSERVABLE_REGISTRY if name.startswith("REFINEMENT.")),
    "SEM": tuple(name for name in OBSERVABLE_REGISTRY if name.startswith("SEM.")),
    "EDS": tuple(name for name in OBSERVABLE_REGISTRY if name.startswith("EDS.")),
    "OUTCOME_TEST": ("OUTCOME.reaction_outcome_utility",),
}


def observable_names_for_modality(modality: str) -> tuple[str, ...]:
    try:
        return MODALITY_OBSERVABLE_NAMES[str(modality).upper()]
    except KeyError as exc:
        raise ValueError(f"No observable schema registered for modality {modality!r}") from exc


def validate_observable(name: str, value: Any, uncertainty: Any | None = None) -> None:
    try:
        definition = OBSERVABLE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown scientific observable {name!r}") from exc
    definition.validate(value, uncertainty)


__all__ = [
    "MODALITY_OBSERVABLE_NAMES",
    "OBSERVABLE_REGISTRY",
    "ObservableDefinition",
    "observable_names_for_modality",
    "validate_observable",
]
