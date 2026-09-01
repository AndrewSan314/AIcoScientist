"""Toy battery cathode material domain pack for architecture validation."""

from __future__ import annotations

from src.domains.toy_material.adapter import ToyMaterialDomainAdapter
from src.domains.toy_material.config import (
    TOY_MATERIAL_DOMAIN_CONFIG,
    TOY_MODALITY_CAPACITY,
    TOY_MODALITY_SEM,
    TOY_OBJECTIVE_CAPACITY,
)
from src.domains.toy_material.hypotheses import (
    CompositionOnlyHypothesis,
    MicrostructureInformedHypothesis,
    TemperatureMediatedHypothesis,
    ToyMaterialHypothesisProvider,
)

__all__ = [
    "CompositionOnlyHypothesis",
    "MicrostructureInformedHypothesis",
    "TOY_MATERIAL_DOMAIN_CONFIG",
    "TOY_MODALITY_CAPACITY",
    "TOY_MODALITY_SEM",
    "TOY_OBJECTIVE_CAPACITY",
    "TemperatureMediatedHypothesis",
    "ToyMaterialDomainAdapter",
    "ToyMaterialHypothesisProvider",
]
