from __future__ import annotations

from src.science.domain import (
    MaterialDomainConfig,
    ModalityDefinition,
    ObjectiveDefinition,
    ObjectiveDirection,
)

AUIRH_MODALITY_XRD = ModalityDefinition(
    name="XRD",
    observation_kind="spectrum",
    cost=1.0,
    metadata={"format": "xy_diffractogram", "embedding_dim": 8},
)

AUIRH_MODALITY_PROPERTY = ModalityDefinition(
    name="PROPERTY",
    observation_kind="scalar_property",
    cost=5.0,
    metadata={"target": "k0", "units": "cm/s", "method": "SECCM"},
)

AUIRH_OBJECTIVE_K0 = ObjectiveDefinition(
    name="k0",
    direction=ObjectiveDirection.MAXIMIZE,
    units="cm/s",
    target_col="k0",
    metadata={"description": "Electrochemical rate constant for hydrogen evolution reaction"},
)

AUIRH_DOMAIN_CONFIG = MaterialDomainConfig(
    domain_id="auirh",
    candidate_features=("Au", "Ir", "Rh"),
    modalities=(AUIRH_MODALITY_XRD, AUIRH_MODALITY_PROPERTY),
    objectives=(AUIRH_OBJECTIVE_K0,),
    metadata={
        "material_system": "Au-Ir-Rh",
        "description": "Noble metal thin-film catalyst library for electrocatalytic HER",
        "total_candidates": 966,
    },
)
