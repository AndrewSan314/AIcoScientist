from __future__ import annotations

from src.science.domain import (
    MaterialDomainConfig,
    ModalityDefinition,
    ObjectiveDefinition,
    ObjectiveDirection,
)

TOY_MODALITY_SEM = ModalityDefinition(
    name="SEM",
    observation_kind="image_features",
    cost=2.0,
    metadata={"description": "Scanning electron microscopy 4D morphology representation"},
)

TOY_MODALITY_CAPACITY = ModalityDefinition(
    name="CAPACITY_TEST",
    observation_kind="scalar_property",
    cost=6.0,
    metadata={"description": "Galvanostatic cycling capacity measurement", "units": "mAh/g"},
)

TOY_OBJECTIVE_CAPACITY = ObjectiveDefinition(
    name="capacity",
    direction=ObjectiveDirection.MAXIMIZE,
    units="mAh/g",
    target_col="capacity",
    metadata={"description": "Specific discharge capacity at 0.1C rate"},
)

TOY_MATERIAL_DOMAIN_CONFIG = MaterialDomainConfig(
    domain_id="toy_material",
    candidate_features=("Li_ratio", "doping_conc", "sintering_temp"),
    modalities=(TOY_MODALITY_SEM, TOY_MODALITY_CAPACITY),
    objectives=(TOY_OBJECTIVE_CAPACITY,),
    metadata={
        "material_system": "Li-Mn-Ni Oxide Cathodes",
        "description": "Synthetic battery cathode exploration domain for architectural portability validation",
    },
)
