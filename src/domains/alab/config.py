from __future__ import annotations

from src.science.domain import (
    MaterialDomainConfig,
    ModalityDefinition,
    ObjectiveDefinition,
    ObjectiveDirection,
)

# Primary Discovery Objective: Reaction Conversion Fraction
ALAB_OBJECTIVE_REACTION_CONVERSION = ObjectiveDefinition(
    name="reaction_conversion",
    direction=ObjectiveDirection.MAXIMIZE,
    units="fraction",
    threshold=0.8,
    metadata={
        "range": (0.0, 1.0),
        "description": "Quantitative solid-state synthesis conversion fraction based on reacted precursor consumption and target formation.",
    },
)

# Experimental Modalities
ALAB_MODALITY_XRD = ModalityDefinition(
    name="XRD",
    observation_kind="characterization",
    cost=1.0,
    observation_key="xrd_embedding",
    metadata={
        "archive": "raw_scans.zip",
        "embedding_dim": 8,
        "description": "Powder X-ray diffraction 2theta intensity scan standardized onto a 450-point angular grid.",
    },
)

ALAB_MODALITY_REFINEMENT = ModalityDefinition(
    name="REFINEMENT",
    observation_kind="derived_analysis",
    cost=0.5,
    requires=("XRD",),
    observation_key="refinement_features",
    metadata={
        "archive": "refinement_pkls.zip",
        "feature_dim": 4,
        "description": "Rietveld quantitative phase refinement extracting phase weight fractions, target purity, and Rwp goodness-of-fit.",
    },
)

ALAB_MODALITY_OUTCOME_TEST = ModalityDefinition(
    name="OUTCOME_TEST",
    observation_kind="objective",
    cost=2.0,
    objective_names=("reaction_conversion",),
    observation_key="reaction_conversion",
    metadata={
        "source": "ledger_outcome",
        "description": "Quantitative synthesis reaction conversion test measuring target phase formation extent.",
    },
)

# Domain Configuration
ALAB_DOMAIN_CONFIG = MaterialDomainConfig(
    domain_id="alab_precursor_genome",
    candidate_features=(
        "reaction_energy_ev_per_atom",
        "heating_temperature_c",
        "heating_time_minutes",
        "precursor_1_idx",
        "precursor_2_idx",
    ),
    objectives=(ALAB_OBJECTIVE_REACTION_CONVERSION,),
    modalities=(
        ALAB_MODALITY_XRD,
        ALAB_MODALITY_REFINEMENT,
        ALAB_MODALITY_OUTCOME_TEST,
    ),
    metadata={
        "domain_name": "A-Lab Precursor Genome Solid-State Synthesis",
        "controllable_variables": (
            "heating_temperature_c",
            "heating_time_minutes",
        ),
        "dataset_key": "precursor_genome_2026",
        "candidate_count": 1035,
        "unique_precursors": 46,
        "citation": "https://doi.org/10.5281/zenodo.21285546",
    },
)
