from __future__ import annotations

from src.science.domain import (
    MaterialDomainConfig,
    ModalityDefinition,
    ObjectiveDefinition,
    ObjectiveDirection,
)

# Canonical list of all 46 precursors in the A-Lab Precursor Genome
ALAB_CANONICAL_PRECURSORS: tuple[str, ...] = (
    "(NH4)2HPO4",
    "Ag2O",
    "Al(OH)3",
    "B(OH)3",
    "BaCO3",
    "BaO2",
    "Bi2O3",
    "CaCO3",
    "Co3O4",
    "CoO",
    "Cr2O3",
    "CuO",
    "Fe2O3",
    "Fe3O4",
    "FeC2O4",
    "Ga2O3",
    "GeO2",
    "HfO2",
    "In2O3",
    "K2CO3",
    "KH2PO4",
    "La(OH)3",
    "Li2CO3",
    "LiOH",
    "MgO",
    "Mn2O3",
    "Mn3O4",
    "MnO",
    "MnO2",
    "MoO3",
    "NH4H2PO4",
    "Na2CO3",
    "Nb2O5",
    "NiO",
    "PbO",
    "Sb2O3",
    "SiO2",
    "SnO2",
    "SrCO3",
    "TiO2",
    "V2O3",
    "V2O5",
    "WO3",
    "Y2O3",
    "ZnO",
    "ZrO2",
)

# Candidate model feature names (49 dimensions: 3 process/thermodynamic + 46 multi-hot precursor indicators)
ALAB_CANDIDATE_FEATURE_NAMES: tuple[str, ...] = (
    "reaction_energy_ev_per_atom",
    "heating_temperature_scaled",
    "heating_time_scaled",
) + tuple(f"prec_{p}" for p in ALAB_CANONICAL_PRECURSORS)

# Primary Discovery Objective: Reaction Outcome Utility (Ordinal synthesis category mapped to decision utility)
ALAB_OBJECTIVE_REACTION_OUTCOME = ObjectiveDefinition(
    name="reaction_outcome_utility",
    direction=ObjectiveDirection.MAXIMIZE,
    units="utility",
    threshold=0.8,
    metadata={
        "range": (0.0, 1.0),
        "semantic_type": "ordinal_decision_utility",
        "description": (
            "Ordinal synthesis outcome decision utility (completely_reacted: 1.0, transformed: 0.75, "
            "partially_reacted: 0.5, unreacted: 0.0). Note: Categorical classification mapped to decision utility "
            "for optimization, not measured physical conversion percentage."
        ),
        "category_utility_mapping": {
            "completely_reacted": 1.0,
            "transformed": 0.75,
            "partially_reacted": 0.5,
            "unreacted": 0.0,
        },
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
        "two_theta_range": (10.0, 100.0),
        "two_theta_min": 10.0,
        "two_theta_max": 100.0,
        "grid_points": 450,
        "description": "Powder X-ray diffraction 2theta intensity scan standardized onto a physical 450-point 10-100 deg 2theta grid.",
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
        "description": "Rietveld quantitative phase refinement extracting target fraction, precursor fraction, other phases, and standardized Rwp.",
    },
)

ALAB_MODALITY_OUTCOME_TEST = ModalityDefinition(
    name="OUTCOME_TEST",
    observation_kind="objective",
    cost=2.0,
    objective_names=("reaction_outcome_utility",),
    observation_key="reaction_outcome_utility",
    metadata={
        "source": "ledger_outcome",
        "description": "Synthesis reaction outcome test evaluating qualitative reaction category and derived decision utility.",
    },
)

# Domain Configuration
ALAB_DOMAIN_CONFIG = MaterialDomainConfig(
    domain_id="alab_precursor_genome",
    candidate_features=ALAB_CANDIDATE_FEATURE_NAMES,
    objectives=(ALAB_OBJECTIVE_REACTION_OUTCOME,),
    modalities=(
        ALAB_MODALITY_XRD,
        ALAB_MODALITY_REFINEMENT,
        ALAB_MODALITY_OUTCOME_TEST,
    ),
    metadata={
        "domain_name": "A-Lab Precursor Genome Solid-State Synthesis",
        "controllable_variables": (
            "heating_temperature_scaled",
            "heating_time_scaled",
        ),
        "dataset_key": "precursor_genome_2026",
        "candidate_count": 1035,
        "unique_precursors": 46,
        "feature_dim": len(ALAB_CANDIDATE_FEATURE_NAMES),
        "citation": "https://doi.org/10.5281/zenodo.21285546",
    },
)
