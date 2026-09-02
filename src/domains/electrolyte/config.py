from __future__ import annotations

from typing import Any

from src.science.domain import (
    MaterialDomainConfig,
    ModalityDefinition,
    ObjectiveDefinition,
    ObjectiveDirection,
)

# Canonical domain identifier
ELECTROLYTE_DOMAIN_ID: str = "anode_free_electrolyte"

# Scientific 11D solvent descriptor features (physically aligned historical and LiFSI discovery space)
ELECTROLYTE_SOLVENT_FEATURES: tuple[str, ...] = (
    "solv_ecfp_pca_0",
    "solv_ecfp_pca_1",
    "solv_ecfp_pca_2",
    "solv_ecfp_pca_3",
    "solv_ecfp_pca_4",
    "solv_ecfp_pca_5",
    "solv_ecfp_pca_6",
    "solv_ecfp_pca_7",
    "solv_ecfp_pca_8",
    "solv_ecfp_pca_9",
    "mol_wt_solv",
)

# Virtual 22D continuous features (for 1M computational scale evaluation)
ELECTROLYTE_VIRTUAL_FEATURES: tuple[str, ...] = (
    "solv_ecfp_pca_0",
    "solv_ecfp_pca_1",
    "solv_ecfp_pca_2",
    "solv_ecfp_pca_3",
    "solv_ecfp_pca_4",
    "solv_ecfp_pca_5",
    "solv_ecfp_pca_6",
    "solv_ecfp_pca_7",
    "solv_ecfp_pca_8",
    "solv_ecfp_pca_9",
    "salt_ecfp_pca_0",
    "salt_ecfp_pca_1",
    "salt_ecfp_pca_2",
    "salt_ecfp_pca_3",
    "salt_ecfp_pca_4",
    "salt_ecfp_pca_5",
    "salt_ecfp_pca_6",
    "salt_ecfp_pca_7",
    "salt_ecfp_pca_8",
    "salt_ecfp_pca_9",
    "mol_wt_solv",
    "mol_wt_salt",
)

# Primary Discovery Objective: Normalized discharge capacity at cycle 20 (C_norm_20)
# Raw dataset column: norm_capacity_3 (proven: act_capacity_20 / theor_capacity)
ELECTROLYTE_OBJECTIVE_CAPACITY: ObjectiveDefinition = ObjectiveDefinition(
    name="C_norm_20",
    direction=ObjectiveDirection.MAXIMIZE,
    units="dimensionless",
    target_col="C_norm_20",
    threshold=None,  # Quantile-derived thresholds used in benchmarks; no unphysical threshold hardcoded
    metadata={
        "raw_column": "norm_capacity_3",
        "canonical_name": "C_norm_20",
        "meaning": "Normalized discharge capacity at the 20th cycle (C_dis^20 / C_theoretical)",
        "cell_configuration": "Cu||LFP zero-excess anode-free coin cell, 1.0 M LiFSI, 50 uL flooding",
    },
)

# Primary (and only real) Experimental Modality: CAPACITY_TEST
ELECTROLYTE_MODALITY_CAPACITY: ModalityDefinition = ModalityDefinition(
    name="CAPACITY_TEST",
    observation_kind="objective_measurement",
    cost=1.0,
    requires=(),
    objective_names=("C_norm_20",),
    observation_key="C_norm_20",
    metadata={
        "description": "Galvanostatic cycling measurement in Cu||LFP coin cell yielding normalized 20th-cycle discharge capacity.",
    },
)

# Domain Configuration Contract
ELECTROLYTE_DOMAIN_CONFIG: MaterialDomainConfig = MaterialDomainConfig(
    domain_id=ELECTROLYTE_DOMAIN_ID,
    candidate_features=ELECTROLYTE_SOLVENT_FEATURES,
    modalities=(ELECTROLYTE_MODALITY_CAPACITY,),
    objectives=(ELECTROLYTE_OBJECTIVE_CAPACITY,),
    metadata={
        "system": "Single-solvent electrolyte screening for anode-free lithium metal batteries",
        "source_dataset": "AmanchukwuLab/AL-anode-free (2025)",
        "feature_dim": 11,
    },
)
