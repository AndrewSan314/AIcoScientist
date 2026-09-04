from __future__ import annotations

from src.domains.electrolyte.adapter import ElectrolyteDomainAdapter
from src.domains.electrolyte.config import (
    ELECTROLYTE_DOMAIN_CONFIG,
    ELECTROLYTE_DOMAIN_ID,
    ELECTROLYTE_MODALITY_CAPACITY,
    ELECTROLYTE_OBJECTIVE_CAPACITY,
    ELECTROLYTE_SOLVENT_FEATURES,
    ELECTROLYTE_VIRTUAL_FEATURES,
)
from src.domains.electrolyte.data import (
    generate_candidate_id,
    load_derived_historical_outcomes,
    load_electrolyte_data_contract,
    load_lifsi_virtual_candidate_chunk,
)
from src.domains.electrolyte.hypotheses import (
    ElectrolyteHypothesisProvider,
    GlobalSmoothDescriptorHypothesis,
    LocalChemicalRegimeHypothesis,
    SparseAdditiveDescriptorHypothesis,
)
from src.domains.electrolyte.oracle import (
    HistoricalElectrolyteOracle,
    SurrogateElectrolyteOracle,
    UnmeasuredElectrolyteCandidateError,
)
from src.domains.electrolyte.screening import (
    ScreeningEvidenceMode,
    benchmark_large_pool_screening,
    screen_large_pool_candidates,
)

__all__ = [
    "ELECTROLYTE_DOMAIN_ID",
    "ELECTROLYTE_DOMAIN_CONFIG",
    "ELECTROLYTE_OBJECTIVE_CAPACITY",
    "ELECTROLYTE_MODALITY_CAPACITY",
    "ELECTROLYTE_SOLVENT_FEATURES",
    "ELECTROLYTE_VIRTUAL_FEATURES",
    "ElectrolyteDomainAdapter",
    "HistoricalElectrolyteOracle",
    "SurrogateElectrolyteOracle",
    "UnmeasuredElectrolyteCandidateError",
    "ElectrolyteHypothesisProvider",
    "GlobalSmoothDescriptorHypothesis",
    "SparseAdditiveDescriptorHypothesis",
    "LocalChemicalRegimeHypothesis",
    "generate_candidate_id",
    "load_derived_historical_outcomes",
    "load_electrolyte_data_contract",
    "load_lifsi_virtual_candidate_chunk",
    "ScreeningEvidenceMode",
    "screen_large_pool_candidates",
    "benchmark_large_pool_screening",
]
