import os
import pytest
import pandas as pd
import numpy as np

from src.domains.electrolyte.adapter import ElectrolyteDomainAdapter
from src.domains.electrolyte.config import (
    ELECTROLYTE_DOMAIN_ID,
    ELECTROLYTE_OBJECTIVE_CAPACITY,
    ELECTROLYTE_MODALITY_CAPACITY,
    ELECTROLYTE_SOLVENT_FEATURES,
)
from src.domains.electrolyte.data import (
    FORBIDDEN_CANDIDATE_COLUMNS,
    generate_candidate_id,
    extract_candidate_pool_from_derived,
)
from src.domains.electrolyte.oracle import (
    HistoricalElectrolyteOracle,
    UnmeasuredElectrolyteCandidateError,
)
from src.science.actions import ScientificAction

FIXTURE_PATH = "tests/fixtures/electrolyte/pool_compatible_deexpanded_outcomes.csv"


def test_candidate_pool_contains_no_hidden_target():
    """Verifies that the visible candidate pool contains zero ground-truth capacity targets."""
    adapter = ElectrolyteDomainAdapter(derived_outcomes_path=FIXTURE_PATH)
    pool = adapter.get_candidate_pool()

    assert len(pool) > 0
    for forbidden in FORBIDDEN_CANDIDATE_COLUMNS:
        assert forbidden not in pool.columns, f"Forbidden column '{forbidden}' leaked into candidate pool!"


def test_raw_norm_capacity_3_maps_to_C_norm_20():
    """Verifies that the canonical objective is C_norm_20 and records the legacy mapping."""
    obj = ELECTROLYTE_OBJECTIVE_CAPACITY
    assert obj.name == "C_norm_20"
    assert obj.metadata.get("raw_column") == "norm_capacity_3"
    assert "20th cycle" in obj.metadata.get("meaning", "")


def test_only_capacity_test_action_exists():
    """Verifies that CAPACITY_TEST is the only real experimental modality in the domain."""
    adapter = ElectrolyteDomainAdapter(derived_outcomes_path=FIXTURE_PATH)
    modalities = adapter.get_modality_schema()

    assert len(modalities) == 1
    assert modalities[0].name == "CAPACITY_TEST"
    assert modalities[0].observation_kind == "objective_measurement"

    actions = adapter.list_valid_actions()
    for act in actions:
        assert act.action_type == "CAPACITY_TEST"


def test_historical_oracle_reveals_known_candidate():
    """Verifies that the historical oracle reveals genuine measurements for known candidates."""
    adapter = ElectrolyteDomainAdapter(derived_outcomes_path=FIXTURE_PATH)
    pool = adapter.get_candidate_pool()
    cid = pool["candidate_id"].iloc[0]

    action = ScientificAction(action_id="test_act_1", candidate_id=cid, action_type="CAPACITY_TEST")
    outcome = adapter.execute_or_reveal(action)

    assert outcome.candidate_id == cid
    assert "C_norm_20" in outcome.revealed_data
    assert isinstance(outcome.revealed_data["C_norm_20"], float)
    assert outcome.canonical_observation == outcome.revealed_data["C_norm_20"]
    assert outcome.provenance.get("oracle_kind") == "historical_experimental_reveal"
    assert outcome.provenance.get("experimental") is True


def test_historical_oracle_fails_closed_for_unknown_candidate():
    """Verifies fail-closed semantics: raises UnmeasuredElectrolyteCandidateError on unmeasured candidates."""
    adapter = ElectrolyteDomainAdapter(derived_outcomes_path=FIXTURE_PATH)
    fake_cid = "ELEC_NON_EXISTENT_CANDIDATE_123"

    action = ScientificAction(action_id="test_act_unk", candidate_id=fake_cid, action_type="CAPACITY_TEST")
    with pytest.raises(UnmeasuredElectrolyteCandidateError):
        adapter.execute_or_reveal(action)


def test_candidate_ids_stable():
    """Verifies that candidate IDs are deterministic and stable across calls."""
    solv = "CCOCCO"
    salt = "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F"
    id1 = generate_candidate_id(solv, salt)
    id2 = generate_candidate_id(solv, salt)
    assert id1 == id2
    assert id1.startswith("ELEC_")
    assert len(id1) > 10


def test_future_batch_not_used_as_feature():
    """Verifies that batch numbers are never included in the candidate feature space."""
    adapter = ElectrolyteDomainAdapter(derived_outcomes_path=FIXTURE_PATH)
    features = adapter.get_config().candidate_features

    assert "batch" not in features
    assert "batch_index" not in features
    assert "batch_number" not in features
    for f in features:
        assert f in ELECTROLYTE_SOLVENT_FEATURES


def test_get_default_initial_actions_is_pure():
    """Phase 2.1: Verifies get_default_initial_actions does not mutate revealed state."""
    adapter = ElectrolyteDomainAdapter(derived_outcomes_path=FIXTURE_PATH)
    assert len(adapter._revealed_cids) == 0
    assert len(adapter._revealed_capacity_obs) == 0

    init_actions = adapter.get_default_initial_actions(n_seed=3, seed=42)
    assert len(init_actions) == 3

    # Adapter state must remain completely pure
    assert len(adapter._revealed_cids) == 0
    assert len(adapter._revealed_capacity_obs) == 0
    assert adapter.get_observations_by_modality()["CAPACITY_TEST"] == {}


def test_bootstrap_reveals_each_seed_once():
    """Phase 2.2: Verifies that engine.initialize executes/reveals seeds exactly once."""
    from src.science.decision_engine import ScientificDecisionEngine

    adapter = ElectrolyteDomainAdapter(derived_outcomes_path=FIXTURE_PATH)
    engine = ScientificDecisionEngine(domain=adapter, seed=42)

    init_actions = adapter.get_default_initial_actions(n_seed=3, seed=42)
    outcomes = engine.initialize(init_actions)

    assert len(outcomes) == 3
    assert len(adapter._revealed_cids) == 3
    assert len(adapter._revealed_capacity_obs) == 3
    assert len(engine.observations_by_modality["CAPACITY_TEST"]) == 3
    for act in init_actions:
        assert act.candidate_id in adapter._revealed_cids
        assert act.candidate_id in engine.observations_by_modality["CAPACITY_TEST"]


def test_random_action_uses_full_scientific_update_lifecycle():
    """Phase 2.3: Verifies execute_external_action undergoes complete Bayesian update and model refitting."""
    from src.science.decision_engine import ScientificDecisionEngine

    adapter = ElectrolyteDomainAdapter(derived_outcomes_path=FIXTURE_PATH)
    engine = ScientificDecisionEngine(domain=adapter, seed=42)
    init_actions = adapter.get_default_initial_actions(n_seed=3, seed=42)
    engine.initialize(init_actions)

    initial_beliefs = engine.ensemble.get_beliefs()
    valid_actions = adapter.list_valid_actions(engine.get_state())
    assert len(valid_actions) > 0

    # Pick an action externally (like a RANDOM policy would)
    external_action = valid_actions[0]
    outcome = engine.execute_external_action(external_action)

    assert outcome is not None
    assert external_action.candidate_id in engine.observations_by_modality["CAPACITY_TEST"]
    assert engine.step == 1

    # Hypothesis models must be refitted and have sample count = 4
    for h in engine.ensemble.hypotheses.values():
        assert h.is_fitted
        assert h.sample_count == 4

    # Bayesian beliefs must have been updated
    updated_beliefs = engine.ensemble.get_beliefs()
    assert any(abs(updated_beliefs[k] - initial_beliefs[k]) > 1e-6 for k in initial_beliefs)

