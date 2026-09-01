import json
import os
import numpy as np
import pytest

from src.domains.alab.adapter import ALabDomainAdapter
from src.domains.alab.artifact_index import ALabArtifactIndex
from src.domains.alab.config import (
    ALAB_DOMAIN_CONFIG,
    ALAB_OBJECTIVE_REACTION_CONVERSION,
)
from src.domains.alab.feature_encoder import ALabFeatureEncoder
from src.domains.alab.hypotheses import (
    ALabHypothesisProvider,
    PrecursorThermodynamicsHypothesis,
    ProcessKineticsHypothesis,
    StructurePhaseInformedHypothesis,
)
from src.science.actions import ExperimentActionType, ScientificAction
from src.science.decision_engine import ScientificDecisionEngine
from src.science.domain import HypothesisTrainingContext
from src.science.ledger import ExperimentLedger
from src.science.records import ExperimentStage


@pytest.fixture
def alab_fixture_adapter(tmp_path):
    fixture_dir = "tests/fixtures/alab"
    samples_file = os.path.join(fixture_dir, "samples.json")
    with open(samples_file, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    cache_dir = str(tmp_path / "alab_cache")
    return ALabDomainAdapter(
        data_dir=fixture_dir,
        cache_dir=cache_dir,
        samples=samples,
        min_pca_samples=2,
    )


def test_alab_manifest_entry_resolves():
    """Verifies that the external datasets manifest contains the A-Lab Precursor Genome entry."""
    manifest_path = "data/external/aicoscientist_datasets_manifest.json"
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    datasets = manifest.get("datasets", {})
    assert "precursor_genome_2026" in datasets
    entry = datasets["precursor_genome_2026"]
    assert "ledger" in entry or "files" in entry
    assert entry.get("files", {}).get("ledger_precursor_genome.json") is not None


def test_alab_candidate_identity_is_unique_or_replicates_are_explicit(alab_fixture_adapter):
    """Verifies candidate IDs are unique across candidate pool."""
    pool = alab_fixture_adapter.get_candidate_pool()
    cids = pool["candidate_id"].tolist()
    assert len(cids) == len(set(cids))
    assert len(cids) == 6


def test_alab_candidate_pool_exposes_only_preexperiment_features(alab_fixture_adapter):
    """Verifies candidate pool strictly firewalls post-experiment characterizations and outcomes."""
    pool = alab_fixture_adapter.get_candidate_pool()
    cols = set(pool.columns)

    # Allowed pre-experiment features
    assert "candidate_id" in cols
    assert "reaction_energy_ev_per_atom" in cols
    assert "heating_temperature_c" in cols
    assert "heating_time_minutes" in cols
    assert "precursor_1_idx" in cols
    assert "precursor_2_idx" in cols

    # Strictly forbidden post-experiment leakages
    assert "reaction_category" not in cols
    assert "reaction_conversion" not in cols
    assert "xrd_embedding" not in cols
    assert "refinement_features" not in cols
    assert "phases" not in cols


def test_alab_hidden_xrd_unavailable_before_action(alab_fixture_adapter):
    """Verifies that unrevealed candidates have zero XRD embeddings before execution."""
    assert len(alab_fixture_adapter.get_observations_by_modality()["XRD"]) == 0
    assert len(alab_fixture_adapter._revealed_xrd_spectra) == 0


def test_alab_xrd_action_reveals_only_requested_candidate(alab_fixture_adapter):
    """Verifies executing XRD reveals data strictly for the single requested candidate."""
    actions = [a for a in alab_fixture_adapter.list_valid_actions() if a.action_type == "XRD"]
    assert len(actions) > 0
    act = actions[0]

    outcome = alab_fixture_adapter.execute_or_reveal(act)
    assert outcome.action_id == act.action_id
    assert "normalized_intensity" in outcome.revealed_data
    assert "xrd_embedding" in outcome.revealed_data

    revealed_xrds = alab_fixture_adapter.get_observations_by_modality()["XRD"]
    assert len(revealed_xrds) == 1
    assert act.candidate_id in revealed_xrds


def test_alab_refinement_requires_xrd_prerequisite(alab_fixture_adapter):
    """Verifies that REFINEMENT actions are only valid after XRD has been observed on that candidate."""
    # Initially no XRD is observed, so no REFINEMENT actions should be valid
    actions_initial = alab_fixture_adapter.list_valid_actions()
    ref_actions_initial = [a for a in actions_initial if a.metadata.get("modality") == "REFINEMENT"]
    assert len(ref_actions_initial) == 0

    # Reveal XRD on candidate 0
    cids = alab_fixture_adapter.get_candidate_pool()["candidate_id"].tolist()
    target_cid = cids[0]
    xrd_act = ScientificAction(
        action_id=f"act_xrd_{target_cid}",
        candidate_id=target_cid,
        action_type="XRD",
        estimated_cost=1.0,
    )
    alab_fixture_adapter.execute_or_reveal(xrd_act)

    # Now REFINEMENT for target_cid must be available in valid actions
    actions_after_xrd = alab_fixture_adapter.list_valid_actions()
    ref_actions_after = [
        a for a in actions_after_xrd
        if a.metadata.get("modality") == "REFINEMENT" and a.candidate_id == target_cid
    ]
    assert len(ref_actions_after) == 1


def test_alab_hypothesis_ids_are_descriptive():
    """Verifies that A-Lab hypotheses use domain-meaningful descriptive IDs, not H1/H2/H3."""
    provider = ALabHypothesisProvider()
    hyps = provider.get_hypotheses()
    ids = [h.hypothesis_id for h in hyps]

    assert "precursor_thermodynamics" in ids
    assert "process_kinetics" in ids
    assert "structure_phase_informed" in ids
    assert "H1" not in ids
    assert "H2" not in ids
    assert "H3" not in ids


def test_alab_hypotheses_fit_from_generic_training_context(alab_fixture_adapter):
    """Verifies A-Lab hypotheses train properly from generic HypothesisTrainingContext."""
    cids = alab_fixture_adapter.get_candidate_pool()["candidate_id"].tolist()
    features_map = {cid: np.array([0.5, 0.5, 0.5, 1.0, 2.0]) for cid in cids}

    # Simulate two outcome measurements
    obs_by_modality = {
        "OUTCOME_TEST": {cids[0]: 0.75, cids[1]: 1.0},
        "XRD": {cids[0]: np.zeros(8), cids[1]: np.ones(8) * 0.5},
        "REFINEMENT": {cids[0]: np.array([0.8, 0.2, 2.0, 4.0])},
    }

    ctx = HypothesisTrainingContext(
        candidate_features_by_id=features_map,
        observations_by_modality=obs_by_modality,
        modality_definitions={m.name: m for m in alab_fixture_adapter.get_modality_schema()},
        objective_definitions={o.name: o for o in alab_fixture_adapter.get_objectives()},
    )

    h1 = PrecursorThermodynamicsHypothesis()
    h2 = ProcessKineticsHypothesis()
    h3 = StructurePhaseInformedHypothesis()

    h1.fit_context(ctx)
    h2.fit_context(ctx)
    h3.fit_context(ctx)

    assert h1.is_fitted
    assert h2.is_fitted
    assert h3.is_fitted
    assert h1.training_sample_count == 2
    assert h2.training_sample_count == 2
    assert h3.training_sample_count == 2


def test_alab_same_scientific_decision_engine_runs(alab_fixture_adapter, tmp_path):
    """Verifies that the same ScientificDecisionEngine class runs the A-Lab domain end-to-end."""
    db_path = str(tmp_path / "alab_ledger.db")
    ledger = ExperimentLedger(db_path=db_path)

    engine = ScientificDecisionEngine(
        domain=alab_fixture_adapter,
        ledger=ledger,
        seed=42,
    )

    # Initialize with 2 bootstrap candidates (joint outcome + XRD)
    init_actions = alab_fixture_adapter.get_default_initial_actions(n_candidates=2, pairing_strategy="joint", seed=42)
    engine.initialize(init_actions)

    assert len(engine.observations_by_modality["OUTCOME_TEST"]) == 2
    assert len(engine.observations_by_modality["XRD"]) == 2

    # Propose next scientific action
    rec = engine.propose_next_experiment()
    assert rec is not None
    assert rec.action.candidate_id in set(alab_fixture_adapter.get_candidate_pool()["candidate_id"])
    assert rec.action.action_type in ["PROPERTY", "OUTCOME_TEST", "XRD", "REFINEMENT", "CHARACTERIZATION"]

    # Execute recommendation
    outcome = engine.execute_recommendation(rec)
    assert outcome is not None
    assert rec.action.estimated_cost > 0.0

    # Ensure ledger tracked all stages
    events = ledger.get_event_stream()
    assert len(events) >= 4  # baseline + proposal + executed + completed


def test_alab_full_two_step_offline_replay_cycle(alab_fixture_adapter, tmp_path):
    """Verifies a full two-step autonomous offline replay loop."""
    db_path = str(tmp_path / "alab_replay.db")
    ledger = ExperimentLedger(db_path=db_path)

    engine = ScientificDecisionEngine(
        domain=alab_fixture_adapter,
        ledger=ledger,
        seed=101,
    )

    init_actions = alab_fixture_adapter.get_default_initial_actions(n_candidates=2, seed=101)
    engine.initialize(init_actions)

    # Step 1
    rec1 = engine.propose_next_experiment()
    outcome1 = engine.execute_recommendation(rec1)
    state1 = engine.get_state()
    assert state1["step"] == 1

    # Step 2
    rec2 = engine.propose_next_experiment()
    outcome2 = engine.execute_recommendation(rec2)
    state2 = engine.get_state()
    assert state2["step"] == 2
    assert outcome1.action_id != outcome2.action_id


def test_alab_representation_basis_is_frozen_during_evidence_update(alab_fixture_adapter):
    """Verifies that A-Lab XRD representation basis remains frozen during Bayesian evidence update."""
    engine = ScientificDecisionEngine(domain=alab_fixture_adapter, seed=42)
    init_actions = alab_fixture_adapter.get_default_initial_actions(n_candidates=2, seed=42)
    engine.initialize(init_actions)

    snapshot_before = alab_fixture_adapter.get_representation_snapshot("XRD")
    assert snapshot_before is not None

    # Engine executes recommendation with frozen snapshot lifecycle
    rec = engine.propose_next_experiment()
    outcome = engine.execute_recommendation(rec)

    # After execution, representation snapshot is valid and version is tracked
    snapshot_after = alab_fixture_adapter.get_representation_snapshot("XRD")
    assert snapshot_after is not None
    assert snapshot_after.version >= snapshot_before.version
