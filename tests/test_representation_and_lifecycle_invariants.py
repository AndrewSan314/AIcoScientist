import numpy as np
import pytest

from src.domains.auirh.adapter import AuIrRhDomainAdapter
from src.science.actions import ActionRecommendation, ExperimentActionType, ScientificAction
from src.science.decision_engine import ScientificDecisionEngine
from src.science.hypothesis_models import (
    CompositionSufficientHypothesis,
    HypothesisEnsemble,
    PredictiveDistribution,
)
from src.science.ledger import ExperimentLedger
from src.science.records import ExperimentStage, ScientificExperimentRecord
from src.science.representation import (
    RepresentationMismatchError,
    RepresentationSnapshot,
)
from src.science.xrd_representation import XRDRepresentationExtractor


def test_xrd_evidence_uses_preregistered_representation_basis():
    """Verifies that evidence evaluation uses the frozen snapshot basis R_N before PCA refits to R_{N+1}."""
    adapter = AuIrRhDomainAdapter()
    engine = ScientificDecisionEngine(domain=adapter, seed=42)

    # Bootstrap with initial actions to establish representation R_N
    init_actions = adapter.get_default_initial_actions(n_property=3, n_characterization=3, seed=42)
    engine.initialize(init_actions)

    snapshot_before = adapter.get_representation_snapshot("XRD")
    assert snapshot_before is not None
    fp_before = snapshot_before.fingerprint

    # Find a valid unobserved XRD action
    valid_xrds = [a for a in adapter.list_valid_actions() if a.action_type == ExperimentActionType.XRD]
    assert len(valid_xrds) > 0
    test_act = valid_xrds[0]

    # Predict under frozen basis
    pre_preds = engine.ensemble.predict_all(
        candidate_id=test_act.candidate_id,
        action_type=test_act.action_type,
        composition=engine._get_candidate_composition(test_act.candidate_id),
        observed_modalities=engine.observations_by_modality,
    )

    # Attach representation fingerprint to pre-predictions as the engine does
    for hid, pred in pre_preds.items():
        pre_preds[hid] = PredictiveDistribution(
            hypothesis_id=pred.hypothesis_id,
            candidate_id=pred.candidate_id,
            action_type=pred.action_type,
            mean=pred.mean,
            variance=pred.variance,
            representation_fingerprint=fp_before,
            representation_version=snapshot_before.version,
            representation_id=snapshot_before.representation_id,
        )

    # Execute and reveal
    outcome = adapter.execute_or_reveal(test_act)
    norm_spec = outcome.revealed_data["normalized_intensity"]

    # Transform with frozen snapshot
    frozen_emb = adapter.transform_with_snapshot("XRD", norm_spec, snapshot_before)

    # Bayesian evidence update must succeed under matching fingerprint
    res = engine.ensemble.record_observation_and_update(
        action_id=test_act.action_id,
        candidate_id=test_act.candidate_id,
        action_type=test_act.action_type,
        observation=frozen_emb,
        pre_predictions=pre_preds,
        observation_representation_fingerprint=fp_before,
    )
    assert res is not None
    assert "after_beliefs" in res

    # Update representation after evidence
    adapter.update_representation_after_evidence("XRD", test_act.candidate_id, norm_spec)
    snapshot_after = adapter.get_representation_snapshot("XRD")
    assert snapshot_after is not None
    assert snapshot_after.version == snapshot_before.version + 1


def test_representation_mismatch_fails_closed():
    """Verifies that an evidence update with mismatched representation basis fails closed."""
    ensemble = HypothesisEnsemble()
    cand_id = "test_cand_01"

    pred_r5 = PredictiveDistribution(
        hypothesis_id="H1",
        candidate_id=cand_id,
        action_type=ExperimentActionType.XRD,
        mean=np.zeros(8),
        variance=np.ones(8),
        representation_fingerprint="fingerprint_R5",
        representation_version=5,
    )

    # Attempt evidence update with observation under R6
    with pytest.raises(RepresentationMismatchError) as exc_info:
        ensemble.record_observation_and_update(
            action_id="act_01",
            candidate_id=cand_id,
            action_type=ExperimentActionType.XRD,
            observation=np.zeros(8),
            pre_predictions={"H1": pred_r5},
            observation_representation_fingerprint="fingerprint_R6",
        )

    assert "Representation mismatch" in str(exc_info.value)
    assert "fingerprint_R5" in str(exc_info.value)
    assert "fingerprint_R6" in str(exc_info.value)


def test_post_update_refit_recomputes_all_revealed_xrd_embeddings():
    """Verifies that all historical revealed XRD embeddings share the current unified basis after update."""
    adapter = AuIrRhDomainAdapter()
    engine = ScientificDecisionEngine(domain=adapter, seed=42)

    init_actions = adapter.get_default_initial_actions(n_property=3, n_characterization=4, seed=42)
    engine.initialize(init_actions)

    revealed_embs = adapter.get_revealed_xrd_embeddings()
    assert len(revealed_embs) == 4

    # Perform a new XRD action through engine
    valid_xrds = [a for a in adapter.list_valid_actions() if a.action_type == ExperimentActionType.XRD]
    test_act = valid_xrds[0]

    engine.execute_recommendation(
        ActionRecommendation(
            action=test_act,
            total_value=0.7,
            scientific_information_value=0.5,
            discovery_value=0.2,
            cost_penalty=0.0,
            hypothesis_id="H1",
            rationale="Test XRD action",
            falsification_criterion="None",
        )
    )

    new_revealed_embs = adapter.get_revealed_xrd_embeddings()
    assert len(new_revealed_embs) == 5

    # Check that engine's XRD observations are synchronized with the updated representation
    engine_xrd_obs = engine.observations_by_modality["XRD"]
    for cid in new_revealed_embs:
        np.testing.assert_allclose(engine_xrd_obs[cid], new_revealed_embs[cid])


def test_auirh_default_initialization_preserves_joint_measurements():
    """Verifies that default AuIrRh bootstrap actions pair property and characterization on the same candidates."""
    adapter = AuIrRhDomainAdapter()
    init_actions = adapter.get_default_initial_actions(n_property=5, n_characterization=5, seed=42)

    prop_cids = [a.candidate_id for a in init_actions if a.action_type == ExperimentActionType.PROPERTY]
    xrd_cids = [a.candidate_id for a in init_actions if a.action_type == ExperimentActionType.XRD]

    assert len(prop_cids) == 5
    assert len(xrd_cids) == 5
    # Under default 'joint' pairing, candidate sets should be identical
    assert set(prop_cids) == set(xrd_cids)


def test_auirh_disjoint_initialization_is_explicit_opt_in():
    """Verifies that disjoint bootstrap actions can be requested explicitly."""
    adapter = AuIrRhDomainAdapter()
    init_actions = adapter.get_default_initial_actions(
        n_property=4,
        n_characterization=4,
        pairing_strategy="disjoint",
        seed=42,
    )

    prop_cids = [a.candidate_id for a in init_actions if a.action_type == ExperimentActionType.PROPERTY]
    xrd_cids = [a.candidate_id for a in init_actions if a.action_type == ExperimentActionType.XRD]

    assert len(prop_cids) == 4
    assert len(xrd_cids) == 4
    # Disjoint candidate sets must have zero overlap
    assert set(prop_cids).isdisjoint(set(xrd_cids))


def test_baseline_ledger_event_contains_imported_scientific_observation(tmp_path):
    """Verifies that baseline ledger events store actual scientific observation payloads."""
    db_path = str(tmp_path / "test_baseline.db")
    ledger = ExperimentLedger(db_path=db_path)
    adapter = AuIrRhDomainAdapter()
    engine = ScientificDecisionEngine(domain=adapter, ledger=ledger, seed=42)

    init_actions = adapter.get_default_initial_actions(n_property=2, n_characterization=2, seed=42)
    engine.initialize(init_actions)

    events = ledger.get_event_stream()
    baseline_events = [e for e in events if e.event_type == "BASELINE_EVIDENCE_IMPORTED"]
    assert len(baseline_events) == 4

    for ev in baseline_events:
        payload = ev.payload
        assert payload["stage"] == "COMPLETED"
        meta = payload.get("proposal_metadata", {})
        # Must contain actual observation
        assert "canonical_observation" in meta or "observed_value" in meta
        if "PROPERTY" in ev.experiment_id:
            assert payload.get("performance", {}).get("k0") is not None


def test_baseline_evidence_validates_as_completed_record(tmp_path):
    """Verifies that record_baseline_evidence enforces COMPLETED stage before validation and storage."""
    db_path = str(tmp_path / "test_baseline_val.db")
    ledger = ExperimentLedger(db_path=db_path)

    # A record in PROPOSED stage must be rejected by record_baseline_evidence
    record_proposed = ScientificExperimentRecord(
        experiment_id="exp_proposed_fail",
        candidate_id="cand_01",
        dataset_name="test_domain",
        stage=ExperimentStage.PROPOSED,
    )

    with pytest.raises(ValueError) as exc_info:
        ledger.record_baseline_evidence(record_proposed)

    assert "record stage must be 'COMPLETED'" in str(exc_info.value)

    # A record in COMPLETED stage must be accepted
    record_completed = ScientificExperimentRecord(
        experiment_id="exp_completed_ok",
        candidate_id="cand_01",
        dataset_name="test_domain",
        stage=ExperimentStage.COMPLETED,
    )
    res = ledger.record_baseline_evidence(record_completed)
    assert res.experiment_id == "exp_completed_ok"
