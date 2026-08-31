from __future__ import annotations

import numpy as np

from src.science.actions import ExperimentActionType, ScientificAction
from src.science.discovery_engine import AutonomousDiscoveryEngine


def test_formal_falsification_engine_curated_scenario_lifecycle() -> None:
    """Tests the complete formal falsification cycle in AutonomousDiscoveryEngine."""
    engine = AutonomousDiscoveryEngine(seed=42)
    # Unequal initial counts: 5 property, 3 XRD
    engine.initialize_curated_scenario(n_init_prop=5, n_init_xrd=3, seed=42)

    assert len(engine.oracle.get_revealed_property_ids()) == 5
    assert len(engine.oracle.get_revealed_xrd_ids()) == 3

    # 1. Propose next experiment using formal falsification mode
    rec1, perspectives1 = engine.propose_next_experiment(use_falsification_first=True, fast_mode=True)
    assert rec1 is not None
    assert rec1.action.candidate_id is not None
    assert rec1.action.action_type in [ExperimentActionType.XRD, ExperimentActionType.PROPERTY]

    # Verify that recommended action is not already observed
    if rec1.action.action_type == ExperimentActionType.XRD:
        assert not engine.oracle.is_xrd_observed(rec1.action.candidate_id)
    else:
        assert not engine.oracle.is_property_observed(rec1.action.candidate_id)

    # Verify finite hypothesis beliefs normalizing to 1
    beliefs1 = engine.ensemble.get_beliefs()
    assert len(beliefs1) == 3
    assert np.isclose(sum(beliefs1.values()), 1.0, atol=1e-5)
    for hid, p in beliefs1.items():
        assert 0.0 <= p <= 1.0

    # 2. Execute the recommended experiment
    exec_summary = engine.execute_experiment(rec1.action)
    assert exec_summary is not None
    assert engine.current_step == 1

    # 3. Refit occurred, now propose second formal recommendation
    rec2, perspectives2 = engine.propose_next_experiment(use_falsification_first=True, fast_mode=True)
    assert rec2 is not None

    beliefs2 = engine.ensemble.get_beliefs()
    assert np.isclose(sum(beliefs2.values()), 1.0, atol=1e-5)
    for hid, p in beliefs2.items():
        assert np.isfinite(p)
        assert 0.0 <= p <= 1.0


def test_formal_falsification_xrd_only_campaign_proposal() -> None:
    """Verifies that an autonomous campaign with XRD observations but zero property observations succeeds."""
    engine = AutonomousDiscoveryEngine(seed=42)
    cand_df = engine.oracle.get_candidate_pool()
    all_cids = cand_df["candidate_id"].tolist()

    # Execute only XRD observations (zero property observations)
    for cid in all_cids[0:4]:
        act = ScientificAction(
            action_id=f"xrd_init_{cid}",
            candidate_id=cid,
            action_type=ExperimentActionType.XRD,
            estimated_cost=1.0,
            requested_at_step=0,
        )
        out = engine.oracle.execute(act)
        engine._record_to_ledger(act, out)

    engine._refit_models()

    assert len(engine.oracle.get_revealed_xrd_ids()) == 4
    assert len(engine.oracle.get_revealed_property_ids()) == 0

    # Request formal recommendation in XRD-only state
    rec, perspectives = engine.propose_next_experiment(use_falsification_first=True, fast_mode=True)
    assert rec is not None
    assert rec.action.candidate_id is not None
    assert np.isfinite(rec.total_value)


def test_formal_falsification_shuffled_candidate_pool_invariance() -> None:
    """Option A Invariance Test: Shuffling candidate pool row order with identical observations must yield identical predictions."""
    engine1 = AutonomousDiscoveryEngine(seed=101)
    cand_pool = engine1.oracle.get_candidate_pool()
    all_cids = cand_pool["candidate_id"].tolist()

    # Select exact explicit property and XRD IDs
    prop_cids = [all_cids[0], all_cids[2], all_cids[5], all_cids[7]]
    xrd_cids = [all_cids[0], all_cids[3], all_cids[5], all_cids[8]]

    # Execute explicit identical actions in Engine 1
    for cid in prop_cids:
        act = ScientificAction(f"prop_{cid}", cid, ExperimentActionType.PROPERTY, 5.0, 0)
        out = engine1.oracle.execute(act)
        engine1._record_to_ledger(act, out)
    for cid in xrd_cids:
        act = ScientificAction(f"xrd_{cid}", cid, ExperimentActionType.XRD, 1.0, 0)
        out = engine1.oracle.execute(act)
        engine1._record_to_ledger(act, out)

    engine1._refit_models()

    test_cid = all_cids[12]
    test_comp = cand_pool.loc[cand_pool["candidate_id"] == test_cid, ["Au", "Ir", "Rh"]].iloc[0].to_numpy(dtype=np.float64)

    preds1_prop = engine1.ensemble.predict_all(test_cid, ExperimentActionType.PROPERTY, test_comp)
    preds1_xrd = engine1.ensemble.predict_all(test_cid, ExperimentActionType.XRD, test_comp)

    # Engine 2: Permuted candidate pool rows, but EXACT SAME observations executed
    engine2 = AutonomousDiscoveryEngine(seed=101)
    shuffled_pool = cand_pool.sample(frac=1.0, random_state=999).reset_index(drop=True)
    engine2.oracle._candidate_pool = shuffled_pool

    for cid in prop_cids:
        act = ScientificAction(f"prop_{cid}", cid, ExperimentActionType.PROPERTY, 5.0, 0)
        out = engine2.oracle.execute(act)
        engine2._record_to_ledger(act, out)
    for cid in xrd_cids:
        act = ScientificAction(f"xrd_{cid}", cid, ExperimentActionType.XRD, 1.0, 0)
        out = engine2.oracle.execute(act)
        engine2._record_to_ledger(act, out)

    engine2._refit_models()

    preds2_prop = engine2.ensemble.predict_all(test_cid, ExperimentActionType.PROPERTY, test_comp)
    preds2_xrd = engine2.ensemble.predict_all(test_cid, ExperimentActionType.XRD, test_comp)

    for hid in ["H1", "H2", "H3"]:
        assert np.isclose(preds1_prop[hid].mean[0], preds2_prop[hid].mean[0], atol=1e-7)
        assert np.isclose(preds1_prop[hid].variance[0], preds2_prop[hid].variance[0], atol=1e-7)
        assert np.allclose(preds1_xrd[hid].mean, preds2_xrd[hid].mean, atol=1e-7)
        assert np.allclose(preds1_xrd[hid].variance, preds2_xrd[hid].variance, atol=1e-7)


def test_disjoint_equal_count_observations_no_cross_assignment() -> None:
    """Verifies that equal count but disjoint candidate sets for XRD and Property do not cross-assign."""
    engine = AutonomousDiscoveryEngine(seed=42)
    cand_df = engine.oracle.get_candidate_pool()
    all_cids = cand_df["candidate_id"].tolist()

    xrd_cids = all_cids[0:3]
    prop_cids = all_cids[3:6]

    for cid in xrd_cids:
        act = ScientificAction(f"xrd_{cid}", cid, ExperimentActionType.XRD, 1.0, 0)
        out = engine.oracle.execute(act)
        engine._record_to_ledger(act, out)

    for cid in prop_cids:
        act = ScientificAction(f"prop_{cid}", cid, ExperimentActionType.PROPERTY, 5.0, 0)
        out = engine.oracle.execute(act)
        engine._record_to_ledger(act, out)

    engine._refit_models()

    h2 = engine.ensemble.hypotheses["H2"]
    assert h2._has_joint_data is False


def test_ui_controls_synchronization_to_falsification_policy() -> None:
    """Verifies that updating UI weight configuration changes FalsificationFirstPolicy configuration and ranking."""
    engine = AutonomousDiscoveryEngine(seed=42)
    engine.initialize_curated_scenario(n_init_prop=4, n_init_xrd=3, seed=42)

    # Heavy Discovery Weight
    engine.policy.w_info = 0.1
    engine.policy.w_disc = 5.0
    engine.policy.w_cost = 0.1

    engine.falsification_policy.w_hig = 0.1
    engine.falsification_policy.w_disc = 5.0
    engine.falsification_policy.w_cost = 0.1

    rec_a, _ = engine.propose_next_experiment(use_falsification_first=True, fast_mode=True)

    # Heavy HIG Falsification Weight
    engine.policy.w_info = 5.0
    engine.policy.w_disc = 0.1
    engine.policy.w_cost = 0.1

    engine.falsification_policy.w_hig = 5.0
    engine.falsification_policy.w_disc = 0.1
    engine.falsification_policy.w_cost = 0.1

    rec_b, _ = engine.propose_next_experiment(use_falsification_first=True, fast_mode=True)

    assert rec_a.total_value != rec_b.total_value
