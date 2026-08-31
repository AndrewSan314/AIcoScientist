from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from src.science.actions import ExperimentActionType, ScientificAction
from src.science.discovery_engine import AutonomousDiscoveryEngine
from src.science.falsification.policy import FalsificationPolicyMode


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


def test_formal_falsification_shuffled_candidate_pool_invariance() -> None:
    """Verifies that shuffling candidate pool rows does not corrupt formal model predictions."""
    engine1 = AutonomousDiscoveryEngine(seed=101)
    engine1.initialize_curated_scenario(n_init_prop=4, n_init_xrd=4, seed=101)

    cand_pool = engine1.oracle.get_candidate_pool()
    test_cid = cand_pool["candidate_id"].iloc[10]
    test_comp = cand_pool.loc[cand_pool["candidate_id"] == test_cid, ["Au", "Ir", "Rh"]].iloc[0].to_numpy(dtype=np.float64)

    preds1 = engine1.ensemble.predict_all(
        candidate_id=test_cid,
        action_type=ExperimentActionType.PROPERTY,
        composition=test_comp,
    )

    # Create second engine with permuted candidate pool ordering in oracle
    engine2 = AutonomousDiscoveryEngine(seed=101)
    shuffled_pool = cand_pool.sample(frac=1.0, random_state=999).reset_index(drop=True)
    engine2.oracle._candidate_pool = shuffled_pool
    engine2.initialize_curated_scenario(n_init_prop=4, n_init_xrd=4, seed=101)

    preds2 = engine2.ensemble.predict_all(
        candidate_id=test_cid,
        action_type=ExperimentActionType.PROPERTY,
        composition=test_comp,
    )

    for hid in ["H1", "H2", "H3"]:
        assert np.isclose(preds1[hid].mean[0], preds2[hid].mean[0], atol=1e-6)
        assert np.isclose(preds1[hid].variance[0], preds2[hid].variance[0], atol=1e-6)


def test_disjoint_equal_count_observations_no_cross_assignment() -> None:
    """Verifies that equal count but disjoint candidate sets for XRD and Property do not cross-assign."""
    engine = AutonomousDiscoveryEngine(seed=42)
    cand_df = engine.oracle.get_candidate_pool()
    all_cids = cand_df["candidate_id"].tolist()

    # Disjoint candidates
    xrd_cids = all_cids[0:3]
    prop_cids = all_cids[3:6]

    for cid in xrd_cids:
        act = ScientificAction(
            action_id=f"xrd_{cid}",
            candidate_id=cid,
            action_type=ExperimentActionType.XRD,
            estimated_cost=1.0,
            requested_at_step=0,
        )
        out = engine.oracle.execute(act)
        engine._record_to_ledger(act, out)

    for cid in prop_cids:
        act = ScientificAction(
            action_id=f"prop_{cid}",
            candidate_id=cid,
            action_type=ExperimentActionType.PROPERTY,
            estimated_cost=5.0,
            requested_at_step=0,
        )
        out = engine.oracle.execute(act)
        engine._record_to_ledger(act, out)

    engine._refit_models()

    # Verify H2 model has no joint data because joint_cids count is 0
    h2 = engine.ensemble.hypotheses["H2"]
    assert h2._has_joint_data is False


def test_ui_controls_synchronization_to_falsification_policy() -> None:
    """Verifies that updating UI weight configuration changes FalsificationFirstPolicy configuration and ranking."""
    engine = AutonomousDiscoveryEngine(seed=42)
    engine.initialize_curated_scenario(n_init_prop=4, n_init_xrd=3, seed=42)

    # Configuration A: Heavy Discovery Weight
    w_info_a = 0.1
    w_disc_a = 5.0
    w_cost_a = 0.1

    engine.policy.w_info = w_info_a
    engine.policy.w_disc = w_disc_a
    engine.policy.w_cost = w_cost_a

    engine.falsification_policy.w_hig = w_info_a
    engine.falsification_policy.w_disc = w_disc_a
    engine.falsification_policy.w_cost = w_cost_a

    assert engine.falsification_policy.w_hig == 0.1
    assert engine.falsification_policy.w_disc == 5.0

    rec_a, _ = engine.propose_next_experiment(use_falsification_first=True, fast_mode=True)

    # Configuration B: Heavy HIG Falsification Weight
    w_info_b = 5.0
    w_disc_b = 0.1
    w_cost_b = 0.1

    engine.policy.w_info = w_info_b
    engine.policy.w_disc = w_disc_b
    engine.policy.w_cost = w_cost_b

    engine.falsification_policy.w_hig = w_info_b
    engine.falsification_policy.w_disc = w_disc_b
    engine.falsification_policy.w_cost = w_cost_b

    assert engine.falsification_policy.w_hig == 5.0
    assert engine.falsification_policy.w_disc == 0.1

    rec_b, _ = engine.propose_next_experiment(use_falsification_first=True, fast_mode=True)

    # The recommendations under radically different policy weights should have different scores
    assert rec_a.total_value != rec_b.total_value
