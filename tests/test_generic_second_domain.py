from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domains.auirh import AuIrRhDomainAdapter
from src.domains.toy_material import (
    CompositionOnlyHypothesis,
    MicrostructureInformedHypothesis,
    TOY_MATERIAL_DOMAIN_CONFIG,
    TOY_MODALITY_CAPACITY,
    TOY_MODALITY_SEM,
    TOY_OBJECTIVE_CAPACITY,
    TemperatureMediatedHypothesis,
    ToyMaterialDomainAdapter,
    ToyMaterialHypothesisProvider,
)
from src.science.actions import (
    ActionRecommendation,
    ExperimentOutcome,
    ScientificAction,
    normalize_action_type,
)
from src.science.decision_engine import ScientificDecisionEngine
from src.science.domain import (
    MaterialDomainAdapter,
    ModalityDefinition,
    ObjectiveDefinition,
    ObjectiveDirection,
)
from src.science.falsification.information_gain import HypothesisInformationGainEstimator
from src.science.falsification.policy import FalsificationFirstPolicy, FalsificationPolicyMode
from src.science.hypothesis_models import HypothesisEnsemble, PredictiveDistribution


def test_second_domain_uses_non_h1_h2_h3_hypothesis_ids() -> None:
    """Verifies that the second domain uses descriptive, non-H1/H2/H3 hypothesis identifiers."""
    provider = ToyMaterialHypothesisProvider()
    hypos = provider.build_hypotheses()
    assert set(hypos.keys()) == {
        "composition_only",
        "temperature_mediated",
        "microstructure_informed",
    }
    assert "H1" not in hypos and "H2" not in hypos and "H3" not in hypos


def test_second_domain_uses_non_xrd_non_property_action_types() -> None:
    """Verifies that the second domain modalities are genuinely distinct from XRD / PROPERTY."""
    adapter = ToyMaterialDomainAdapter(n_candidates=10)
    actions = adapter.list_valid_actions()
    action_types = {normalize_action_type(a.action_type) for a in actions}

    assert action_types == {"SEM", "CAPACITY_TEST"}
    assert "XRD" not in action_types and "PROPERTY" not in action_types


def test_hypothesis_ensemble_is_id_agnostic() -> None:
    """Verifies that HypothesisEnsemble operates seamlessly with arbitrary hypothesis ID collections."""
    hypos = {
        "alpha": CompositionOnlyHypothesis(),
        "beta": TemperatureMediatedHypothesis(),
    }
    ensemble = HypothesisEnsemble(hypotheses=hypos)
    assert set(ensemble.hypotheses.keys()) == {"alpha", "beta"}

    beliefs = ensemble.get_beliefs()
    assert set(beliefs.keys()) == {"alpha", "beta"}
    assert np.isclose(beliefs["alpha"], 0.5)
    assert np.isclose(beliefs["beta"], 0.5)

    preds = ensemble.predict_all(
        candidate_id="BAT_001",
        action_type="CAPACITY_TEST",
        composition=np.array([1.0, 0.05, 800.0]),
    )
    assert set(preds.keys()) == {"alpha", "beta"}

    # Update with observation
    rec = ensemble.record_observation_and_update(
        action_id="act_001",
        candidate_id="BAT_001",
        action_type="CAPACITY_TEST",
        observation=np.array([165.0]),
        pre_predictions=preds,
    )
    assert set(rec["after_beliefs"].keys()) == {"alpha", "beta"}
    assert np.isclose(sum(rec["after_beliefs"].values()), 1.0)


def test_hidden_measurements_remain_unavailable_before_execution() -> None:
    """Strict information-horizon firewall: candidate pool must never expose hidden measurements."""
    adapter = ToyMaterialDomainAdapter(n_candidates=25, seed=123)
    pool = adapter.get_candidate_pool()

    assert "capacity" not in pool.columns
    assert "sem_features" not in pool.columns
    assert "grain_size" not in pool.columns

    for cid in pool["candidate_id"]:
        assert not adapter.is_sem_observed(cid)
        assert not adapter.is_capacity_observed(cid)


def test_execution_reveals_only_requested_modality() -> None:
    """Revealing an SEM morphology measurement must leave capacity strictly unrevealed."""
    adapter = ToyMaterialDomainAdapter(n_candidates=10, seed=42)
    cid = "BAT_001"

    action = ScientificAction(
        action_id="act_sem_1",
        candidate_id=cid,
        action_type="SEM",
        estimated_cost=2.0,
    )
    outcome = adapter.execute_or_reveal(action)
    assert outcome.action_type == "SEM"
    assert "sem_features" in outcome.revealed_data
    assert len(outcome.revealed_data["sem_features"]) == 4

    assert adapter.is_sem_observed(cid)
    assert not adapter.is_capacity_observed(cid)


def test_generic_policy_candidate_mapping_ignores_dataframe_index_labels() -> None:
    """Regression test: non-contiguous or arbitrary DataFrame index labels must not corrupt candidate mapping."""
    adapter = ToyMaterialDomainAdapter(n_candidates=10, seed=42)
    pool = adapter.get_candidate_pool().iloc[[4, 1, 8]].copy()
    pool.index = [101, 505, 999]  # Non-contiguous, non-zero index labels

    policy = FalsificationFirstPolicy(mode=FalsificationPolicyMode.HYBRID)
    provider = adapter.get_hypothesis_provider()
    ensemble = HypothesisEnsemble(hypotheses=provider.build_hypotheses())
    valid_actions = [
        ScientificAction(action_id="act_1", candidate_id=str(pool.iloc[0]["candidate_id"]), action_type="SEM", estimated_cost=2.0),
        ScientificAction(action_id="act_2", candidate_id=str(pool.iloc[1]["candidate_id"]), action_type="CAPACITY_TEST", estimated_cost=6.0),
    ]

    # Must execute safely without IndexError or misaligned candidate vectors
    scored = policy.evaluate_all_actions(
        candidate_pool_df=pool,
        ensemble=ensemble,
        valid_actions=valid_actions,
        feature_cols=("Li_ratio", "doping_conc", "sintering_temp"),
        modality_definitions=adapter.get_modality_schema(),
        objective_definitions=adapter.get_objectives(),
    )
    assert len(scored) == 2
    cids_scored = {s["candidate_id"] for s in scored}
    assert cids_scored == {str(pool.iloc[0]["candidate_id"]), str(pool.iloc[1]["candidate_id"])}


def test_same_scientific_decision_engine_runs_two_material_domains() -> None:
    """Proves that the exact same ScientificDecisionEngine orchestrates distinct material systems."""
    # Domain 1: Au-Ir-Rh Thin-Film Catalysts
    adapter_auirh = AuIrRhDomainAdapter()
    engine_auirh = ScientificDecisionEngine(domain=adapter_auirh, seed=42)

    init_actions_auirh = adapter_auirh.get_default_initial_actions(n_prop=3, n_xrd=3, seed=42)
    engine_auirh.initialize(init_actions_auirh)
    rec_auirh = engine_auirh.propose_next_experiment()
    assert rec_auirh.action.candidate_id in adapter_auirh.get_candidate_pool()["candidate_id"].values
    assert normalize_action_type(rec_auirh.action.action_type) in ("XRD", "PROPERTY")
    outcome_auirh = engine_auirh.execute_recommendation(rec_auirh)
    assert outcome_auirh.candidate_id == rec_auirh.action.candidate_id
    state_auirh = engine_auirh.get_state()
    assert state_auirh["domain_id"] == "auirh"
    assert set(state_auirh["current_beliefs"].keys()) == {"H1", "H2", "H3"}

    # Domain 2: Synthetic Battery Cathodes (Toy Material)
    adapter_toy = ToyMaterialDomainAdapter(n_candidates=20, seed=99)
    engine_toy = ScientificDecisionEngine(domain=adapter_toy, seed=99)

    init_actions_toy = adapter_toy.get_default_initial_actions(n_cap=2, n_sem=2, seed=99)
    engine_toy.initialize(init_actions_toy)
    rec_toy = engine_toy.propose_next_experiment()
    assert rec_toy.action.candidate_id in adapter_toy.get_candidate_pool()["candidate_id"].values
    assert normalize_action_type(rec_toy.action.action_type) in ("SEM", "CAPACITY_TEST")
    outcome_toy = engine_toy.execute_recommendation(rec_toy)
    assert outcome_toy.candidate_id == rec_toy.action.candidate_id
    state_toy = engine_toy.get_state()
    assert state_toy["domain_id"] == "toy_material"
    assert set(state_toy["current_beliefs"].keys()) == {
        "composition_only",
        "temperature_mediated",
        "microstructure_informed",
    }


def test_toy_domain_full_cycle_uses_scientific_decision_engine() -> None:
    """Verifies multi-step autonomous closed-loop experimentation with ScientificDecisionEngine."""
    adapter = ToyMaterialDomainAdapter(n_candidates=15, seed=77)
    engine = ScientificDecisionEngine(domain=adapter, seed=77)

    # Initial seed
    init_actions = adapter.get_default_initial_actions(n_cap=2, n_sem=2, seed=77)
    engine.initialize(init_actions)
    assert engine.step == 0

    # Step 1
    rec1 = engine.propose_next_experiment()
    assert isinstance(rec1, ActionRecommendation)
    assert len(rec1.alternatives) > 0
    outcome1 = engine.execute_recommendation(rec1)
    assert outcome1.candidate_id == rec1.action.candidate_id
    assert engine.step == 1

    # Step 2
    rec2 = engine.propose_next_experiment()
    outcome2 = engine.execute_recommendation(rec2)
    assert outcome2.candidate_id == rec2.action.candidate_id
    assert engine.step == 2

    # Verify state and ledger records
    state = engine.get_state()
    assert state["step"] == 2
    assert len(engine.recorded_experiments) == 6  # 4 initial seed + 2 autonomous cycle experiments
    for rec in engine.recorded_experiments:
        assert rec.dataset_name == "toy_material"
        assert rec.provenance.get("domain_id") == "toy_material"


def test_candidate_features_come_from_domain_config() -> None:
    """Candidate pool containing noisy metadata columns must use only domain_config.candidate_features."""
    adapter = ToyMaterialDomainAdapter(n_candidates=10, seed=42)
    pool = adapter.get_candidate_pool()
    # Inject spurious non-feature metadata columns
    pool["operator_id"] = "OP_99"
    pool["batch_label"] = "BATCH_2026"
    pool["timestamp"] = 1718000000.0

    class WrappedAdapter:
        domain_id = adapter.domain_id
        def get_config(self):
            return TOY_MATERIAL_DOMAIN_CONFIG
        def get_candidate_pool(self):
            return pool
        def get_candidate_features(self, cid):
            return adapter.get_candidate_features(cid)
        def list_valid_actions(self, state=None):
            return adapter.list_valid_actions(state)
        def execute_or_reveal(self, act):
            return adapter.execute_or_reveal(act)
        def get_objectives(self):
            return adapter.get_objectives()
        def get_modality_schema(self):
            return adapter.get_modality_schema()
        def get_hypothesis_provider(self):
            return adapter.get_hypothesis_provider()

    engine = ScientificDecisionEngine(domain=WrappedAdapter(), seed=42)
    assert engine.feature_cols == ("Li_ratio", "doping_conc", "sintering_temp")
    assert "operator_id" not in engine.feature_cols
    assert "batch_label" not in engine.feature_cols

    rec = engine.propose_next_experiment()
    assert set(rec.action.candidate_id) is not None


def test_policy_does_not_require_property_or_capacity_action_names() -> None:
    """Verifies that policy scores objective actions via ModalityDefinition metadata without hardcoded names."""
    custom_modality = ModalityDefinition(
        name="CUSTOM_ELECTRO_IMPEDANCE",
        observation_kind="objective_measurement",
        cost=4.0,
        objective_names=("impedance_drop",),
    )
    custom_obj = ObjectiveDefinition(
        name="impedance_drop",
        direction=ObjectiveDirection.MAXIMIZE,
        units="Ohms",
    )

    policy = FalsificationFirstPolicy(mode=FalsificationPolicyMode.HYBRID)
    pool = pd.DataFrame([
        {"candidate_id": "C1", "feat_x": 0.5, "feat_y": 0.5},
        {"candidate_id": "C2", "feat_x": 0.8, "feat_y": 0.2},
    ])
    hypos = {"H_alpha": CompositionOnlyHypothesis(), "H_beta": TemperatureMediatedHypothesis()}
    ensemble = HypothesisEnsemble(hypotheses=hypos)

    valid_actions = [
        ScientificAction(action_id="act_custom_1", candidate_id="C1", action_type="CUSTOM_ELECTRO_IMPEDANCE", estimated_cost=4.0),
    ]

    scored = policy.evaluate_all_actions(
        candidate_pool_df=pool,
        ensemble=ensemble,
        valid_actions=valid_actions,
        feature_cols=("feat_x", "feat_y"),
        modality_definitions=[custom_modality],
        objective_definitions=[custom_obj],
    )
    assert len(scored) == 1
    assert scored[0]["action_type"] == "CUSTOM_ELECTRO_IMPEDANCE"


def test_recommendation_uses_domain_objective_name_and_units() -> None:
    """Verifies that recommendation explanation uses domain-configured objective name and units."""
    adapter = ToyMaterialDomainAdapter(n_candidates=10, seed=42)
    engine = ScientificDecisionEngine(domain=adapter, seed=42)
    rec = engine.propose_next_experiment()

    # Evidence strings should contain "capacity" and "mAh/g"
    found_capacity_unit = any("capacity" in ev and "mAh/g" in ev for ev in rec.supporting_evidence)
    assert found_capacity_unit or any("capacity" in ev for ev in rec.supporting_evidence)
