from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
from src.science.domain import MaterialDomainAdapter
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


def test_hidden_measurement_unavailable_before_action() -> None:
    """Strict information-horizon firewall: candidate pool must never expose hidden measurements."""
    adapter = ToyMaterialDomainAdapter(n_candidates=25, seed=123)
    pool = adapter.get_candidate_pool()

    assert "capacity" not in pool.columns
    assert "sem_features" not in pool.columns
    assert "grain_size" not in pool.columns

    for cid in pool["candidate_id"]:
        assert not adapter.is_sem_observed(cid)
        assert not adapter.is_capacity_observed(cid)


def test_only_requested_measurement_revealed() -> None:
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


def test_second_domain_runs_core_decision_cycle() -> None:
    """Proves that a completely new scientific domain runs the entire core decision and evidence cycle.

    Workflow:
    1. Candidate pool + feature schema from domain adapter
    2. Competing hypotheses constructed from domain provider
    3. Predictive distributions generated across candidates
    4. HIG evaluated with Monte Carlo estimator
    5. Action recommended by FalsificationFirstPolicy
    6. Action executed via adapter
    7. Hypothesis posterior updated on realized evidence
    """
    adapter = ToyMaterialDomainAdapter(n_candidates=15, seed=99)
    pool_df = adapter.get_candidate_pool()
    provider = adapter.get_hypothesis_provider()
    ensemble = HypothesisEnsemble(hypotheses=provider.build_hypotheses())

    initial_beliefs = ensemble.get_beliefs()
    assert set(initial_beliefs.keys()) == {
        "composition_only",
        "temperature_mediated",
        "microstructure_informed",
    }
    initial_entropy = ensemble.get_entropy()
    assert initial_entropy > 0.0

    policy = FalsificationFirstPolicy(
        mode=FalsificationPolicyMode.HYBRID,
        cost_xrd=2.0,  # SEM cost
        cost_property=6.0,  # Capacity test cost
    )

    observed_sem: set[str] = set()
    observed_cap: set[str] = set()

    # Step 1: Recommend next best experiment using domain's valid actions
    valid_actions = adapter.list_valid_actions()
    rec = policy.recommend_next_experiment(
        candidate_pool_df=pool_df,
        ensemble=ensemble,
        step=0,
        valid_actions=valid_actions,
    )
    assert isinstance(rec, ActionRecommendation)
    assert rec.action.candidate_id in pool_df["candidate_id"].values
    assert normalize_action_type(rec.action.action_type) in ("SEM", "CAPACITY_TEST")

    # Step 2: Pre-compute predictions for the chosen action
    c_features = adapter.get_candidate_features(rec.action.candidate_id)
    comp_vec = np.array([c_features["Li_ratio"], c_features["doping_conc"], c_features["sintering_temp"]])

    pre_preds = ensemble.predict_all(
        candidate_id=rec.action.candidate_id,
        action_type=rec.action.action_type,
        composition=comp_vec,
    )
    assert len(pre_preds) == 3

    # Step 3: Execute the action via domain adapter
    outcome = adapter.execute_or_reveal(rec.action)
    assert outcome.candidate_id == rec.action.candidate_id

    if normalize_action_type(rec.action.action_type) == "SEM":
        obs_val = np.array(outcome.revealed_data["sem_features"], dtype=np.float64)
        observed_sem.add(rec.action.candidate_id)
    else:
        obs_val = np.array([float(outcome.revealed_data["capacity"])], dtype=np.float64)
        observed_cap.add(rec.action.candidate_id)

    # Step 4: Update hypothesis ensemble with observed evidence
    update_rec = ensemble.record_observation_and_update(
        action_id=rec.action.action_id,
        candidate_id=rec.action.candidate_id,
        action_type=rec.action.action_type,
        observation=obs_val,
        pre_predictions=pre_preds,
    )

    post_beliefs = ensemble.get_beliefs()
    assert np.isclose(sum(post_beliefs.values()), 1.0)
    assert update_rec["step"] == 1
    assert "log_predictive_scores" in update_rec
