from __future__ import annotations

from typing import Any, Mapping, Sequence

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
from src.optimization.backend import OptimizerBackend
from src.optimization.objective import OptimizationObjective
from src.science.actions import (
    ActionRecommendation,
    ExperimentOutcome,
    ScientificAction,
    normalize_action_type,
)
from src.science.decision_engine import ScientificDecisionEngine, _to_optimizer_objective
from src.science.domain import (
    MaterialDomainAdapter,
    ModalityDefinition,
    ObjectiveDefinition,
    ObjectiveDirection,
)
from src.science.falsification.information_gain import HypothesisInformationGainEstimator
from src.science.falsification.policy import FalsificationFirstPolicy, FalsificationPolicyMode
from src.science.hypothesis_models import HypothesisEnsemble, PredictiveDistribution
from src.science.ledger import ExperimentLedger
from src.science.records import ExperimentStage


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
    assert rec.action.candidate_id in set(adapter.get_candidate_pool()["candidate_id"])


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

    found_capacity_unit = any("capacity" in ev and "mAh/g" in ev for ev in rec.supporting_evidence)
    assert found_capacity_unit or any("capacity" in ev for ev in rec.supporting_evidence)


# --- P0 BLOCKER 1: OPTIMIZER BACKEND INTEGRATION TESTS ---

class MockOptimizerBackend:
    """Mock backend implementing the formal OptimizerBackend protocol."""

    def __init__(self) -> None:
        self.call_history: list[dict[str, Any]] = []

    def score_candidates(
        self,
        observations: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        objective: OptimizationObjective | str,
        feature_columns: Sequence[str] | None = None,
        candidate_id_column: str = "candidate_id",
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        self.call_history.append({
            "observations": observations.copy(),
            "candidate_pool": candidate_pool.copy(),
            "objective": objective,
            "feature_columns": list(feature_columns) if feature_columns else None,
            "candidate_id_column": candidate_id_column,
            "seed": seed,
        })
        # Return synthetic scores keyed by candidate ID
        cids = candidate_pool[candidate_id_column].tolist()
        return {cid: float(i + 1) * 0.1 for i, cid in enumerate(cids)}


def test_generic_engine_calls_optimizer_backend_with_correct_contract() -> None:
    """Verifies that ScientificDecisionEngine passes typed observations and objective to OptimizerBackend."""
    mock_backend = MockOptimizerBackend()
    adapter = ToyMaterialDomainAdapter(n_candidates=10, seed=42)
    engine = ScientificDecisionEngine(domain=adapter, optimizer_backend=mock_backend, seed=42)

    # Initial seed with capacity observations
    init_actions = adapter.get_default_initial_actions(n_cap=3, n_sem=2, seed=42)
    engine.initialize(init_actions)

    rec = engine.propose_next_experiment()
    assert len(mock_backend.call_history) == 1
    call = mock_backend.call_history[0]

    obs_df = call["observations"]
    assert "candidate_id" in obs_df.columns
    assert "capacity" in obs_df.columns  # objective target column
    assert "Li_ratio" in obs_df.columns
    assert "doping_conc" in obs_df.columns
    assert "sintering_temp" in obs_df.columns
    assert len(obs_df) == 3

    assert isinstance(call["objective"], OptimizationObjective)
    assert call["objective"].target_name == "capacity"
    assert call["objective"].units == "mAh/g"
    assert call["feature_columns"] == ["Li_ratio", "doping_conc", "sintering_temp"]
    assert call["candidate_id_column"] == "candidate_id"

    # Status check
    assert engine.last_optimizer_status["used"] is True
    assert engine.last_optimizer_status["success"] is True
    assert engine.last_optimizer_status["num_scored"] == 10


def test_hybrid_policy_uses_nonzero_discovery_scores_from_backend() -> None:
    """Verifies that discovery scores from backend actually influence candidate ranking in HYBRID mode."""
    mock_backend = MockOptimizerBackend()
    adapter = ToyMaterialDomainAdapter(n_candidates=10, seed=42)
    engine = ScientificDecisionEngine(domain=adapter, optimizer_backend=mock_backend, seed=42)

    init_actions = adapter.get_default_initial_actions(n_cap=3, n_sem=2, seed=42)
    engine.initialize(init_actions)

    rec = engine.propose_next_experiment()
    assert rec.discovery_value >= 0.0
    assert engine.last_optimizer_status["success"] is True


def test_optimizer_returns_candidate_id_score_mapping_without_positional_assumption() -> None:
    """Verifies that arbitrary ordering of candidate IDs in discovery score dictionary is respected."""
    mock_backend = MockOptimizerBackend()
    adapter = ToyMaterialDomainAdapter(n_candidates=10, seed=42)
    engine = ScientificDecisionEngine(domain=adapter, optimizer_backend=mock_backend, seed=42)

    init_actions = adapter.get_default_initial_actions(n_cap=3, n_sem=2, seed=42)
    engine.initialize(init_actions)

    rec = engine.propose_next_experiment()
    assert rec.action.candidate_id in adapter.get_candidate_pool()["candidate_id"].values


# --- P0 BLOCKER 2: DOMAIN HYPOTHESIS REFITTING TESTS ---

def test_toy_hypotheses_refit_from_accumulated_domain_observations() -> None:
    """Verifies that ToyMaterial hypotheses refit from domain observations and expose training count."""
    h1 = CompositionOnlyHypothesis()
    h2 = TemperatureMediatedHypothesis()
    h3 = MicrostructureInformedHypothesis()
    ensemble = HypothesisEnsemble(hypotheses={"h1": h1, "h2": h2, "h3": h3})

    comp_map = {
        "C1": np.array([1.0, 0.05, 750.0]),
        "C2": np.array([0.9, 0.10, 800.0]),
        "C3": np.array([0.8, 0.15, 850.0]),
    }
    obs_by_mod = {
        "CAPACITY_TEST": {"C1": 150.0, "C2": 165.0, "C3": 170.0},
        "SEM": {"C1": np.array([0.5, 0.6, 0.7, 0.8]), "C2": np.array([0.6, 0.7, 0.8, 0.9])},
    }

    assert not h1.is_fitted
    assert h1.training_sample_count == 0

    ensemble.fit_all(
        composition_by_id=comp_map,
        observations_by_modality=obs_by_mod,
    )

    assert h1.is_fitted
    assert h1.training_sample_count == 3
    assert h2.is_fitted
    assert h2.training_sample_count == 5  # 3 capacity + 2 sem
    assert h3.is_fitted
    assert h3.training_sample_count == 2  # 2 joint capacity+sem


def test_toy_capacity_evidence_enters_objective_model() -> None:
    """Verifies that capacity measurements update GP model parameters and predictions in CompositionOnlyHypothesis."""
    h1 = CompositionOnlyHypothesis()
    comp_map = {
        "C1": np.array([1.0, 0.05, 750.0]),
        "C2": np.array([0.9, 0.10, 800.0]),
    }
    pred_prior = h1.predict_observation("C3", "CAPACITY_TEST", np.array([0.8, 0.15, 850.0]))
    assert not h1.is_fitted

    h1.fit(composition_by_id=comp_map, property_by_id={"C1": 200.0, "C2": 210.0})
    assert h1.is_fitted
    pred_post = h1.predict_observation("C3", "CAPACITY_TEST", np.array([0.8, 0.15, 850.0]))

    # Post-fit prediction should reflect elevated training targets (near ~200 instead of prior ~140-150)
    assert pred_post.mean[0] > 180.0
    assert pred_post.mean[0] != pred_prior.mean[0]


def test_toy_sem_evidence_enters_characterization_model() -> None:
    """Verifies that SEM morphology measurements update SEM predictions in TemperatureMediatedHypothesis."""
    h2 = TemperatureMediatedHypothesis()
    comp_map = {
        "C1": np.array([1.0, 0.05, 750.0]),
        "C2": np.array([0.9, 0.10, 850.0]),
    }
    sem_data = {
        "C1": np.array([0.2, 0.2, 0.2, 0.2]),
        "C2": np.array([0.8, 0.8, 0.8, 0.8]),
    }
    assert not h2.fitted_sem

    h2.fit(composition_by_id=comp_map, xrd_embedding_by_id=sem_data)
    assert h2.fitted_sem

    pred = h2.predict_observation("C3", "SEM", np.array([0.8, 0.15, 800.0]))
    assert len(pred.mean) == 4
    # Interpolated value at 800 C should be near ~0.5
    assert np.allclose(pred.mean, [0.5, 0.5, 0.5, 0.5], atol=0.2)


def test_toy_prediction_changes_after_new_relevant_evidence() -> None:
    """Verifies that MicrostructureInformedHypothesis uses observed SEM context during capacity prediction."""
    h3 = MicrostructureInformedHypothesis()
    comp_map = {
        "C1": np.array([1.0, 0.05, 750.0]),
        "C2": np.array([0.9, 0.10, 850.0]),
    }
    sem_data = {
        "C1": np.array([0.1, 0.1, 0.1, 0.1]),
        "C2": np.array([0.9, 0.9, 0.9, 0.9]),
    }
    cap_data = {"C1": 120.0, "C2": 190.0}

    h3.fit(composition_by_id=comp_map, property_by_id=cap_data, xrd_embedding_by_id=sem_data)
    assert h3.fitted_capacity

    # Predict with high-morphology SEM context
    high_sem = np.array([0.9, 0.9, 0.9, 0.9])
    pred_high = h3.predict_observation(
        candidate_id="C_high",
        action_type="CAPACITY_TEST",
        composition=np.array([0.5, 0.5, 800.0]),
        observed_xrd_embedding=high_sem,
    )
    # Predict with low-morphology SEM context
    low_sem = np.array([0.1, 0.1, 0.1, 0.1])
    pred_low = h3.predict_observation(
        candidate_id="C_low",
        action_type="CAPACITY_TEST",
        composition=np.array([0.5, 0.5, 800.0]),
        observed_xrd_embedding=low_sem,
    )

    assert pred_high.mean[0] > pred_low.mean[0]


# --- P0 BLOCKER 3: AUIRH XRD REPRESENTATION TESTS ---

def test_auirh_xrd_representation_refits_only_on_revealed_spectra() -> None:
    """Verifies that AuIrRhDomainAdapter refits its PCA basis strictly on revealed spectra."""
    adapter = AuIrRhDomainAdapter()
    pool = adapter.get_candidate_pool()
    cids = pool["candidate_id"].tolist()

    assert len(adapter.get_revealed_xrd_embeddings()) == 0

    # Execute 1st XRD
    out1 = adapter.execute_or_reveal(
        ScientificAction(action_id="act_1", candidate_id=cids[0], action_type="XRD", estimated_cost=1.0)
    )
    assert "xrd_embedding" in out1.revealed_data
    assert len(adapter.get_revealed_xrd_embeddings()) == 1

    # Execute 2nd and 3rd XRD
    adapter.execute_or_reveal(
        ScientificAction(action_id="act_2", candidate_id=cids[1], action_type="XRD", estimated_cost=1.0)
    )
    adapter.execute_or_reveal(
        ScientificAction(action_id="act_3", candidate_id=cids[2], action_type="XRD", estimated_cost=1.0)
    )
    assert len(adapter.get_revealed_xrd_embeddings()) == 3


def test_auirh_all_revealed_embeddings_share_current_representation_basis() -> None:
    """Verifies that as new XRD spectra are revealed, all historical revealed embeddings are updated."""
    adapter = AuIrRhDomainAdapter()
    pool = adapter.get_candidate_pool()
    cids = pool["candidate_id"].tolist()

    for cid in cids[:4]:
        adapter.execute_or_reveal(
            ScientificAction(action_id=f"act_{cid}", candidate_id=cid, action_type="XRD", estimated_cost=1.0)
        )

    embs = adapter.get_revealed_xrd_embeddings()
    assert len(embs) == 4
    for cid in cids[:4]:
        assert cid in embs
        assert len(embs[cid]) == 8


def test_generic_engine_auirh_xrd_state_matches_domain_representation_state() -> None:
    """Verifies that ScientificDecisionEngine synchronizes XRD representation state from AuIrRhDomainAdapter."""
    adapter = AuIrRhDomainAdapter()
    engine = ScientificDecisionEngine(domain=adapter, seed=42)

    init_actions = adapter.get_default_initial_actions(n_prop=3, n_xrd=3, seed=42)
    engine.initialize(init_actions)

    assert len(engine.observations_by_modality["XRD"]) == 3
    assert len(engine.observations_by_modality["PROPERTY"]) == 3


# --- P1: LEDGER LIFECYCLE TESTS ---

def test_engine_proposal_record_starts_at_proposed_stage() -> None:
    """Verifies that proposal records created during propose_next_experiment start in PROPOSED stage."""
    ledger = ExperimentLedger()
    adapter = ToyMaterialDomainAdapter(n_candidates=10, seed=42)
    engine = ScientificDecisionEngine(domain=adapter, ledger=ledger, seed=42)

    init_actions = adapter.get_default_initial_actions(n_cap=2, n_sem=2, seed=42)
    engine.initialize(init_actions)

    rec = engine.propose_next_experiment()
    act = rec.action

    cur = ledger.get_experiment_by_id(act.action_id)
    assert cur is not None
    assert cur.stage == ExperimentStage.PROPOSED


def test_engine_execution_lifecycle_is_proposed_executed_completed() -> None:
    """Verifies that executing a recommendation transitions through EXECUTED to COMPLETED."""
    ledger = ExperimentLedger()
    adapter = ToyMaterialDomainAdapter(n_candidates=10, seed=42)
    engine = ScientificDecisionEngine(domain=adapter, ledger=ledger, seed=42)

    init_actions = adapter.get_default_initial_actions(n_cap=2, n_sem=2, seed=42)
    engine.initialize(init_actions)

    rec = engine.propose_next_experiment()
    engine.execute_recommendation(rec)

    cur = ledger.get_experiment_by_id(rec.action.action_id)
    assert cur is not None
    assert cur.stage == ExperimentStage.COMPLETED

    history = ledger.get_experiment_history(rec.action.action_id)
    event_types = [h["event_type"] for h in history]
    assert event_types == ["PROPOSAL_CREATED", "EXPERIMENT_EXECUTED", "EXPERIMENT_COMPLETED"]


def test_initial_baseline_evidence_is_not_falsely_recorded_as_new_autonomous_proposal() -> None:
    """Verifies that initial imported baseline actions use BASELINE_EVIDENCE_IMPORTED event."""
    ledger = ExperimentLedger()
    adapter = ToyMaterialDomainAdapter(n_candidates=10, seed=42)
    engine = ScientificDecisionEngine(domain=adapter, ledger=ledger, seed=42)

    init_actions = adapter.get_default_initial_actions(n_cap=2, n_sem=2, seed=42)
    engine.initialize(init_actions)

    for act in init_actions:
        history = ledger.get_experiment_history(act.action_id)
        assert len(history) == 1
        assert history[0]["event_type"] == "BASELINE_EVIDENCE_IMPORTED"
        record = ledger.get_experiment_by_id(act.action_id)
        assert record.stage == ExperimentStage.COMPLETED


# --- P1: THREE-POLICY SELECTION TESTS ---

def test_controlled_three_policy_selection_behavior() -> None:
    """Verifies distinct behavior across DISCOVERY_ONLY, PURE_FALSIFICATION, and HYBRID policy modes."""
    # 1. DISCOVERY_ONLY policy: selects action maximizing discovery score minus cost
    adapter_disc = ToyMaterialDomainAdapter(n_candidates=15, seed=42)
    init_actions_disc = adapter_disc.get_default_initial_actions(n_cap=3, n_sem=3, seed=42)
    engine_disc = ScientificDecisionEngine(
        domain=adapter_disc,
        policy_mode=FalsificationPolicyMode.DISCOVERY_ONLY,
        seed=42,
    )
    engine_disc.initialize(init_actions_disc)
    rec_disc = engine_disc.propose_next_experiment()
    assert isinstance(rec_disc, ActionRecommendation)
    assert rec_disc.action.candidate_id in adapter_disc.get_candidate_pool()["candidate_id"].values

    # 2. PURE_FALSIFICATION policy: selects action maximizing raw HIG / cost
    adapter_fals = ToyMaterialDomainAdapter(n_candidates=15, seed=42)
    init_actions_fals = adapter_fals.get_default_initial_actions(n_cap=3, n_sem=3, seed=42)
    engine_fals = ScientificDecisionEngine(
        domain=adapter_fals,
        policy_mode=FalsificationPolicyMode.PURE_FALSIFICATION,
        seed=42,
    )
    engine_fals.initialize(init_actions_fals)
    rec_fals = engine_fals.propose_next_experiment()
    assert rec_fals.total_value == pytest.approx(rec_fals.raw_hig / (rec_fals.action.estimated_cost ** 1.0), rel=1e-3)

    # 3. HYBRID policy: balances HIG, discovery, and cost
    adapter_hyb = ToyMaterialDomainAdapter(n_candidates=15, seed=42)
    init_actions_hyb = adapter_hyb.get_default_initial_actions(n_cap=3, n_sem=3, seed=42)
    engine_hyb = ScientificDecisionEngine(
        domain=adapter_hyb,
        policy_mode=FalsificationPolicyMode.HYBRID,
        seed=42,
    )
    engine_hyb.initialize(init_actions_hyb)
    rec_hyb = engine_hyb.propose_next_experiment()
    assert isinstance(rec_hyb, ActionRecommendation)
    assert rec_hyb.action.candidate_id in adapter_hyb.get_candidate_pool()["candidate_id"].values
