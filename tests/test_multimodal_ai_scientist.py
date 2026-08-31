from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.datasets.auirh_actions import AuIrRhMultimodalOracle
from src.science.actions import (
    ActionRecommendation,
    ExperimentActionType,
    ExperimentOutcome,
    ScientificAction,
)
from src.science.agents import (
    EvidenceAuditorAgent,
    EvidenceProvenanceAgent,
    ExperimentDesignerAgent,
    FalsificationScientistAgent,
    HypothesisScientistAgent,
    MultiAgentPresentationLayer,
)
from src.science.discovery_engine import AutonomousDiscoveryEngine
from src.science.hypotheses import HypothesisEngine, get_default_hypotheses
from src.science.experiment_policy import NextBestExperimentPolicy
from src.science.scientific_models import PropertySurrogateModel, StructureSurrogateModel
from src.science.xrd_representation import XRDRepresentationExtractor


# ---------------------------------------------------------------------------
# Test Fixture: Clean Oracle
# ---------------------------------------------------------------------------
@pytest.fixture
def auirh_oracle() -> AuIrRhMultimodalOracle:
    return AuIrRhMultimodalOracle()


# ---------------------------------------------------------------------------
# 1. Action Space Validity
# ---------------------------------------------------------------------------
def test_action_space_validity(auirh_oracle: AuIrRhMultimodalOracle) -> None:
    engine = AutonomousDiscoveryEngine(oracle=auirh_oracle, seed=42)
    engine.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=42)
    cand_df = auirh_oracle.get_candidate_pool()
    observed_xrd = set(auirh_oracle.get_revealed_xrd_ids())
    observed_prop = set(auirh_oracle.get_revealed_property_ids())

    scored = engine.policy.evaluate_actions(
        candidate_pool_df=cand_df,
        observed_xrd_ids=observed_xrd,
        observed_property_ids=observed_prop,
        structure_model=engine.structure_model,
        property_model=engine.property_model,
        hypothesis_engine=engine.hypothesis_engine,
    )
    assert len(scored) > 0
    for a in scored:
        assert a["action_type"] in {ExperimentActionType.XRD, ExperimentActionType.PROPERTY}
        assert a["candidate_id"].startswith("AUIRH_")


# ---------------------------------------------------------------------------
# 2. Repeat Actions Rejected
# ---------------------------------------------------------------------------
def test_repeat_actions_rejected(auirh_oracle: AuIrRhMultimodalOracle) -> None:
    cand_id = "AUIRH_Au-rich_005"
    auirh_oracle.execute_xrd(cand_id)
    with pytest.raises(ValueError, match="XRD characterization already executed"):
        auirh_oracle.execute_xrd(cand_id)

    auirh_oracle.execute_property(cand_id)
    with pytest.raises(ValueError, match="Property measurement already executed"):
        auirh_oracle.execute_property(cand_id)


# ---------------------------------------------------------------------------
# 3. Budget Accounting Exact
# ---------------------------------------------------------------------------
def test_budget_accounting_exact(auirh_oracle: AuIrRhMultimodalOracle) -> None:
    engine = AutonomousDiscoveryEngine(oracle=auirh_oracle, cost_xrd=1.5, cost_property=6.0, seed=42)
    engine.initialize_curated_scenario(n_init_prop=3, n_init_xrd=2, seed=42)
    # 3 * 6.0 + 2 * 1.5 = 18.0 + 3.0 = 21.0
    assert np.isclose(engine.total_budget_spent, 21.0)


# ---------------------------------------------------------------------------
# 4. Public Oracle Firewall & Defensive Copies
# ---------------------------------------------------------------------------
def test_public_oracle_firewall(auirh_oracle: AuIrRhMultimodalOracle) -> None:
    cand_df = auirh_oracle.get_candidate_pool()
    assert len(cand_df) == 966
    assert set(cand_df.columns) == {"candidate_id", "Library", "Area", "Au", "Ir", "Rh"}

    # Defensive copies
    revealed_xrd = auirh_oracle.get_revealed_xrd()
    assert isinstance(revealed_xrd, dict)
    revealed_xrd["fake"] = None  # Mutation should not affect internal oracle state
    assert "fake" not in auirh_oracle.get_revealed_xrd()


# ---------------------------------------------------------------------------
# 5. Unrevealed Spectra Never Accessed
# ---------------------------------------------------------------------------
def test_unrevealed_spectra_never_accessed(auirh_oracle: AuIrRhMultimodalOracle) -> None:
    obs_df = auirh_oracle.get_observable_dataset()
    assert obs_df["xrd_observed"].sum() == 0
    assert obs_df["property_observed"].sum() == 0
    assert obs_df["k0"].isna().all()


# ---------------------------------------------------------------------------
# 6. PCA Low Sample Fallback
# ---------------------------------------------------------------------------
def test_pca_low_sample_fallback() -> None:
    extractor = XRDRepresentationExtractor(min_pca_samples=3)
    spec1 = np.linspace(0, 1, 450)
    extractor.fit([spec1])
    assert not extractor.is_pca_fitted
    emb1 = extractor.transform(spec1)
    assert len(emb1) == 8


# ---------------------------------------------------------------------------
# 7. Property Surrogate Fits and Predicts
# ---------------------------------------------------------------------------
def test_property_surrogate_fits_and_predicts() -> None:
    model = PropertySurrogateModel(random_state=42)
    comps = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    targets = np.array([0.001, 0.005, 0.003])
    model.fit(comps, targets)
    assert model.is_fitted
    means, stds = model.predict(comps)
    assert len(means) == 3
    assert len(stds) == 3
    assert (stds >= 0.0).all()


# ---------------------------------------------------------------------------
# 8. Structure Surrogate Fits and Predicts
# ---------------------------------------------------------------------------
def test_structure_surrogate_fits_and_predicts() -> None:
    model = StructureSurrogateModel(random_state=42)
    comps = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    embs = np.random.default_rng(42).normal(size=(3, 8))
    model.fit(comps, embs)
    assert model.is_fitted
    pred_embs, unc = model.predict(comps)
    assert pred_embs.shape == (3, 8)
    assert len(unc) == 3


# ---------------------------------------------------------------------------
# 9. Structure Advantage LOO-CV Contract
# ---------------------------------------------------------------------------
def test_structure_advantage_loo_cv() -> None:
    model = PropertySurrogateModel(random_state=42)
    comps = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    targets = np.array([0.001, 0.005])
    embs = np.random.default_rng(42).normal(size=(2, 8))

    # N < 3 -> neutral
    res_small = model.evaluate_structure_predictive_advantage(comps, targets, embs)
    assert res_small["structure_advantage_ratio"] == 0.0

    # N >= 3 -> true LOO-CV
    comps_3 = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]])
    targets_3 = np.array([0.001, 0.005, 0.003, 0.004])
    embs_3 = np.random.default_rng(42).normal(size=(4, 8))
    res_loo = model.evaluate_structure_predictive_advantage(comps_3, targets_3, embs_3)
    assert "structure_advantage_ratio" in res_loo
    assert -1.0 <= res_loo["structure_advantage_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# 10. Evidence Events Event-Driven Invariant
# ---------------------------------------------------------------------------
def test_evidence_events_event_driven() -> None:
    engine = AutonomousDiscoveryEngine(seed=42)
    engine.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=42)

    h1_sup_before = engine.hypothesis_engine.hypotheses["H1"].supporting_evidence_count
    h2_sup_before = engine.hypothesis_engine.hypotheses["H2"].supporting_evidence_count

    # Calling refit repeated times without new experiment events MUST NOT increase counters
    engine._refit_models_and_recompute_beliefs()
    engine._refit_models_and_recompute_beliefs()

    assert engine.hypothesis_engine.hypotheses["H1"].supporting_evidence_count == h1_sup_before
    assert engine.hypothesis_engine.hypotheses["H2"].supporting_evidence_count == h2_sup_before


# ---------------------------------------------------------------------------
# 11. Hypothesis Beliefs Sum To One
# ---------------------------------------------------------------------------
def test_hypothesis_beliefs_sum_to_one() -> None:
    hypo_engine = HypothesisEngine(get_default_hypotheses())
    hypo_engine.record_evidence_event(
        event_id="test_ev_1",
        action_type="XRD",
        candidate_id="AUIRH_Au-rich_001",
        structure_residual=0.35,
        structure_novelty=0.45,
    )
    total_belief = sum(h.belief_score for h in hypo_engine.hypotheses.values())
    assert np.isclose(total_belief, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 12. Policy Scoring Normalization
# ---------------------------------------------------------------------------
def test_policy_scoring_normalization(auirh_oracle: AuIrRhMultimodalOracle) -> None:
    engine = AutonomousDiscoveryEngine(oracle=auirh_oracle, seed=42)
    engine.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=42)
    rec, _ = engine.propose_next_experiment()

    assert 0.0 <= rec.scientific_information_value <= 1.000001
    assert 0.0 <= rec.discovery_value <= 1.000001
    assert rec.cost_penalty >= 0.0


# ---------------------------------------------------------------------------
# 13. Policy Modality Ranking Switch
# ---------------------------------------------------------------------------
def test_policy_modality_ranking_switch() -> None:
    engine_cheap_xrd = AutonomousDiscoveryEngine(cost_xrd=0.01, cost_property=20.0, seed=42)
    engine_cheap_xrd.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=42)
    rec_cheap_xrd, _ = engine_cheap_xrd.propose_next_experiment()

    engine_cheap_prop = AutonomousDiscoveryEngine(cost_xrd=20.0, cost_property=0.01, seed=42)
    engine_cheap_prop.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=42)
    rec_cheap_prop, _ = engine_cheap_prop.propose_next_experiment()

    assert rec_cheap_xrd.action.action_type == ExperimentActionType.XRD
    assert rec_cheap_prop.action.action_type == ExperimentActionType.PROPERTY


# ---------------------------------------------------------------------------
# 14. Counterfactual Rationales Generated
# ---------------------------------------------------------------------------
def test_counterfactual_rationales_generated() -> None:
    engine = AutonomousDiscoveryEngine(seed=42)
    engine.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=42)
    rec, _ = engine.propose_next_experiment()
    assert len(rec.alternatives) >= 2
    for alt in rec.alternatives:
        assert len(alt.contrastive_rationale) > 10


# ---------------------------------------------------------------------------
# 15. Agent Perspectives Generated
# ---------------------------------------------------------------------------
def test_agent_perspectives_generated() -> None:
    engine = AutonomousDiscoveryEngine(seed=42)
    engine.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=42)
    _, perspectives = engine.propose_next_experiment()
    role_names = [p.role_name for p in perspectives]
    assert "Hypothesis Scientist" in role_names
    assert "Falsification Scientist" in role_names
    assert "Experiment Designer" in role_names
    assert "Evidence Provenance" in role_names


# ---------------------------------------------------------------------------
# 16. Ledger Persists DB Path on Reset
# ---------------------------------------------------------------------------
def test_ledger_persists_db_path_on_reset(tmp_path: Path) -> None:
    db_file = tmp_path / "persistence_test.db"
    engine = AutonomousDiscoveryEngine(db_path=db_file, seed=42)
    assert str(engine._db_path) == str(db_file)
    engine.reset()
    assert str(engine.ledger.db_path) == str(db_file)


# ---------------------------------------------------------------------------
# 17. Ledger Lifecycle Transitions
# ---------------------------------------------------------------------------
def test_ledger_lifecycle_transitions(tmp_path: Path) -> None:
    db_file = tmp_path / "lifecycle_test.db"
    engine = AutonomousDiscoveryEngine(db_path=db_file, seed=42)
    engine.initialize_curated_scenario(n_init_prop=2, n_init_xrd=2, seed=42)
    records = engine.ledger.list_records()
    assert len(records) == 4
    for r in records:
        assert r.stage.value == "COMPLETED"


# ---------------------------------------------------------------------------
# 18. Candidate Pool Real Dataset Integrity
# ---------------------------------------------------------------------------
def test_candidate_pool_real_dataset_integrity(auirh_oracle: AuIrRhMultimodalOracle) -> None:
    cand_df = auirh_oracle.get_candidate_pool()
    assert len(cand_df) == 966
    for col in ["Au", "Ir", "Rh"]:
        assert (cand_df[col] >= 0.0).all()
        assert (cand_df[col] <= 100.0).all()
    # Composition sums to ~100
    comp_sums = cand_df["Au"] + cand_df["Ir"] + cand_df["Rh"]
    assert np.allclose(comp_sums, 100.0, atol=1.0)
