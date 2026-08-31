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
# Requirement 1 & 2: XRD and k0 Hidden Prior to Action Execution
# ---------------------------------------------------------------------------
def test_oracle_firewall_hides_unobserved_measurements(auirh_oracle: AuIrRhMultimodalOracle) -> None:
    cand_df = auirh_oracle.get_candidate_pool()
    assert len(cand_df) == 966
    # Visible candidate table has ONLY composition features
    assert set(cand_df.columns) == {"candidate_id", "Library", "Area", "Au", "Ir", "Rh"}

    # No targets in observable dataset before execution
    obs_df = auirh_oracle.get_observable_dataset()
    assert obs_df["xrd_observed"].sum() == 0
    assert obs_df["property_observed"].sum() == 0
    assert obs_df["k0"].isna().all()


# ---------------------------------------------------------------------------
# Requirement 3 & 4: Exact Physical Sample Measurement Reveal
# ---------------------------------------------------------------------------
def test_oracle_reveals_exact_physical_measurements(auirh_oracle: AuIrRhMultimodalOracle) -> None:
    cand_id = "AUIRH_Au-rich_001"

    # 1. Execute XRD
    xrd_out = auirh_oracle.execute_xrd(cand_id)
    assert xrd_out.candidate_id == cand_id
    assert xrd_out.action_type == ExperimentActionType.XRD
    assert len(xrd_out.revealed_data["two_theta"]) == 4500
    assert len(xrd_out.revealed_data["intensity"]) == 4500
    assert len(xrd_out.revealed_data["downsampled_two_theta"]) == 450
    assert xrd_out.provenance["library"] == "Au-rich"
    assert xrd_out.provenance["area"] == 1

    # 2. Execute Property
    prop_out = auirh_oracle.execute_property(cand_id)
    assert prop_out.candidate_id == cand_id
    assert prop_out.action_type == ExperimentActionType.PROPERTY
    assert "k0" in prop_out.revealed_data
    assert prop_out.revealed_data["k0"] > 0.0

    # Verify visible dataset reflects reveals
    obs_df = auirh_oracle.get_observable_dataset()
    row = obs_df[obs_df["candidate_id"] == cand_id].iloc[0]
    assert bool(row["xrd_observed"]) is True
    assert bool(row["property_observed"]) is True
    assert not np.isnan(float(row["k0"]))


# ---------------------------------------------------------------------------
# Requirement 5 & 6: Strict Prohibition of Repeated Measurements
# ---------------------------------------------------------------------------
def test_oracle_rejects_duplicate_actions(auirh_oracle: AuIrRhMultimodalOracle) -> None:
    cand_id = "AUIRH_Au-rich_005"

    auirh_oracle.execute_xrd(cand_id)
    with pytest.raises(ValueError, match="XRD characterization already executed"):
        auirh_oracle.execute_xrd(cand_id)

    auirh_oracle.execute_property(cand_id)
    with pytest.raises(ValueError, match="Property measurement already executed"):
        auirh_oracle.execute_property(cand_id)


# ---------------------------------------------------------------------------
# Requirement 7: Candidate Identity Preserved Across Steps
# ---------------------------------------------------------------------------
def test_candidate_identity_preservation(auirh_oracle: AuIrRhMultimodalOracle) -> None:
    cand_pool = auirh_oracle.get_candidate_pool()
    all_cids = cand_pool["candidate_id"].tolist()
    assert len(all_cids) == 966
    assert len(set(all_cids)) == 966
    for cid in all_cids:
        assert cid.startswith("AUIRH_")


# ---------------------------------------------------------------------------
# Requirement 8 & 12: Policy Invariants, Scores, and Absence of Hidden Leakage
# ---------------------------------------------------------------------------
def test_policy_scoring_invariants_and_components(auirh_oracle: AuIrRhMultimodalOracle) -> None:
    engine = AutonomousDiscoveryEngine(oracle=auirh_oracle, seed=42)
    engine.initialize_curated_scenario(n_init_prop=5, n_init_xrd=3, seed=42)

    rec, perspectives = engine.propose_next_experiment()
    assert isinstance(rec, ActionRecommendation)
    assert rec.action.candidate_id.startswith("AUIRH_")
    assert rec.action.action_type in {ExperimentActionType.XRD, ExperimentActionType.PROPERTY}

    # All required score components present
    assert isinstance(rec.total_value, float)
    assert isinstance(rec.scientific_information_value, float)
    assert isinstance(rec.discovery_value, float)
    assert isinstance(rec.cost_penalty, float)
    assert rec.hypothesis_id in {"H1", "H2", "H3"}
    assert len(rec.falsification_criterion) > 10
    assert len(rec.alternatives) >= 2


# ---------------------------------------------------------------------------
# Requirement 9 & 10: Deterministic Scenario & Reset Replay
# ---------------------------------------------------------------------------
def test_deterministic_scenario_and_reset_reproducibility() -> None:
    engine1 = AutonomousDiscoveryEngine(seed=42)
    engine1.initialize_curated_scenario(n_init_prop=5, n_init_xrd=3, seed=42)
    rec1, _ = engine1.propose_next_experiment()

    engine2 = AutonomousDiscoveryEngine(seed=42)
    engine2.initialize_curated_scenario(n_init_prop=5, n_init_xrd=3, seed=42)
    rec2, _ = engine2.propose_next_experiment()

    assert rec1.action.candidate_id == rec2.action.candidate_id
    assert rec1.action.action_type == rec2.action.action_type
    assert np.isclose(rec1.total_value, rec2.total_value)

    # Test Reset
    engine1.reset()
    assert engine1.current_step == 0
    assert engine1.total_budget_spent == 0.0
    assert len(engine1.timeline) == 0


# ---------------------------------------------------------------------------
# Requirement 11: Action Ledger Records and Replay
# ---------------------------------------------------------------------------
def test_action_ledger_records(tmp_path: Path) -> None:
    db_file = tmp_path / "action_test.db"
    oracle = AuIrRhMultimodalOracle()
    engine = AutonomousDiscoveryEngine(oracle=oracle, db_path=db_file, seed=42)
    engine.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=42)

    # 4 property + 2 XRD = 6 records
    records = engine.ledger.list_records()
    assert len(records) == 6
    for r in records:
        assert r.dataset_name == "Au-Ir-Rh_Multimodal_Demo"
        assert r.stage.value == "COMPLETED"


# ---------------------------------------------------------------------------
# Requirement 13: Cost Changes Alter Policy Action Ranking
# ---------------------------------------------------------------------------
def test_cost_sensitivity_alters_policy_ranking() -> None:
    engine_cheap_xrd = AutonomousDiscoveryEngine(cost_xrd=0.1, cost_property=10.0, seed=42)
    engine_cheap_xrd.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=42)
    rec_cheap_xrd, _ = engine_cheap_xrd.propose_next_experiment()

    engine_expensive_xrd = AutonomousDiscoveryEngine(cost_xrd=10.0, cost_property=0.1, seed=42)
    engine_expensive_xrd.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=42)
    rec_expensive_xrd, _ = engine_expensive_xrd.propose_next_experiment()

    assert rec_cheap_xrd.action.action_type == ExperimentActionType.XRD
    assert rec_expensive_xrd.action.action_type == ExperimentActionType.PROPERTY


# ---------------------------------------------------------------------------
# Requirement 14: Hypothesis Evidence Updates Only After Observed Evidence
# ---------------------------------------------------------------------------
def test_hypothesis_evidence_updates_from_observations() -> None:
    hypo_engine = HypothesisEngine(get_default_hypotheses())
    # Prior state is uniform
    assert np.isclose(hypo_engine.hypotheses["H1"].belief_score, 1.0 / 3.0)

    # Simulate strong structural predictive advantage
    beliefs = hypo_engine.update_evidence(
        num_xrd=8,
        num_prop=8,
        structure_advantage_ratio=0.35,
        structure_novelty_mean=0.40,
        structure_residual_norm=0.30,
        property_residual_norm=0.10,
    )
    # H2 (structure-mediated) should rise significantly
    assert beliefs["H2"] > beliefs["H1"]
    assert np.isclose(sum(beliefs.values()), 1.0)


# ---------------------------------------------------------------------------
# Requirement 15 & 16: Multi-Agent Perspectives & Falsification Criteria
# ---------------------------------------------------------------------------
def test_multi_agent_perspectives_and_falsification() -> None:
    engine = AutonomousDiscoveryEngine(seed=42)
    engine.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=42)
    rec, perspectives = engine.propose_next_experiment()

    role_names = [p.role_name for p in perspectives]
    assert "Hypothesis Scientist" in role_names
    assert "Falsification Scientist" in role_names
    assert "Experiment Designer" in role_names
    assert "Evidence Auditor" in role_names

    fals_agent = next(p for p in perspectives if p.role_name == "Falsification Scientist")
    assert len(fals_agent.body) > 20
    assert len(fals_agent.key_points) >= 2


# ---------------------------------------------------------------------------
# Requirement 17: Counterfactual Explanation Decompositions
# ---------------------------------------------------------------------------
def test_counterfactual_explanation_decompositions() -> None:
    engine = AutonomousDiscoveryEngine(seed=42)
    engine.initialize_curated_scenario(n_init_prop=4, n_init_xrd=2, seed=42)
    rec, _ = engine.propose_next_experiment()

    assert len(rec.alternatives) >= 2
    for alt in rec.alternatives:
        assert isinstance(alt.total_value, float)
        assert isinstance(alt.scientific_information_value, float)
        assert isinstance(alt.discovery_value, float)
        assert len(alt.contrastive_rationale) > 10
        # Recommended action score is greater than or equal to alternative
        assert rec.total_value >= alt.total_value


# ---------------------------------------------------------------------------
# Requirement: PCA Leakage Contract (Fitted only on revealed spectra)
# ---------------------------------------------------------------------------
def test_xrd_pca_leakage_contract() -> None:
    extractor = XRDRepresentationExtractor(min_pca_samples=3)

    # 1. Zero or 1 sample -> fallback binning
    spec1 = np.linspace(0, 1, 450)
    extractor.fit([spec1])
    assert not extractor.is_pca_fitted
    emb1 = extractor.transform(spec1)
    assert len(emb1) == 8

    # 2. >= 3 samples -> PCA fits on those 3 samples only
    spec2 = np.sin(np.linspace(0, 3.14, 450))
    spec3 = np.cos(np.linspace(0, 3.14, 450))
    extractor.fit([spec1, spec2, spec3])
    assert extractor.is_pca_fitted
    emb3 = extractor.transform(spec3)
    assert len(emb3) == 8
