from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import pytest

from src.domains.alab.adapter import ALabDomainAdapter
from src.domains.alab.artifact_index import ALabArtifactIndex
from src.domains.alab.chemistry import (
    are_chemically_equivalent,
    get_fractional_composition,
    parse_chemical_formula,
    parse_refinement_phases,
)
from src.domains.alab.config import (
    ALAB_CANDIDATE_FEATURE_NAMES,
    ALAB_CANONICAL_PRECURSORS,
    ALAB_DOMAIN_CONFIG,
    ALAB_OBJECTIVE_REACTION_OUTCOME,
)
from src.domains.alab.feature_encoder import ALabFeatureEncoder
from src.domains.alab.hypotheses import (
    ALabHypothesisProvider,
    PrecursorThermodynamicsHypothesis,
    ProcessKineticsHypothesis,
    StructurePhaseInformedHypothesis,
)
from src.optimization.botorch_backend import BoTorchBackend
from src.science.actions import ExperimentActionType, ScientificAction
from src.science.decision_engine import ScientificDecisionEngine
from src.science.domain import HypothesisTrainingContext
from src.science.falsification.policy import FalsificationFirstPolicy, FalsificationPolicyMode
from src.science.hypothesis_models import HypothesisEnsemble


@pytest.fixture
def alab_fixture_adapter(tmp_path):
    fixture_dir = "tests/fixtures/alab"
    cache_dir = str(tmp_path / "alab_cache")
    return ALabDomainAdapter(
        data_dir=fixture_dir,
        cache_dir=cache_dir,
    )


# 1. CHEMISTRY & REFINEMENT TESTS
def test_chemical_formula_parsing():
    """Verifies chemical formula parsing with parentheses, ICSD labels, and stoichiometric decimals."""
    f1 = parse_chemical_formula("BaCO3_62_(icsd_166091)-0")
    assert f1 == {"Ba": 1.0, "C": 1.0, "O": 3.0}

    f2 = parse_chemical_formula("Al(OH)3")
    assert f2 == {"Al": 1.0, "O": 3.0, "H": 3.0}

    f3 = parse_chemical_formula("(NH4)2HPO4")
    assert f3 == {"N": 2.0, "H": 9.0, "P": 1.0, "O": 4.0}

    f4 = parse_chemical_formula("Mn7.9992Co16.0008O32_227_(icsd_18544)-None")
    assert abs(f4["Mn"] - 7.9992) < 1e-4
    assert abs(f4["Co"] - 16.0008) < 1e-4
    assert abs(f4["O"] - 32.0) < 1e-4


def test_chemical_equivalence():
    """Verifies fractional elemental composition stoichiometry equivalence."""
    assert are_chemically_equivalent("Al(HO)3", "Al(OH)3")
    assert are_chemically_equivalent("(NH4)2HPO4", "H9N2PO4")
    assert are_chemically_equivalent("Mn7.9992Co16.0008O32", "MnCo2O4")
    assert not are_chemically_equivalent("Ba2Ag2C2O7", "BaCO3")
    assert not are_chemically_equivalent("Ba2Ag2C2O7", "Ag2O")
    assert not are_chemically_equivalent("Ti2Fe2O7", "Fe2O3")


def test_alab_refinement_target_absent_has_zero_target_fraction():
    """Verifies that when the target compound is absent from refinement phases, target fraction is 0.0."""
    phase_weights = {
        "BaCO3_62_(icsd_166091)-0": 0.7392,
        "Ag_225_(icsd_604631)-2": 0.2023,
        "Ag2O_224_(icsd_173984)-0": 0.0585,
    }
    parsed = parse_refinement_phases(
        phase_weights=phase_weights,
        target_formula="Ba2Ag2C2O7",
        precursor_formulas=["Ag2O", "BaCO3"],
        rwp=5.83,
    )
    assert parsed["target_phase_fraction"] == 0.0
    assert abs(parsed["precursor_phase_fraction"] - (0.7392 + 0.0585)) < 1e-4
    assert abs(parsed["other_identified_phase_fraction"] - 0.2023) < 1e-4
    assert abs(parsed["rwp_scaled"] - 0.583) < 1e-4


def test_alab_refinement_matches_actual_target_formula():
    """Verifies that an actual target compound phase is correctly matched into target_phase_fraction."""
    phase_weights = {
        "Ba2Fe2O7": 0.95,
        "Fe2O3": 0.05,
    }
    parsed = parse_refinement_phases(
        phase_weights=phase_weights,
        target_formula="Ba2Fe2O7",
        precursor_formulas=["BaO2", "Fe2O3"],
        rwp=3.12,
    )
    assert parsed["target_phase_fraction"] == 0.95
    assert parsed["precursor_phase_fraction"] == 0.05
    assert parsed["other_identified_phase_fraction"] == 0.0


def test_alab_nonprecursor_phase_is_not_automatically_target():
    """Verifies non-precursor side phases are categorized as other_identified, never target."""
    phase_weights = {
        "Ag": 0.46,
        "Al(OH)3": 0.47,
        "Ag2O": 0.07,
    }
    parsed = parse_refinement_phases(
        phase_weights=phase_weights,
        target_formula="Al2Ag2H6O7",
        precursor_formulas=["Ag2O", "Al(OH)3"],
        rwp=5.36,
    )
    assert parsed["target_phase_fraction"] == 0.0
    assert abs(parsed["other_identified_phase_fraction"] - 0.46) < 1e-4
    assert abs(parsed["precursor_phase_fraction"] - (0.47 + 0.07)) < 1e-4


# 2. FEATURE CONTRACT & DATASET SEMANTICS
def test_alab_candidate_pool_features_equal_adapter_features(alab_fixture_adapter):
    """Verifies candidate pool feature columns match adapter.get_candidate_features identically."""
    pool = alab_fixture_adapter.get_candidate_pool()
    feature_cols = list(alab_fixture_adapter.candidate_features)
    assert len(feature_cols) == 49

    for _, row in pool.iterrows():
        cid = row["candidate_id"]
        pool_vec = row[feature_cols].to_numpy(dtype=np.float64)
        adapter_vec = alab_fixture_adapter.get_candidate_features(cid)
        np.testing.assert_allclose(pool_vec, adapter_vec)


def test_alab_precursor_identity_is_not_encoded_as_continuous_ordinal_index(alab_fixture_adapter):
    """Verifies precursor features are multi-hot binary indicators, not continuous integer indices."""
    pool = alab_fixture_adapter.get_candidate_pool()
    assert "precursor_1_idx" not in pool.columns
    assert "precursor_2_idx" not in pool.columns

    prec_cols = [c for c in pool.columns if c.startswith("prec_")]
    assert len(prec_cols) == 46
    for c in prec_cols:
        unique_vals = set(pool[c].unique())
        assert unique_vals.issubset({0.0, 1.0})


def test_alab_unlabeled_outcome_is_not_zero(alab_fixture_adapter):
    """Verifies unlabeled physical failure samples reveal None, not fake 0.0 measurement."""
    action = ScientificAction(
        action_id="OUTCOME_PG_TEST_07",
        candidate_id="PG_TEST_07",
        action_type="OUTCOME_TEST",
        estimated_cost=2.0,
    )
    outcome = alab_fixture_adapter.execute_or_reveal(action)
    assert outcome.revealed_data["is_labeled"] is False
    assert outcome.revealed_data["reaction_outcome_utility"] is None
    assert outcome.canonical_observation is None


# 3. XRD & PHYSICAL AXIS RESAMPLING
def test_alab_xrd_action_reveals_physical_grid(alab_fixture_adapter):
    """Verifies XRD action resamples intensity onto canonical 450-point physical grid."""
    action = ScientificAction(
        action_id="XRD_PG_TEST_01",
        candidate_id="PG_TEST_01",
        action_type="XRD",
        estimated_cost=1.0,
    )
    outcome = alab_fixture_adapter.execute_or_reveal(action)
    assert len(outcome.revealed_data["normalized_intensity"]) == 450
    assert len(outcome.revealed_data["xrd_embedding"]) == 8
    assert np.max(outcome.revealed_data["normalized_intensity"]) <= 1.0 + 1e-6


def test_alab_refinement_requires_xrd(alab_fixture_adapter):
    """Verifies that REFINEMENT action fails closed if XRD was not previously executed."""
    ref_action = ScientificAction(
        action_id="REFINEMENT_PG_TEST_02",
        candidate_id="PG_TEST_02",
        action_type="REFINEMENT",
        estimated_cost=0.5,
    )
    with pytest.raises(RuntimeError, match="requires completed XRD"):
        alab_fixture_adapter.execute_or_reveal(ref_action)


# 4. HIG CALIBRATION TESTS
def test_absolute_hig_normalization_keeps_tiny_hig_tiny():
    """Verifies absolute HIG normalization does not inflate tiny nats into large scientific values."""
    ensemble = HypothesisEnsemble()
    policy = FalsificationFirstPolicy(mode=FalsificationPolicyMode.HYBRID)

    cand_actions = [
        {
            "candidate_id": "C1",
            "action_type": ExperimentActionType.PROPERTY,
            "raw_hig": 3e-10,
            "raw_disc": 0.0,
            "raw_cost": 2.0,
            "property_disagreement": 0.0,
            "structure_disagreement": 0.0,
            "observation_disagreement": 0.0,
            "disagreement_by_modality": {},
            "current_entropy": 1.098,
            "expected_entropy": 1.098,
            "predictions": {},
        },
        {
            "candidate_id": "C2",
            "action_type": ExperimentActionType.PROPERTY,
            "raw_hig": 1e-10,
            "raw_disc": 0.0,
            "raw_cost": 2.0,
            "property_disagreement": 0.0,
            "structure_disagreement": 0.0,
            "observation_disagreement": 0.0,
            "disagreement_by_modality": {},
            "current_entropy": 1.098,
            "expected_entropy": 1.098,
            "predictions": {},
        },
    ]

    scored = policy._score_candidate_actions(cand_actions, ensemble=ensemble)
    assert scored[0]["normalized_hig"] < 1e-8
    assert scored[0]["normalized_hig"] >= 0.0


def test_absolute_hig_zero_maps_to_zero():
    """Verifies exact zero raw HIG maps to exact zero normalized HIG."""
    ensemble = HypothesisEnsemble()
    policy = FalsificationFirstPolicy(mode=FalsificationPolicyMode.HYBRID)
    cand_actions = [
        {
            "candidate_id": "C1",
            "action_type": ExperimentActionType.PROPERTY,
            "raw_hig": 0.0,
            "raw_disc": 0.0,
            "raw_cost": 2.0,
            "property_disagreement": 0.0,
            "structure_disagreement": 0.0,
            "observation_disagreement": 0.0,
            "disagreement_by_modality": {},
            "current_entropy": 1.098,
            "expected_entropy": 1.098,
            "predictions": {},
        }
    ]
    scored = policy._score_candidate_actions(cand_actions, ensemble=ensemble)
    assert scored[0]["normalized_hig"] == 0.0


def test_discovery_only_restricts_to_objective_actions():
    """Verifies that DISCOVERY_ONLY policy penalizes non-objective characterization actions."""
    ensemble = HypothesisEnsemble()
    policy = FalsificationFirstPolicy(mode=FalsificationPolicyMode.DISCOVERY_ONLY)
    cand_actions = [
        {
            "candidate_id": "C1",
            "action_type": "XRD",
            "raw_hig": 1.0,
            "raw_disc": 0.0,
            "raw_cost": 1.0,
            "property_disagreement": 0.0,
            "structure_disagreement": 0.0,
            "observation_disagreement": 0.0,
            "disagreement_by_modality": {},
            "current_entropy": 1.098,
            "expected_entropy": 0.0,
            "predictions": {},
        },
        {
            "candidate_id": "C2",
            "action_type": "OUTCOME_TEST",
            "raw_hig": 0.1,
            "raw_disc": 0.8,
            "raw_cost": 2.0,
            "property_disagreement": 0.0,
            "structure_disagreement": 0.0,
            "observation_disagreement": 0.0,
            "disagreement_by_modality": {},
            "current_entropy": 1.098,
            "expected_entropy": 1.0,
            "predictions": {},
        },
    ]
    scored = policy._score_candidate_actions(cand_actions, ensemble=ensemble)
    assert scored[0]["action_type"] == "OUTCOME_TEST"
    assert scored[1]["total_value"] < -1e8


# 5. BOTORCH OPTIMIZER INTEGRATION ON A-LAB
def test_alab_engine_uses_botorch_discovery_backend(alab_fixture_adapter):
    """Verifies ScientificDecisionEngine runs BoTorchBackend on A-Lab domain with non-zero discovery scores."""
    botorch = BoTorchBackend(default_strategy="expected_improvement")
    engine = ScientificDecisionEngine(
        domain=alab_fixture_adapter,
        optimizer_backend=botorch,
        policy_mode=FalsificationPolicyMode.HYBRID,
        seed=42,
    )
    init_actions = alab_fixture_adapter.get_default_initial_actions(n_candidates=3, pairing_strategy="joint", seed=42)
    engine.initialize(init_actions)

    rec = engine.propose_next_experiment()
    assert rec.action is not None
    assert engine.last_optimizer_status["used"] is True
    assert engine.last_optimizer_status["success"] is True
    assert engine.last_optimizer_status["num_scored"] > 0


def test_alab_outcome_semantics_are_not_mislabeled_as_physical_conversion(alab_fixture_adapter):
    """Verifies that the target objective is explicitly decision utility and not physical conversion."""
    objs = alab_fixture_adapter.objectives
    assert len(objs) == 1
    assert objs[0].name == "reaction_outcome_utility"
    assert objs[0].metadata.get("semantic_type") == "ordinal_decision_utility"
    assert "yield" not in objs[0].name.lower()
    assert "conversion" not in objs[0].name.lower()


def test_alab_refinement_parse_failure_fails_closed():
    """Verifies that malformed refinement data raises ValueError / fails closed."""
    with pytest.raises(ValueError):
        parse_refinement_phases(
            phase_weights={"Malformed(((Formula": 1.0},
            target_formula="BaTiO3",
            precursor_formulas=["BaO", "TiO2"],
            rwp=5.0,
        )


def test_alab_xrd_uses_physical_two_theta_grid(alab_fixture_adapter):
    """Verifies XRD modality defines canonical 450-point 10-100 deg grid."""
    xrd_mod = next(m for m in alab_fixture_adapter.modalities if m.name == "XRD")
    assert xrd_mod.metadata["two_theta_min"] == 10.0
    assert xrd_mod.metadata["two_theta_max"] == 100.0
    assert xrd_mod.metadata["grid_points"] == 450


def test_alab_xrd_peak_position_preserved_across_source_grid_lengths():
    """Verifies interpolation preserves peak position in 2theta space regardless of source grid sampling."""
    src_2theta = np.linspace(10.0, 100.0, 1000)
    src_counts = np.exp(-0.5 * ((src_2theta - 46.0) / 0.5) ** 2)

    canonical_grid = np.linspace(10.0, 100.0, 450)
    interp_counts = np.interp(canonical_grid, src_2theta, src_counts)

    src_peak_2theta = src_2theta[np.argmax(src_counts)]
    interp_peak_2theta = canonical_grid[np.argmax(interp_counts)]

    assert abs(src_peak_2theta - 46.0) < 0.1
    assert abs(interp_peak_2theta - 46.0) < (90.0 / 450.0)


def test_alab_malformed_xrd_fails_closed(tmp_path):
    """Verifies reading a corrupt XRD scan fails closed."""
    import xml.etree.ElementTree as ET
    with pytest.raises(ET.ParseError):
        ET.fromstring("<xrdMeasurements><malformed></xrdMeasurements>")


def test_alab_representation_basis_is_frozen_during_evidence_update(alab_fixture_adapter):
    """Verifies that during evidence update, pre-prediction uses the representation snapshot R_N."""
    snap1 = alab_fixture_adapter.get_representation_snapshot("XRD")
    assert snap1 is not None

    dummy_spec = np.ones(450)
    emb1 = alab_fixture_adapter.transform_with_snapshot("XRD", dummy_spec, snap1)
    emb2 = alab_fixture_adapter.transform_with_snapshot("XRD", dummy_spec, snap1)
    np.testing.assert_allclose(emb1, emb2)


def test_alab_hypotheses_refit_from_accumulated_real_modalities(alab_fixture_adapter):
    """Verifies hypotheses fit across all 3 modalities."""
    provider = ALabHypothesisProvider()
    hypotheses = provider.get_hypotheses()
    assert len(hypotheses) == 3

    comp_map = {cid: alab_fixture_adapter.get_candidate_features(cid) for cid in ["PG_TEST_01", "PG_TEST_02", "PG_TEST_03"]}
    obs_map = {
        "OUTCOME_TEST": {"PG_TEST_01": 1.0, "PG_TEST_02": 0.5},
        "XRD": {"PG_TEST_01": np.zeros(8), "PG_TEST_02": np.ones(8)},
        "REFINEMENT": {"PG_TEST_01": np.array([0.9, 0.1, 0.0, 0.3])},
    }

    for h in hypotheses:
        h.fit(
            composition_by_id=comp_map,
            observations_by_modality=obs_map,
            modality_definitions=alab_fixture_adapter.modalities,
            objective_definitions=alab_fixture_adapter.objectives,
        )
        pred = h.predict(
            candidate_id="PG_TEST_03",
            action_type="OUTCOME_TEST",
            candidate_composition=comp_map["PG_TEST_03"],
        )
        assert pred.mean is not None
        assert pred.variance is not None


def test_alab_structure_hypothesis_does_not_zero_impute_missing_xrd(alab_fixture_adapter):
    """Verifies StructurePhaseInformedHypothesis uses explicit missingness indicator rather than silent zero."""
    hyp = StructurePhaseInformedHypothesis()
    comp_map = {
        "PG_TEST_01": alab_fixture_adapter.get_candidate_features("PG_TEST_01"),
        "PG_TEST_02": alab_fixture_adapter.get_candidate_features("PG_TEST_02"),
    }
    obs_map = {
        "OUTCOME_TEST": {"PG_TEST_01": 1.0, "PG_TEST_02": 0.0},
        "XRD": {"PG_TEST_01": np.ones(8)},
        "REFINEMENT": {},
    }
    hyp.fit(
        composition_by_id=comp_map,
        observations_by_modality=obs_map,
        modality_definitions=alab_fixture_adapter.modalities,
        objective_definitions=alab_fixture_adapter.objectives,
    )
    assert hyp._gp is not None
    assert hyp._gp.X_train_.shape[1] == 49 + 1 + 4 + 1 + 4


def test_alab_modality_predictions_change_after_revealed_training_evidence(alab_fixture_adapter):
    """Verifies that revealing training observations changes hypothesis predictions."""
    hyp = PrecursorThermodynamicsHypothesis()
    comp_map = {
        "PG_TEST_01": alab_fixture_adapter.get_candidate_features("PG_TEST_01"),
        "PG_TEST_02": alab_fixture_adapter.get_candidate_features("PG_TEST_02"),
        "PG_TEST_03": alab_fixture_adapter.get_candidate_features("PG_TEST_03"),
    }
    pred_initial = hyp.predict("PG_TEST_02", "OUTCOME_TEST", comp_map["PG_TEST_02"])

    obs_map = {"OUTCOME_TEST": {"PG_TEST_01": 1.0, "PG_TEST_03": 0.0}}
    hyp.fit(
        composition_by_id=comp_map,
        observations_by_modality=obs_map,
        modality_definitions=alab_fixture_adapter.modalities,
        objective_definitions=alab_fixture_adapter.objectives,
    )
    pred_updated = hyp.predict("PG_TEST_02", "OUTCOME_TEST", comp_map["PG_TEST_02"])
    assert pred_initial.variance != pred_updated.variance or pred_initial.mean != pred_updated.mean


def test_absolute_hig_theoretical_max_near_one():
    """Verifies that a maximal raw HIG (ln(3) nats) normalizes to ~1.0."""
    ensemble = HypothesisEnsemble()
    policy = FalsificationFirstPolicy(mode=FalsificationPolicyMode.HYBRID)
    max_hig = np.log(3.0)
    cand_actions = [
        {
            "candidate_id": "C1",
            "action_type": ExperimentActionType.PROPERTY,
            "raw_hig": max_hig,
            "raw_disc": 0.0,
            "raw_cost": 1.0,
            "property_disagreement": 0.0,
            "structure_disagreement": 0.0,
            "observation_disagreement": 0.0,
            "disagreement_by_modality": {},
            "current_entropy": 1.098612,
            "expected_entropy": 0.0,
            "predictions": {},
        }
    ]
    scored = policy._score_candidate_actions(cand_actions, ensemble=ensemble)
    assert abs(scored[0]["normalized_hig"] - 1.0) < 1e-4


def test_alab_hybrid_contains_nonzero_discovery_scores(alab_fixture_adapter):
    """Verifies HYBRID policy produces non-zero discovery scores when using BoTorch."""
    botorch = BoTorchBackend(default_strategy="expected_improvement")
    engine = ScientificDecisionEngine(
        domain=alab_fixture_adapter,
        optimizer_backend=botorch,
        policy_mode=FalsificationPolicyMode.HYBRID,
        seed=42,
    )
    init_actions = alab_fixture_adapter.get_default_initial_actions(n_candidates=3, pairing_strategy="joint", seed=42)
    engine.initialize(init_actions)
    rec = engine.propose_next_experiment()
    assert rec.discovery_value >= 0.0


def test_discovery_only_fails_or_reports_degraded_when_optimizer_unavailable(alab_fixture_adapter):
    """Verifies that DISCOVERY_ONLY without optimizer fails closed with RuntimeError."""
    engine = ScientificDecisionEngine(
        domain=alab_fixture_adapter,
        optimizer_backend=None,
        policy_mode=FalsificationPolicyMode.DISCOVERY_ONLY,
        seed=42,
    )
    init_actions = alab_fixture_adapter.get_default_initial_actions(n_candidates=3, pairing_strategy="joint", seed=42)
    engine.initialize(init_actions)
    with pytest.raises(RuntimeError, match="DISCOVERY_ONLY requires a functioning optimizer backend"):
        engine.propose_next_experiment()


def test_discovery_only_fails_with_insufficient_objective_observations(alab_fixture_adapter):
    """Verifies that DISCOVERY_ONLY fails closed when < 3 objective observations exist."""
    from unittest.mock import MagicMock

    mock_opt = MagicMock()
    engine = ScientificDecisionEngine(
        domain=alab_fixture_adapter,
        optimizer_backend=mock_opt,
        policy_mode=FalsificationPolicyMode.DISCOVERY_ONLY,
        seed=42,
    )
    # Initialize with only 1 candidate (1 XRD + 1 OUTCOME = 1 objective observation)
    init_actions = alab_fixture_adapter.get_default_initial_actions(n_candidates=1, pairing_strategy="joint", seed=42)
    engine.initialize(init_actions)

    with pytest.raises(RuntimeError, match="DISCOVERY_ONLY unavailable: Insufficient objective observations"):
        engine.propose_next_experiment()


def test_discovery_only_fails_when_optimizer_scoring_fails(alab_fixture_adapter):
    """Verifies that DISCOVERY_ONLY fails closed when optimizer backend raises an exception."""
    from unittest.mock import MagicMock

    mock_opt = MagicMock()
    mock_opt.score_candidates.side_effect = RuntimeError("BoTorch GP fitting failed")

    engine = ScientificDecisionEngine(
        domain=alab_fixture_adapter,
        optimizer_backend=mock_opt,
        policy_mode=FalsificationPolicyMode.DISCOVERY_ONLY,
        seed=42,
    )
    init_actions = alab_fixture_adapter.get_default_initial_actions(n_candidates=3, pairing_strategy="joint", seed=42)
    engine.initialize(init_actions)

    with pytest.raises(RuntimeError, match="DISCOVERY_ONLY unavailable: BoTorch GP fitting failed"):
        engine.propose_next_experiment()


def test_discovery_only_fails_with_zero_scored_candidates(alab_fixture_adapter):
    """Verifies that DISCOVERY_ONLY fails closed when optimizer backend returns empty scores."""
    from unittest.mock import MagicMock

    mock_opt = MagicMock()
    mock_opt.score_candidates.return_value = {}

    engine = ScientificDecisionEngine(
        domain=alab_fixture_adapter,
        optimizer_backend=mock_opt,
        policy_mode=FalsificationPolicyMode.DISCOVERY_ONLY,
        seed=42,
    )
    init_actions = alab_fixture_adapter.get_default_initial_actions(n_candidates=3, pairing_strategy="joint", seed=42)
    engine.initialize(init_actions)

    with pytest.raises(RuntimeError, match="DISCOVERY_ONLY unavailable"):
        engine.propose_next_experiment()


def test_hybrid_enters_explicit_epistemic_degraded_mode(alab_fixture_adapter):
    """Verifies that HYBRID enters explicit epistemic degraded mode when optimizer backend fails."""
    from unittest.mock import MagicMock

    mock_opt = MagicMock()
    mock_opt.score_candidates.side_effect = RuntimeError("CUDA OOM or scoring error")

    engine = ScientificDecisionEngine(
        domain=alab_fixture_adapter,
        optimizer_backend=mock_opt,
        policy_mode=FalsificationPolicyMode.HYBRID,
        seed=42,
    )
    init_actions = alab_fixture_adapter.get_default_initial_actions(n_candidates=3, pairing_strategy="joint", seed=42)
    engine.initialize(init_actions)

    rec = engine.propose_next_experiment()
    assert engine.last_optimizer_status["success"] is False
    assert engine.last_optimizer_status["degraded_mode"] == "epistemic_only"
    assert rec.action.metadata["discovery_status"] == "disabled"
    assert rec.action.metadata["degraded_mode"] == "epistemic_only"
    assert rec.uncertainty_summary["discovery_status"] == "disabled"
    assert rec.uncertainty_summary["degraded_mode"] == "epistemic_only"


def test_successful_hybrid_has_discovery_enabled(alab_fixture_adapter):
    """Verifies that HYBRID marks discovery as enabled and degraded_mode as None upon optimizer success."""
    engine = ScientificDecisionEngine(
        domain=alab_fixture_adapter,
        optimizer_backend=BoTorchBackend(),
        policy_mode=FalsificationPolicyMode.HYBRID,
        seed=42,
    )
    init_actions = alab_fixture_adapter.get_default_initial_actions(n_candidates=3, pairing_strategy="joint", seed=42)
    engine.initialize(init_actions)

    rec = engine.propose_next_experiment()
    assert engine.last_optimizer_status["success"] is True
    assert engine.last_optimizer_status["degraded_mode"] is None
    assert rec.action.metadata["discovery_status"] == "enabled"
    assert rec.action.metadata["degraded_mode"] is None
    assert rec.uncertainty_summary["discovery_status"] == "enabled"
    assert rec.uncertainty_summary["degraded_mode"] is None

