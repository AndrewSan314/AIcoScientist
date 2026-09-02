from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domains.alab.canonical import (
    get_canonical_refinement_case,
    get_canonical_scan,
    normalize_phase_weights,
)
from src.domains.alab.chemistry import parse_refinement_phases
from src.domains.alab.hypotheses import (
    DEFAULT_BROAD_REF_MEAN,
    DEFAULT_BROAD_REF_VAR,
    DEFAULT_BROAD_XRD_MEAN,
    DEFAULT_BROAD_XRD_VAR,
    PrecursorThermodynamicsHypothesis,
    ProcessKineticsHypothesis,
    StructurePhaseInformedHypothesis,
)
from src.science.actions import ExperimentActionType, ExperimentOutcome, ScientificAction
from src.science.decision_engine import ScientificDecisionEngine
from src.science.domain import (
    HypothesisTrainingContext,
    MaterialDomainAdapter,
    MaterialDomainConfig,
    ModalityDefinition,
    ObjectiveDefinition,
    ObjectiveDirection,
)
from src.science.falsification.policy import (
    FalsificationFirstPolicy,
    FalsificationPolicyMode,
)
from src.science.hypothesis_models import HypothesisEnsemble


def test_canonical_scan_selection():
    # 1. Active scan index explicitly specified
    sample1 = {
        "active_scan_index": 1,
        "characterization": {
            "xrd": {
                "scans": [
                    {"filename": "scan0.xrdml", "status": "draft"},
                    {"filename": "scan1.xrdml", "status": "valid"},
                ]
            }
        },
    }
    scan, idx, method = get_canonical_scan(sample1)
    assert idx == 1
    assert scan["filename"] == "scan1.xrdml"
    assert method == "ledger_active_scan_index"

    # 2. Fallback to status valid / active
    sample2 = {
        "active_scan_index": None,
        "characterization": {
            "xrd": {
                "scans": [
                    {"filename": "scan0.xrdml", "status": "aborted"},
                    {"filename": "scan1.xrdml", "status": "valid"},
                ]
            }
        },
    }
    scan, idx, method = get_canonical_scan(sample2)
    assert idx == 1
    assert method == "status_active_or_valid"

    # 3. Fallback to scan with refinement cases
    sample3 = {
        "active_scan_index": None,
        "characterization": {
            "xrd": {
                "scans": [
                    {"filename": "scan0.xrdml"},
                    {"filename": "scan1.xrdml", "refinement_cases": [{"rank": 1}]},
                ]
            }
        },
    }
    scan, idx, method = get_canonical_scan(sample3)
    assert idx == 1
    assert method == "has_refinement_cases"

    # 4. Fallback to first scan
    sample4 = {
        "characterization": {
            "xrd": {
                "scans": [
                    {"filename": "scan0.xrdml"},
                    {"filename": "scan1.xrdml"},
                ]
            }
        }
    }
    scan, idx, method = get_canonical_scan(sample4)
    assert idx == 0
    assert method == "fallback_first_scan"


def test_canonical_refinement_case_selection():
    # 1. Active case index explicitly specified
    scan1 = {
        "active_case_index": 1,
        "refinement_cases": [
            {"rank": 1, "rwp": 0.12},
            {"rank": 2, "rwp": 0.18},
        ],
    }
    case, idx, method = get_canonical_refinement_case(scan1)
    assert idx == 1
    assert case["rank"] == 2
    assert method == "ledger_active_case_index"

    # 2. Manual case preferred over automated case
    scan2 = {
        "active_case_index": None,
        "refinement_cases": [
            {"rank": 1, "rwp": 0.08, "origin": "automated_db"},
            {"rank": -1, "rwp": 0.14, "origin": "manual"},
        ],
    }
    case, idx, method = get_canonical_refinement_case(scan2)
    assert idx == 1
    assert case["rank"] == -1
    assert method == "fallback_manual_preferred"

    # 3. Human accepted case preferred
    scan3 = {
        "active_case_index": None,
        "refinement_cases": [
            {"rank": 1, "rwp": 0.08, "verification": {"is_accepted": False}},
            {"rank": 2, "rwp": 0.12, "verification": {"is_accepted": True, "human_quality_score": 1}},
        ],
    }
    case, idx, method = get_canonical_refinement_case(scan3)
    assert idx == 1
    assert case["rank"] == 2
    assert method == "fallback_human_accepted"

    # 4. Lowest Rwp tie-break
    scan4 = {
        "active_case_index": None,
        "refinement_cases": [
            {"rank": 1, "rwp": 0.15},
            {"rank": 2, "rwp": 0.07},
        ],
    }
    case, idx, method = get_canonical_refinement_case(scan4)
    assert idx == 1
    assert case["rwp"] == 0.07


def test_normalize_phase_weights():
    # Percentage input
    pct_weights = {"Li2O": 45.0, "MnO2": 55.0}
    norm_w, residual, unit = normalize_phase_weights(pct_weights)
    assert unit == "percentage"
    assert pytest.approx(norm_w["Li2O"], rel=1e-4) == 0.45
    assert pytest.approx(norm_w["MnO2"], rel=1e-4) == 0.55
    assert pytest.approx(residual, rel=1e-4) == 0.0

    # Fraction input
    frac_weights = {"Li2O": 0.40, "MnO2": 0.50}
    norm_w2, residual2, unit2 = normalize_phase_weights(frac_weights)
    assert unit2 == "fraction"
    assert pytest.approx(norm_w2["Li2O"], rel=1e-4) == 0.40
    assert pytest.approx(residual2, rel=1e-4) == 0.10

    # parse_refinement_phases integration
    parsed = parse_refinement_phases(pct_weights, "Li2O", ["MnO2"], rwp=7.5)
    assert pytest.approx(parsed["target_phase_fraction"], rel=1e-4) == 0.45
    assert pytest.approx(parsed["precursor_phase_fraction"], rel=1e-4) == 0.55
    assert parsed["phase_weight_unit"] == "percentage"


@pytest.fixture
def alab_fixture_adapter(tmp_path):
    from src.domains.alab.adapter import ALabDomainAdapter
    fixture_dir = "tests/fixtures/alab"
    cache_dir = str(tmp_path / "alab_cache")
    return ALabDomainAdapter(
        data_dir=fixture_dir,
        cache_dir=cache_dir,
    )


def test_unclassified_candidate_has_no_outcome_action(alab_fixture_adapter):
    actions = alab_fixture_adapter.list_valid_actions()
    # Ensure unclassified samples do not have OUTCOME actions
    for act in actions:
        if act.action_type == "OUTCOME_TEST":
            assert alab_fixture_adapter.has_revealable_outcome(act.candidate_id)


def test_unclassified_candidate_execute_fails_closed(alab_fixture_adapter):
    # If an unclassified candidate outcome is executed by ScientificDecisionEngine, it must fail closed
    unclassified_cid = None
    for cid, sample in alab_fixture_adapter._samples_by_id.items():
        outcome = sample.get("outcome") or {}
        if outcome.get("reaction_category") is None:
            unclassified_cid = cid
            break

    if unclassified_cid is not None:
        act = ScientificAction(
            action_id=f"OUTCOME_{unclassified_cid}",
            candidate_id=unclassified_cid,
            action_type="OUTCOME_TEST",
            estimated_cost=2.0,
            metadata={"modality_hint": "OUTCOME_TEST"},
        )
        from src.science.falsification.policy import ActionRecommendation
        engine = ScientificDecisionEngine(
            domain=alab_fixture_adapter,
            ensemble=HypothesisEnsemble({"h_simple": SimpleObjectiveHypothesis()}),
            policy_mode=FalsificationPolicyMode.PURE_FALSIFICATION,
        )
        rec = ActionRecommendation(
            action=act,
            total_value=1.0,
            scientific_information_value=0.5,
            discovery_value=0.5,
            cost_penalty=0.1,
            hypothesis_id="h_simple",
            rationale="test",
            falsification_criterion="test criterion",
        )
        with pytest.raises(RuntimeError, match="unexpectedly returned None"):
            engine.execute_recommendation(rec)


class SimpleObjectiveHypothesis:
    def __init__(self, hypothesis_id: str = "h_simple"):
        self._id = hypothesis_id
        self.hypothesis_id = hypothesis_id

    def supports_action(self, a):
        return True

    def fit_context(self, context):
        return self

    def fit(self, *args, **kwargs):
        return self

    def predict_observation(self, candidate_id, action_type, **kwargs):
        from src.science.hypothesis_models import PredictiveDistribution
        return PredictiveDistribution(
            hypothesis_id=self._id,
            candidate_id=candidate_id,
            action_type=action_type,
            mean=np.array([0.5], dtype=np.float64),
            variance=np.array([0.1], dtype=np.float64),
        )

    def log_predictive_density(self, observation=None, prediction=None, **kwargs):
        return 0.0

    def falsification_summary(self, *args, **kwargs):
        return "mock criterion"


def test_missing_objective_not_written_as_zero_to_ledger():
    # Mock adapter that returns non-numeric data for an objective
    class MockAdapter(MaterialDomainAdapter):
        @property
        def domain_id(self) -> str:
            return "mock_domain"

        def get_config(self) -> MaterialDomainConfig:
            return MaterialDomainConfig(
                domain_id="mock_domain",
                candidate_features=("feat1",),
                objectives=(ObjectiveDefinition(name="yield", direction=ObjectiveDirection.MAXIMIZE, units="%"),),
                modalities=(ModalityDefinition(name="OUTCOME_TEST", observation_kind="objective", cost=1.0),),
            )

        def get_modality_schema(self):
            return self.get_config().modalities

        def get_objectives(self):
            return self.get_config().objectives

        def get_candidate_pool(self) -> pd.DataFrame:
            return pd.DataFrame([{"candidate_id": "c1", "feat1": 1.0}])

        def get_candidate_features(self, candidate_id: str) -> dict[str, float]:
            return {"feat1": 1.0}

        def list_valid_actions(self) -> list[ScientificAction]:
            return [ScientificAction(action_id="act_1", candidate_id="c1", action_type="OUTCOME_TEST", estimated_cost=1.0)]

        def execute_or_reveal(self, action: ScientificAction) -> ExperimentOutcome:
            return ExperimentOutcome(
                action_id=action.action_id,
                candidate_id=action.candidate_id,
                action_type=action.action_type,
                revealed_data={"yield": None},
                canonical_observation=None,
            )

    adapter = MockAdapter()
    ensemble = HypothesisEnsemble({"h_simple": SimpleObjectiveHypothesis()})
    engine = ScientificDecisionEngine(domain=adapter, ensemble=ensemble)
    # Baseline initialization with non-numeric / None objective observation
    baseline_act = ScientificAction(action_id="base_1", candidate_id="c1", action_type="OUTCOME_TEST", estimated_cost=1.0)
    engine.initialize(initial_actions=[baseline_act])

    # Verify record does NOT have performance['yield'] = 0.0
    rec = engine.recorded_experiments[0]
    assert "yield" not in rec.performance


def test_objective_action_returning_none_fails_closed():
    class MockNoneAdapter(MaterialDomainAdapter):
        @property
        def domain_id(self) -> str:
            return "mock_none"

        def get_config(self) -> MaterialDomainConfig:
            return MaterialDomainConfig(
                domain_id="mock_none",
                candidate_features=("feat1",),
                objectives=(ObjectiveDefinition(name="yield", direction=ObjectiveDirection.MAXIMIZE, units="%"),),
                modalities=(ModalityDefinition(name="OUTCOME_TEST", observation_kind="objective", cost=1.0),),
            )

        def get_modality_schema(self):
            return self.get_config().modalities

        def get_objectives(self):
            return self.get_config().objectives

        def get_candidate_pool(self) -> pd.DataFrame:
            return pd.DataFrame([{"candidate_id": "c1", "feat1": 1.0}])

        def get_candidate_features(self, candidate_id: str) -> dict[str, float]:
            return {"feat1": 1.0}

        def list_valid_actions(self) -> list[ScientificAction]:
            return [ScientificAction(action_id="act_1", candidate_id="c1", action_type="OUTCOME_TEST", estimated_cost=1.0)]

        def execute_or_reveal(self, action: ScientificAction) -> ExperimentOutcome:
            return ExperimentOutcome(
                action_id=action.action_id,
                candidate_id=action.candidate_id,
                action_type=action.action_type,
                revealed_data={"yield": None},
                canonical_observation=None,
            )

    adapter = MockNoneAdapter()
    ensemble = HypothesisEnsemble({"h_simple": SimpleObjectiveHypothesis()})
    engine = ScientificDecisionEngine(domain=adapter, ensemble=ensemble, policy_mode=FalsificationPolicyMode.PURE_FALSIFICATION)
    rec = engine.propose_next_experiment()
    with pytest.raises(RuntimeError, match="unexpectedly returned None"):
        engine.execute_recommendation(rec)


def test_discovery_only_requires_optimizer():
    class MockDummyAdapter(MaterialDomainAdapter):
        @property
        def domain_id(self) -> str:
            return "mock_dummy"

        def get_config(self) -> MaterialDomainConfig:
            return MaterialDomainConfig(
                domain_id="mock_dummy",
                candidate_features=("feat1",),
                objectives=(ObjectiveDefinition(name="yield", direction=ObjectiveDirection.MAXIMIZE, units="%"),),
                modalities=(ModalityDefinition(name="OUTCOME_TEST", observation_kind="objective", cost=1.0),),
            )

        def get_modality_schema(self):
            return self.get_config().modalities

        def get_objectives(self):
            return self.get_config().objectives

        def get_candidate_pool(self) -> pd.DataFrame:
            return pd.DataFrame([{"candidate_id": "c1", "feat1": 1.0}])

        def get_candidate_features(self, candidate_id: str) -> dict[str, float]:
            return {"feat1": 1.0}

        def list_valid_actions(self) -> list[ScientificAction]:
            return [ScientificAction(action_id="act_1", candidate_id="c1", action_type="OUTCOME_TEST", estimated_cost=1.0)]

        def execute_or_reveal(self, action: ScientificAction) -> ExperimentOutcome:
            return ExperimentOutcome(action_id=action.action_id, candidate_id=action.candidate_id, action_type=action.action_type, revealed_data={"yield": 0.5}, canonical_observation=0.5)

    adapter = MockDummyAdapter()
    engine = ScientificDecisionEngine(domain=adapter, policy_mode=FalsificationPolicyMode.DISCOVERY_ONLY, optimizer_backend=None)
    with pytest.raises(RuntimeError, match="DISCOVERY_ONLY requires a functioning optimizer backend"):
        engine.propose_next_experiment()


def test_alab_hypotheses_no_handcrafted_priors():
    h_thermo = PrecursorThermodynamicsHypothesis()
    h_kin = ProcessKineticsHypothesis()
    h_struct = StructurePhaseInformedHypothesis()

    comp = np.array([0.1, 0.8, 0.5] + [0.0] * 46)

    # Untrained XRD predictive distribution
    pred_xrd_thermo = h_thermo.predict_observation("c1", "XRD", composition=comp)
    pred_xrd_kin = h_kin.predict_observation("c1", "XRD", composition=comp)
    pred_xrd_struct = h_struct.predict_observation("c1", "XRD", composition=comp)

    # Verify they output identical broad priors (HIG == 0.0)
    np.testing.assert_allclose(pred_xrd_thermo.mean, DEFAULT_BROAD_XRD_MEAN)
    np.testing.assert_allclose(pred_xrd_kin.mean, DEFAULT_BROAD_XRD_MEAN)
    np.testing.assert_allclose(pred_xrd_struct.mean, DEFAULT_BROAD_XRD_MEAN)
    np.testing.assert_allclose(pred_xrd_thermo.variance, DEFAULT_BROAD_XRD_VAR)
    np.testing.assert_allclose(pred_xrd_kin.variance, DEFAULT_BROAD_XRD_VAR)
    np.testing.assert_allclose(pred_xrd_struct.variance, DEFAULT_BROAD_XRD_VAR)

    # Untrained Refinement predictive distribution
    pred_ref_thermo = h_thermo.predict_observation("c1", "REFINEMENT", composition=comp)
    pred_ref_kin = h_kin.predict_observation("c1", "REFINEMENT", composition=comp)
    pred_ref_struct = h_struct.predict_observation("c1", "REFINEMENT", composition=comp)

    np.testing.assert_allclose(pred_ref_thermo.mean, DEFAULT_BROAD_REF_MEAN)
    np.testing.assert_allclose(pred_ref_kin.mean, DEFAULT_BROAD_REF_MEAN)
    np.testing.assert_allclose(pred_ref_struct.mean, DEFAULT_BROAD_REF_MEAN)


def test_characterization_surrogates_learn_from_evidence():
    h_thermo = PrecursorThermodynamicsHypothesis()
    h_kin = ProcessKineticsHypothesis()

    # Synthetic observations for 4 candidates
    c_feats = {
        "c1": np.array([0.1, 0.2, 0.2] + [1.0] + [0.0] * 45),
        "c2": np.array([0.2, 0.8, 0.5] + [1.0] + [0.0] * 45),
        "c3": np.array([0.3, 0.5, 0.3] + [0.0] * 46),
        "c4": np.array([0.4, 0.9, 0.8] + [0.0] * 46),
    }
    xrd_obs = {
        "c1": np.array([0.1] * 8),
        "c2": np.array([0.5] * 8),
        "c3": np.array([0.9] * 8),
        "c4": np.array([0.3] * 8),
    }

    ctx = HypothesisTrainingContext(
        candidate_features_by_id=c_feats,
        observations_by_modality={"XRD": xrd_obs},
        modality_definitions={},
        objective_definitions={},
    )
    h_thermo.fit_context(ctx)
    h_kin.fit_context(ctx)

    assert h_thermo._xrd_surrogate is not None
    assert h_kin._xrd_surrogate is not None

    # Predictions now differ from default broad prior and adapt to candidate features
    pred_thermo = h_thermo.predict_observation("c1", "XRD", composition=c_feats["c1"])
    pred_kin = h_kin.predict_observation("c1", "XRD", composition=c_feats["c1"])

    assert pred_thermo.metadata["fitted"] is True
    assert pred_kin.metadata["fitted"] is True
    assert not np.allclose(pred_thermo.mean, DEFAULT_BROAD_XRD_MEAN)
    assert not np.allclose(pred_kin.mean, DEFAULT_BROAD_XRD_MEAN)


def test_absolute_hig_independent_of_candidate_pool():
    policy = FalsificationFirstPolicy(mode=FalsificationPolicyMode.HYBRID)
    ensemble = HypothesisEnsemble(hypotheses={
        "h1": PrecursorThermodynamicsHypothesis(),
        "h2": ProcessKineticsHypothesis(),
        "h3": StructurePhaseInformedHypothesis(),
    })

    # Pool with Candidate 1 and Candidate 2
    actions_pool = [
        {"candidate_id": "c1", "action_type": ExperimentActionType.PROPERTY, "raw_hig": 0.45, "raw_disc": 0.6, "raw_cost": 1.0, "current_entropy": 1.098},
        {"candidate_id": "c2", "action_type": ExperimentActionType.PROPERTY, "raw_hig": 0.20, "raw_disc": 0.3, "raw_cost": 1.0, "current_entropy": 1.098},
    ]
    scored_pool = policy._score_candidate_actions(actions_pool, ensemble=ensemble)
    c1_abs_hig_pool = scored_pool[0]["absolute_hig_normalized"]

    # Candidate 1 isolated in pool
    actions_single = [
        {"candidate_id": "c1", "action_type": ExperimentActionType.PROPERTY, "raw_hig": 0.45, "raw_disc": 0.6, "raw_cost": 1.0, "current_entropy": 1.098},
    ]
    scored_single = policy._score_candidate_actions(actions_single, ensemble=ensemble)
    c1_abs_hig_single = scored_single[0]["absolute_hig_normalized"]

    # Invariance check: absolute HIG of c1 does NOT change when c2 is removed
    assert pytest.approx(c1_abs_hig_pool, abs=1e-9) == c1_abs_hig_single
    assert pytest.approx(c1_abs_hig_single, rel=1e-4) == 0.45 / np.log(3.0)
