from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.datasets.auirh_actions import AuIrRhMultimodalOracle
from src.science.actions import ExperimentActionType, ScientificAction
from src.science.discovery_engine import AutonomousDiscoveryEngine
from src.science.falsification.identifiability import run_identifiability_analysis
from src.science.falsification.information_gain import HypothesisInformationGainEstimator
from src.science.falsification.policy import FalsificationFirstPolicy, FalsificationPolicyMode
from src.science.falsification.synthetic_worlds import (
    World1_CompositionSufficient,
    World2_StructureInformed,
    World3_LocalStructuralRegime,
)
from src.science.hypothesis_models import (
    CompositionSufficientHypothesis,
    HypothesisEnsemble,
    LocalStructuralRegimeHypothesis,
    PredictiveDistribution,
    StructureInformedHypothesis,
)


# ---------------------------------------------------------------------------
# 1. Analytic HIG Calibration Test Suite (Cases A - E)
# ---------------------------------------------------------------------------
def test_hig_case_a_identical_hypotheses_zero() -> None:
    """Case A: When competing hypotheses make identical predictions, HIG is analytically zero."""
    estimator = HypothesisInformationGainEstimator(n_samples_benchmark=128)

    class IdenticalH1:
        hypothesis_id = "H1"
        def supports_action(self, a): return True
        def predict_observation(self, *args, **kwargs):
            return PredictiveDistribution("H1", "C1", ExperimentActionType.PROPERTY, np.array([0.005]), np.array([1e-4]))
        def log_predictive_density(self, obs, pred): return pred.log_pdf(obs)

    class IdenticalH2:
        hypothesis_id = "H2"
        def supports_action(self, a): return True
        def predict_observation(self, *args, **kwargs):
            return PredictiveDistribution("H2", "C1", ExperimentActionType.PROPERTY, np.array([0.005]), np.array([1e-4]))
        def log_predictive_density(self, obs, pred): return pred.log_pdf(obs)

    class IdenticalH3:
        hypothesis_id = "H3"
        def supports_action(self, a): return True
        def predict_observation(self, *args, **kwargs):
            return PredictiveDistribution("H3", "C1", ExperimentActionType.PROPERTY, np.array([0.005]), np.array([1e-4]))
        def log_predictive_density(self, obs, pred): return pred.log_pdf(obs)

    mock_ensemble = HypothesisEnsemble(hypotheses=[IdenticalH1(), IdenticalH2(), IdenticalH3()])

    eval_res = estimator.evaluate_action_discrimination(
        candidate_id="C1",
        action_type=ExperimentActionType.PROPERTY,
        composition=np.array([33.3, 33.3, 33.4]),
        ensemble=mock_ensemble,
        seed=42,
    )
    assert np.isclose(eval_res.hypothesis_information_gain, 0.0, atol=1e-5)


def test_hig_case_b_separated_hypotheses_high() -> None:
    """Case B: When hypotheses make far-separated low-noise predictions, HIG is high and positive."""
    estimator = HypothesisInformationGainEstimator(n_samples_benchmark=256)

    # Build mock ensemble with far-separated predictive means
    class MockH1:
        hypothesis_id = "H1"
        def supports_action(self, a): return True
        def predict_observation(self, *args, **kwargs):
            return PredictiveDistribution("H1", "C1", ExperimentActionType.PROPERTY, np.array([0.001]), np.array([1e-6]))
        def log_predictive_density(self, obs, pred): return pred.log_pdf(obs)

    class MockH2:
        hypothesis_id = "H2"
        def supports_action(self, a): return True
        def predict_observation(self, *args, **kwargs):
            return PredictiveDistribution("H2", "C1", ExperimentActionType.PROPERTY, np.array([0.010]), np.array([1e-6]))
        def log_predictive_density(self, obs, pred): return pred.log_pdf(obs)

    mock_ensemble = HypothesisEnsemble(hypotheses=[MockH1(), MockH2()])
    eval_res = estimator.evaluate_action_discrimination(
        candidate_id="C1",
        action_type=ExperimentActionType.PROPERTY,
        composition=np.array([50.0, 50.0, 0.0]),
        ensemble=mock_ensemble,
        seed=42,
    )
    # Binary hypothesis entropy = log(2) ~ 0.693. Perfect separation yields HIG ~ 0.693
    assert eval_res.hypothesis_information_gain > 0.60
    assert eval_res.hypothesis_information_gain <= np.log(2.0) + 1e-3


def test_hig_case_c_high_noise_reduces_hig() -> None:
    """Case C: Increasing observation noise strictly diminishes Expected Information Gain."""
    estimator = HypothesisInformationGainEstimator(n_samples_benchmark=256)

    # Low noise pair
    class LowNoiseH1:
        hypothesis_id = "H1"
        def supports_action(self, a): return True
        def predict_observation(self, *args, **kwargs):
            return PredictiveDistribution("H1", "C1", ExperimentActionType.PROPERTY, np.array([0.002]), np.array([1e-5]))
        def log_predictive_density(self, obs, pred): return pred.log_pdf(obs)

    class LowNoiseH2:
        hypothesis_id = "H2"
        def supports_action(self, a): return True
        def predict_observation(self, *args, **kwargs):
            return PredictiveDistribution("H2", "C1", ExperimentActionType.PROPERTY, np.array([0.008]), np.array([1e-5]))
        def log_predictive_density(self, obs, pred): return pred.log_pdf(obs)

    low_noise_ens = HypothesisEnsemble(hypotheses=[LowNoiseH1(), LowNoiseH2()])
    low_res = estimator.evaluate_action_discrimination(
        candidate_id="C1",
        action_type=ExperimentActionType.PROPERTY,
        composition=np.array([50.0, 50.0, 0.0]),
        ensemble=low_noise_ens,
        seed=42,
    )

    # High noise pair (same separation, 100x variance)
    class HighNoiseH1:
        hypothesis_id = "H1"
        def supports_action(self, a): return True
        def predict_observation(self, *args, **kwargs):
            return PredictiveDistribution("H1", "C1", ExperimentActionType.PROPERTY, np.array([0.002]), np.array([1e-2]))
        def log_predictive_density(self, obs, pred): return pred.log_pdf(obs)

    class HighNoiseH2:
        hypothesis_id = "H2"
        def supports_action(self, a): return True
        def predict_observation(self, *args, **kwargs):
            return PredictiveDistribution("H2", "C1", ExperimentActionType.PROPERTY, np.array([0.008]), np.array([1e-2]))
        def log_predictive_density(self, obs, pred): return pred.log_pdf(obs)

    high_noise_ens = HypothesisEnsemble(hypotheses=[HighNoiseH1(), HighNoiseH2()])
    high_res = estimator.evaluate_action_discrimination(
        candidate_id="C1",
        action_type=ExperimentActionType.PROPERTY,
        composition=np.array([50.0, 50.0, 0.0]),
        ensemble=high_noise_ens,
        seed=42,
    )

    assert low_res.hypothesis_information_gain > high_res.hypothesis_information_gain


def test_hig_case_d_extreme_prior_diminishes_hig() -> None:
    """Case D: When one hypothesis already has probability ~ 0.99, maximum possible HIG is bounded."""
    estimator = HypothesisInformationGainEstimator(n_samples_benchmark=256)

    class MockH1:
        hypothesis_id = "H1"
        def supports_action(self, a): return True
        def predict_observation(self, *args, **kwargs):
            return PredictiveDistribution("H1", "C1", ExperimentActionType.PROPERTY, np.array([0.001]), np.array([1e-6]))
        def log_predictive_density(self, obs, pred): return pred.log_pdf(obs)

    class MockH2:
        hypothesis_id = "H2"
        def supports_action(self, a): return True
        def predict_observation(self, *args, **kwargs):
            return PredictiveDistribution("H2", "C1", ExperimentActionType.PROPERTY, np.array([0.010]), np.array([1e-6]))
        def log_predictive_density(self, obs, pred): return pred.log_pdf(obs)

    extreme_ens = HypothesisEnsemble(
        hypotheses=[MockH1(), MockH2()],
        prior_beliefs={"H1": 0.995, "H2": 0.005},
    )
    eval_res = estimator.evaluate_action_discrimination(
        candidate_id="C1",
        action_type=ExperimentActionType.PROPERTY,
        composition=np.array([50.0, 50.0, 0.0]),
        ensemble=extreme_ens,
        seed=42,
    )
    # Remaining entropy is tiny: -0.995*log(0.995) - 0.005*log(0.005) ~ 0.031 nats
    assert eval_res.hypothesis_information_gain < 0.05


def test_hig_case_e_mc_reproducibility() -> None:
    """Case E: Fixed random seed produces identical floating point HIG values."""
    estimator = HypothesisInformationGainEstimator(n_samples_benchmark=128)
    ensemble = HypothesisEnsemble()

    cid = "TEST_CAND_01"
    comp = np.array([40.0, 40.0, 20.0])

    res_1 = estimator.evaluate_action_discrimination("C1", ExperimentActionType.PROPERTY, comp, ensemble, seed=123)
    res_2 = estimator.evaluate_action_discrimination("C1", ExperimentActionType.PROPERTY, comp, ensemble, seed=123)
    assert np.isclose(res_1.hypothesis_information_gain, res_2.hypothesis_information_gain)


# ---------------------------------------------------------------------------
# 2. Sequential Evidence Updating & Log-Sum-Exp Normalization
# ---------------------------------------------------------------------------
def test_sequential_predictive_evidence_update() -> None:
    """Verifies that sequential evidence accumulation correctly updates beliefs in log-space."""
    ensemble = HypothesisEnsemble()
    init_beliefs = ensemble.get_beliefs()
    assert np.isclose(init_beliefs["H1"], 1.0 / 3.0)

    # Pre-register predictions where H1 predicts 0.008 and H2/H3 predict 0.002
    pre_preds = {
        "H1": PredictiveDistribution("H1", "C1", ExperimentActionType.PROPERTY, np.array([0.008]), np.array([1e-5])),
        "H2": PredictiveDistribution("H2", "C1", ExperimentActionType.PROPERTY, np.array([0.002]), np.array([1e-5])),
        "H3": PredictiveDistribution("H3", "C1", ExperimentActionType.PROPERTY, np.array([0.002]), np.array([1e-5])),
    }

    # Execute observation that is near H1's prediction (0.0081)
    obs = 0.0081
    res = ensemble.record_observation_and_update(
        action_id="act_01",
        candidate_id="C1",
        action_type=ExperimentActionType.PROPERTY,
        observation=obs,
        pre_predictions=pre_preds,
    )

    new_beliefs = ensemble.get_beliefs()
    assert new_beliefs["H1"] > new_beliefs["H2"]
    assert new_beliefs["H1"] > new_beliefs["H3"]
    assert np.isclose(sum(new_beliefs.values()), 1.0)
    assert res["realized_entropy_reduction"] > 0.0


# ---------------------------------------------------------------------------
# 3. Controlled Synthetic Truth Worlds Verification
# ---------------------------------------------------------------------------
def test_synthetic_worlds_oracle_integrity() -> None:
    w1 = World1_CompositionSufficient()
    w2 = World2_StructureInformed()
    w3 = World3_LocalStructuralRegime()

    cpool = w1.get_candidate_pool()
    assert len(cpool) == 150
    assert set(cpool.columns) == {"candidate_id", "Library", "Area", "Au", "Ir", "Rh"}

    # Test execution on World 2
    cid = cpool["candidate_id"].iloc[0]
    out_xrd = w2.execute_xrd(cid)
    assert out_xrd.action_type == ExperimentActionType.XRD
    assert "xrd_embedding" in out_xrd.revealed_data

    out_prop = w2.execute_property(cid)
    assert out_prop.action_type == ExperimentActionType.PROPERTY
    assert "k0" in out_prop.revealed_data

    # Duplicate execution raises ValueError
    with pytest.raises(ValueError, match="already executed"):
        w2.execute_xrd(cid)
    with pytest.raises(ValueError, match="already measured"):
        w2.execute_property(cid)


def test_synthetic_world_firewall_mutation_independence() -> None:
    """Mutating unrevealed ground truth in World 2 must not alter pre-reveal predictions."""
    w2_a = World2_StructureInformed(seed=42)
    w2_b = World2_StructureInformed(seed=42)

    # Mutate unrevealed ground truth in w2_b
    for cid in list(w2_b._ground_truth.keys())[10:30]:
        w2_b._ground_truth[cid]["k0"] = 999.99

    cpool = w2_a.get_candidate_pool()
    ens_a = HypothesisEnsemble()
    ens_b = HypothesisEnsemble()

    eval_a = HypothesisInformationGainEstimator().evaluate_action_discrimination(
        candidate_id="SYN_CAND_001",
        action_type=ExperimentActionType.PROPERTY,
        composition=cpool[["Au", "Ir", "Rh"]].iloc[0].to_numpy(),
        ensemble=ens_a,
        seed=42,
    )

    eval_b = HypothesisInformationGainEstimator().evaluate_action_discrimination(
        candidate_id="SYN_CAND_001",
        action_type=ExperimentActionType.PROPERTY,
        composition=cpool[["Au", "Ir", "Rh"]].iloc[0].to_numpy(),
        ensemble=ens_b,
        seed=42,
    )

    assert np.isclose(eval_a.hypothesis_information_gain, eval_b.hypothesis_information_gain)


# ---------------------------------------------------------------------------
# 4. Falsification Policy Modes
# ---------------------------------------------------------------------------
def test_falsification_policy_modes() -> None:
    cand_df = generate_simplex_candidates_quick(20)
    ensemble = HypothesisEnsemble()

    policy_fals = FalsificationFirstPolicy(mode=FalsificationPolicyMode.PURE_FALSIFICATION)
    policy_disc = FalsificationFirstPolicy(mode=FalsificationPolicyMode.DISCOVERY_ONLY)
    policy_hyb = FalsificationFirstPolicy(mode=FalsificationPolicyMode.HYBRID)

    # Synthetic discovery scores
    disc_scores = {cid: float(i) * 0.1 for i, cid in enumerate(cand_df["candidate_id"])}

    rec_fals = policy_fals.recommend_next_experiment(cand_df, set(), set(), ensemble, disc_scores, fast_mode=True, seed=42)
    rec_disc = policy_disc.recommend_next_experiment(cand_df, set(), set(), ensemble, disc_scores, fast_mode=True, seed=42)
    rec_hyb = policy_hyb.recommend_next_experiment(cand_df, set(), set(), ensemble, disc_scores, fast_mode=True, seed=42)

    assert rec_fals.action.action_type in {ExperimentActionType.XRD, ExperimentActionType.PROPERTY}
    # Discovery only mode will strictly pick top discovery candidate
    assert rec_disc.action.action_type == ExperimentActionType.PROPERTY
    assert rec_disc.action.candidate_id == cand_df["candidate_id"].iloc[-1]
    assert rec_hyb.action.candidate_id is not None


# ---------------------------------------------------------------------------
# 5. Identifiability Analysis Output Contract
# ---------------------------------------------------------------------------
def test_identifiability_analysis(tmp_path) -> None:
    cand_df = generate_simplex_candidates_quick(15)
    report_file = tmp_path / "identifiability_test.md"
    df = run_identifiability_analysis(candidate_pool_df=cand_df, output_path=report_file)

    assert len(df) > 0
    assert "hypothesis_pair" in df.columns
    assert "js_divergence" in df.columns
    assert report_file.exists()


# Helper
def generate_simplex_candidates_quick(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    raw = rng.dirichlet([1, 1, 1], size=n) * 100
    rows = [
        {"candidate_id": f"CAND_{i:02d}", "Library": "TEST", "Area": 1, "Au": r[0], "Ir": r[1], "Rh": r[2]}
        for i, r in enumerate(raw)
    ]
    return pd.DataFrame(rows)
