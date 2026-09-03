import json
import os
import numpy as np
import pytest

from src.domains.electrolyte.adapter import ElectrolyteDomainAdapter
from src.evaluation.electrolyte_benchmark import (
    evaluate_historical_policy,
    run_comprehensive_historical_benchmark,
)

FIXTURE_PATH = "tests/fixtures/electrolyte/pool_compatible_deexpanded_outcomes.csv"


def test_bootstrap_not_credited_as_autonomous_discovery():
    """Verifies that the historical benchmark strictly separates bootstrap from autonomous discoveries."""
    res = evaluate_historical_policy(
        policy_name="RANDOM",
        derived_outcomes_path=FIXTURE_PATH,
        seed=42,
        max_steps=3,
    )

    assert len(res.bootstrap_candidate_ids) == 3
    # Autonomous discoveries cannot include bootstrap candidate IDs
    for auto_cid in res.autonomous_actions:
        assert auto_cid not in res.bootstrap_candidate_ids

    # Improvement over bootstrap must be max(0, best_autonomous - bootstrap_best)
    expected_imprv = max(0.0, res.best_autonomous_found - res.bootstrap_best)
    assert abs(res.improvement_over_bootstrap - expected_imprv) < 1e-4


def test_historical_replay_labeled_retrospective():
    """Verifies that the benchmark metadata clearly identifies finite retrospective replay."""
    res_dict = run_comprehensive_historical_benchmark(
        derived_outcomes_path=FIXTURE_PATH,
        seeds=(42,),
        policies=("RANDOM", "DISCOVERY_ONLY"),
        max_steps=2,
    )
    title = res_dict["benchmark_metadata"]["title"]
    assert "Retrospective" in title
    assert "Historical" in title


def test_surrogate_results_labeled_simulated():
    """Verifies that surrogate oracle outcomes are strictly tagged as SIMULATED."""
    from src.domains.electrolyte.data import load_derived_historical_outcomes
    from src.domains.electrolyte.oracle import SurrogateElectrolyteOracle
    from src.science.actions import ScientificAction

    df_hist = load_derived_historical_outcomes(FIXTURE_PATH)
    oracle = SurrogateElectrolyteOracle(df_train=df_hist)

    action = ScientificAction(action_id="sim_act", candidate_id="c_test", action_type="CAPACITY_TEST")
    comp = np.zeros(11, dtype=np.float64)
    outcome = oracle.reveal(action, comp)

    assert outcome.provenance["oracle_kind"] == "surrogate_simulation"
    assert outcome.provenance["experimental"] is False
    assert "SIMULATED" in outcome.provenance["label"]


def test_temporal_benchmark_has_no_future_leakage():
    """Verifies that in temporal round evaluations, training data contains strictly batches <= t."""
    audit_gen_path = "outputs/electrolyte/audit/campaign_generalization.json"
    if os.path.exists(audit_gen_path):
        with open(audit_gen_path) as f:
            data = json.load(f)
        for r in data["rounds"]:
            t_test = r["test_batch"]
            tr_batches = r["train_batches"]  # e.g. "0..0", "0..1", etc.
            max_tr = int(tr_batches.split("..")[1])
            assert max_tr < t_test, f"Future leakage detected: train {max_tr} >= test {t_test}"


def test_metric_report_matches_json():
    """Verifies that benchmark metadata contains all required accounting fields."""
    res_dict = run_comprehensive_historical_benchmark(
        derived_outcomes_path=FIXTURE_PATH,
        seeds=(42,),
        policies=("RANDOM",),
        max_steps=2,
    )
    meta = res_dict["benchmark_metadata"]
    assert "historical_pool_size" in meta
    assert "global_pool_maximum" in meta
    assert "bootstrap_best_capacity" in meta
    assert "objective_saturation_status" in meta
    assert "saturation_ratio" in meta


def test_botorch_ei_and_gpucb_direct_baselines_run_successfully():
    """Phase 5: Verifies that BOTORCH_EI_DIRECT and BOTORCH_GPUCB_DIRECT run as real baselines."""
    res_ei = evaluate_historical_policy(
        policy_name="BOTORCH_EI_DIRECT",
        derived_outcomes_path=FIXTURE_PATH,
        seed=42,
        max_steps=3,
    )
    assert res_ei.policy_name == "BOTORCH_EI_DIRECT"
    assert res_ei.steps_count == 3
    assert len(res_ei.autonomous_observations) == 3
    assert res_ei.final_entropy >= 0.0

    res_ucb = evaluate_historical_policy(
        policy_name="BOTORCH_GPUCB_DIRECT",
        derived_outcomes_path=FIXTURE_PATH,
        seed=42,
        max_steps=3,
    )
    assert res_ucb.policy_name == "BOTORCH_GPUCB_DIRECT"
    assert res_ucb.steps_count == 3


def test_hig_separated_into_nats_and_normalized():
    """Phase 3: Verifies cumulative_hig_nats and cumulative_hig_normalized are both recorded."""
    res = evaluate_historical_policy(
        policy_name="HYBRID",
        derived_outcomes_path=FIXTURE_PATH,
        seed=42,
        max_steps=3,
    )
    assert hasattr(res, "cumulative_hig_nats")
    assert hasattr(res, "cumulative_hig_normalized")
    assert res.cumulative_hig_nats >= 0.0
    assert res.cumulative_hig_normalized >= 0.0
    assert len(res.step_diagnostics) == 3
    for diag in res.step_diagnostics:
        assert "raw_hig_nats" in diag
        assert "normalized_hig" in diag
        assert "max_belief_shift" in diag


def test_wow_scenario_returns_honest_fallback_when_preregistered_criteria_unmet():
    """Phase 10: Verifies that when no candidate meets preregistered divergence criteria, fallback is returned."""
    from src.evaluation.electrolyte_benchmark import find_natural_wow_scenario

    # Evaluate identical dummy runs
    res1 = evaluate_historical_policy("DISCOVERY_ONLY", derived_outcomes_path=FIXTURE_PATH, seed=42, max_steps=2)
    # Even if passing res1 twice, disc_cand == hyb_cand, so no divergence
    wow = find_natural_wow_scenario([res1], [res1])
    assert wow["scenario_found"] is False
    assert wow["message"] == "NO NATURAL ELECTROLYTE WOW SCENARIO FOUND UNDER PREREGISTERED SETTINGS"


def test_retrospective_next_batch_ranking_separates_rf_from_aicoscientist(tmp_path):
    """Phase 12: Verifies next-batch ranking separates AIcoScientist from RF baseline."""
    from src.evaluation.electrolyte_benchmark import run_retrospective_next_batch_ranking

    aico_res, rf_res = run_retrospective_next_batch_ranking(
        derived_outcomes_path=FIXTURE_PATH,
        out_dir=str(tmp_path),
    )
    assert "AIcoScientist" in aico_res["model_architecture"]
    assert "RandomForest" in rf_res["model_architecture"]
    assert "REFERENCE ONLY" in rf_res["model_architecture"]
    assert os.path.exists(tmp_path / "aicoscientist_temporal_next_batch.json")
    assert os.path.exists(tmp_path / "rf_temporal_baseline.json")

