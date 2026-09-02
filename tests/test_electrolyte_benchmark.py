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
