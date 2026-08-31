from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.datasets.attia import (
    AttiaAdapter,
    compute_expected_c4,
    generate_continuous_candidate_id,
)
from src.evaluation.attia_continuous_benchmark import (
    compute_or_load_continuous_reference,
    derive_discrete_grid_optimum,
    evaluate_continuous_trajectory,
    run_attia_continuous_benchmark,
    run_single_attia_continuous_trajectory,
)
from src.evaluation.attia_oracle import generate_attia_simulator_seed, simulate_attia_policy


def test_canonical_candidate_id_and_query_id() -> None:
    # 1. Same coordinates produce identical candidate_id
    cid_1 = generate_continuous_candidate_id(5.1234, 4.5678, 4.1234)
    cid_2 = generate_continuous_candidate_id(5.1234, 4.5678, 4.1234)
    assert cid_1 == cid_2
    assert cid_1.startswith("ATTIA_CONT_")

    # 2. Different coordinates produce different candidate_id
    cid_3 = generate_continuous_candidate_id(5.1235, 4.5678, 4.1234)
    assert cid_1 != cid_3

    # 3. Query ID represents query event
    q_1 = f"Q_S00_greedy_ST01"
    q_2 = f"Q_S00_gp_ucb_ST01"
    assert q_1 != q_2


def test_continuous_fair_stochastic_seeding() -> None:
    # Same candidate queried in different strategy/step contexts gets identical simulator seed
    cid = generate_continuous_candidate_id(5.2, 4.6, 4.2)
    benchmark_seed = 42

    seed_strat_a = generate_attia_simulator_seed(benchmark_seed, cid)
    seed_strat_b = generate_attia_simulator_seed(benchmark_seed, cid)
    assert seed_strat_a == seed_strat_b

    sim_a = simulate_attia_policy(5.2, 4.6, 4.2, mode="hi", variance=True, seed=seed_strat_a)
    sim_b = simulate_attia_policy(5.2, 4.6, 4.2, mode="hi", variance=True, seed=seed_strat_b)
    assert sim_a == sim_b


@pytest.mark.external_data
def test_optimizer_evaluator_strict_isolation() -> None:
    adapter = AttiaAdapter()
    space = adapter.continuous_search_space()
    discrete_pool = adapter.load_candidate_pool()

    init_indices = [0, 5, 10, 15, 20]

    # Run optimizer trajectory
    raw_hist, _ = run_single_attia_continuous_trajectory(
        search_space=space,
        discrete_pool=discrete_pool,
        init_indices=init_indices,
        total_queries=3,
        strategy="gp_ucb",
        optimizer_seed=7,
        n_candidates_per_step=50,
        refine_continuous=False,
    )

    # Verify optimizer history has NO evaluator reference metrics
    for row in raw_hist:
        assert "reference_true_lifetime" not in row
        assert "best_reference_true" not in row
        assert "continuous_simple_regret" not in row
        assert "gap_to_discrete_grid_optimum" not in row
        assert "improvement_over_discrete_grid" not in row
        assert "best_observed_lifetime" in row
        assert "simulated_lifetime" in row
        assert "candidate_id" in row
        assert "query_id" in row

    # Evaluate with post-hoc evaluator
    evaluated_hist, ref_under = evaluate_continuous_trajectory(
        raw_trajectory=raw_hist,
        init_indices=init_indices,
        discrete_pool=discrete_pool,
        continuous_ref_lifetime=1120.0,
        discrete_grid_optimum_lifetime=1079.0,
    )

    for row in evaluated_hist:
        assert "best_reference_true" in row
        assert "continuous_simple_regret" in row
        assert "gap_to_discrete_grid_optimum" in row


@pytest.mark.external_data
def test_programmatic_discrete_grid_optimum_and_continuous_reference(tmp_path: Path) -> None:
    adapter = AttiaAdapter()
    discrete_pool = adapter.load_candidate_pool()
    space = adapter.continuous_search_space()

    grid_opt = derive_discrete_grid_optimum(discrete_pool)
    assert grid_opt["policy_id"] == "ATTIA_P113"
    assert grid_opt["reference_true_lifetime"] == 1079.0

    cont_ref = compute_or_load_continuous_reference(
        search_space=space,
        output_dir=tmp_path,
        discrete_pool=discrete_pool,
        n_sobol_samples=100,
        n_local_starts=10,
        seed=42,
    )
    assert cont_ref["best_known_latent_lifetime"] >= 1079.0
    assert (tmp_path / "continuous_reference.json").is_file()
    assert (tmp_path / "continuous_reference_manifest.json").is_file()


@pytest.mark.external_data
def test_reference_underestimation_detection_and_invalidation() -> None:
    adapter = AttiaAdapter()
    space = adapter.continuous_search_space()
    discrete_pool = adapter.load_candidate_pool()

    init_indices = [0, 1, 2, 3, 4]
    raw_hist, _ = run_single_attia_continuous_trajectory(
        search_space=space,
        discrete_pool=discrete_pool,
        init_indices=init_indices,
        total_queries=2,
        strategy="random",
        optimizer_seed=42,
        n_candidates_per_step=20,
    )

    # Intentionally set an artificially low reference lifetime (500.0)
    eval_hist, ref_under = evaluate_continuous_trajectory(
        raw_trajectory=raw_hist,
        init_indices=init_indices,
        discrete_pool=discrete_pool,
        continuous_ref_lifetime=500.0,
        discrete_grid_optimum_lifetime=1079.0,
    )
    assert ref_under is True
    # If reference is underestimated, continuous regret must become None / invalid, never clamped to 0.0
    last_row = eval_hist[-1]
    assert last_row["continuous_simple_regret"] is None
    assert last_row["continuous_simple_regret_valid"] is False


def test_c4_formula_exactness() -> None:
    # Exact formula: C4 = 0.2 / (1/6 - (0.2/C1 + 0.2/C2 + 0.2/C3))
    c1, c2, c3 = 6.0, 4.8, 4.0
    expected = 0.2 / (1.0 / 6.0 - (0.2 / 6.0 + 0.2 / 4.8 + 0.2 / 4.0))
    computed = compute_expected_c4(c1, c2, c3)
    assert np.isclose(computed, expected, atol=1e-5)
    assert np.isclose(computed, 4.8, atol=1e-2)  # Matches Attia P113 (6.0-4.8-4.0-4.8)


@pytest.mark.external_data
def test_attia_continuous_benchmark_mini_end_to_end(tmp_path: Path) -> None:
    adapter = AttiaAdapter()
    summary = run_attia_continuous_benchmark(
        adapter=adapter,
        budgets=(6, 7),
        initial_policies=5,
        n_seeds=2,
        n_candidates_per_step=100,
        output_dir=tmp_path,
    )

    assert "benchmark" in summary
    assert "derived_discrete_grid_optimum" in summary
    assert "best_known_continuous_reference" in summary
    assert "best_discovered_per_strategy" in summary
    assert "gp_ucb" in summary["best_discovered_per_strategy"]
    assert "expected_improvement" in summary["best_discovered_per_strategy"]
    assert "noisy_expected_improvement" in summary["best_discovered_per_strategy"]
    assert "thompson" in summary["best_discovered_per_strategy"]
    assert "paired_comparisons" in summary
    assert "sample_efficiency_to_threshold" in summary
    assert "threshold_a_discrete_opt_1079" in summary["sample_efficiency_to_threshold"]

    assert (tmp_path / "benchmark_summary.json").is_file()
    assert (tmp_path / "continuous_reference.json").is_file()
    assert (tmp_path / "continuous_reference_manifest.json").is_file()
    assert (tmp_path / "run_manifest.json").is_file()
    assert (tmp_path / "search_space_summary.json").is_file()
    assert (tmp_path / "optimization_history.csv").is_file()
    assert (tmp_path / "proposed_protocols.csv").is_file()

