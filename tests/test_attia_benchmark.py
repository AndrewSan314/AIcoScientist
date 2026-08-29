from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.datasets.attia import (
    ATTIA_CANDIDATE_COLUMNS,
    ATTIA_FEATURE_COLUMNS,
    AttiaAdapter,
    compute_expected_c4,
    load_raw_attia_policies,
)
from src.datasets.registry import get_dataset_adapter
from src.evaluation.attia_benchmark import (
    run_attia_optimization_benchmark,
    run_single_attia_optimization_trajectory,
)
from src.evaluation.attia_oracle import (
    AttiaSimulatorOracle,
    compute_or_load_reference_landscape,
    simulate_attia_policy,
)


def test_attia_candidate_space_schema() -> None:
    adapter = AttiaAdapter()
    cand_pool = adapter.load_candidate_pool()

    expected_cols = ["policy_id", *ATTIA_CANDIDATE_COLUMNS]
    assert list(cand_pool.columns) == expected_cols, f"Candidate pool columns must be {expected_cols}, got {list(cand_pool.columns)}"
    assert len(cand_pool) == 224, f"Expected 224 policies, got {len(cand_pool)}"

    # Check unique IDs
    assert cand_pool["policy_id"].nunique() == 224
    assert cand_pool["policy_id"].iloc[0] == "ATTIA_P000"
    assert cand_pool["policy_id"].iloc[-1] == "ATTIA_P223"

    # Verify all feature values are finite
    for col in ATTIA_FEATURE_COLUMNS:
        assert np.all(np.isfinite(cand_pool[col].to_numpy(dtype=float)))

    # Verify zero oracle leakage in candidate space
    forbidden_cols = ["simulated_lifetime", "lifetime", "target", "reference_mean_lifetime", "cycles"]
    for col in forbidden_cols:
        assert col not in cand_pool.columns


def test_attia_policy_c4_validation() -> None:
    adapter = AttiaAdapter()
    cand_pool = adapter.load_candidate_pool()

    c1 = cand_pool["C1"].to_numpy(dtype=float)
    c2 = cand_pool["C2"].to_numpy(dtype=float)
    c3 = cand_pool["C3"].to_numpy(dtype=float)
    c4 = cand_pool["C4"].to_numpy(dtype=float)

    expected_c4 = compute_expected_c4(c1, c2, c3)
    diff = np.abs(c4 - expected_c4)
    assert np.max(diff) <= 1e-3, f"Max C4 discrepancy {np.max(diff)} exceeds tolerance 1e-3"

    # Bounds check
    assert np.all(c4 >= 0.1) and np.all(c4 <= 4.81)

    # Baseline exclusion check
    baseline = (c1 == 4.8) & (c2 == 4.8) & (c3 == 4.8)
    assert not np.any(baseline), "Baseline (4.8, 4.8, 4.8, 4.8) must be excluded"


def test_attia_malformed_policy_rejection() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Non-numeric value
        bad_csv1 = tmp_path / "bad1.csv"
        bad_csv1.write_text("3.6,6.0,5.6,invalid\n", encoding="utf-8")
        with pytest.raises(ValueError, match="non-numeric"):
            load_raw_attia_policies(bad_csv1, expected_policies=None)

        # 2. C4 constraint violation
        bad_csv2 = tmp_path / "bad2.csv"
        bad_csv2.write_text("3.6,6.0,5.6,1.000\n", encoding="utf-8")
        with pytest.raises(ValueError, match="violates C4 charging constraint"):
            load_raw_attia_policies(bad_csv2, expected_policies=None)

        # 3. Includes baseline
        bad_csv3 = tmp_path / "bad3.csv"
        bad_csv3.write_text("4.8,4.8,4.8,4.800\n", encoding="utf-8")
        with pytest.raises(ValueError, match="exclude the baseline policy"):
            load_raw_attia_policies(bad_csv3, expected_policies=None)

        # 4. Duplicate rows
        bad_csv4 = tmp_path / "bad4.csv"
        bad_csv4.write_text("3.6,6.0,5.6,4.755\n3.6,6.0,5.6,4.755\n", encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            load_raw_attia_policies(bad_csv4, expected_policies=None)


def test_attia_simulator_oracle_determinism_and_stochasticity() -> None:
    adapter = AttiaAdapter()
    pool = adapter.load_candidate_pool()
    oracle = AttiaSimulatorOracle(pool, mode="hi", variance=True)

    cand = pool.iloc[0]

    # Determinism: same seed produces identical lifetime
    res1 = oracle.query(cand, seed=42)
    res2 = oracle.query(cand, seed=42)
    assert res1.target == res2.target
    assert res1.metadata["simulated"] is True
    assert res1.metadata["data_type"] == "simulated_lifetime"
    assert res1.metadata["simulator_seed"] == 42

    # Stochasticity: different seeds produce different lifetimes
    draws = [oracle.query(cand, seed=s).target for s in range(20)]
    assert len(set(draws)) > 1, "Stochastic simulator should produce variation across different seeds"


def test_attia_paired_seed_fairness() -> None:
    adapter = AttiaAdapter()
    pool = adapter.load_candidate_pool()
    oracle = AttiaSimulatorOracle(pool, mode="hi", variance=True)
    feature_cols = list(adapter.spec.feature_columns)

    ref_df, ref_meta = compute_or_load_reference_landscape(adapter)
    ref_lookup = dict(zip(ref_df["policy_id"].astype(str), ref_df["reference_mean_lifetime"].astype(float)))
    evaluator_meta = {
        "global_max": ref_meta["global_max"],
        "top_10_pct_val": ref_meta["top_10_pct_val"],
        "top_5_pct_val": ref_meta["top_5_pct_val"],
        "ref_lookup": ref_lookup,
    }

    init_indices = [0, 1, 2, 3, 4]
    opt_seed = 123

    # Run Greedy and GP-UCB with same warmup and seed
    hist_greedy = run_single_attia_optimization_trajectory(
        candidate_pool=pool,
        oracle=oracle,
        feature_cols=feature_cols,
        strategy="greedy",
        init_indices=init_indices,
        total_queries=3,
        evaluator_meta=evaluator_meta,
        optimizer_seed=opt_seed,
    )
    hist_ucb = run_single_attia_optimization_trajectory(
        candidate_pool=pool,
        oracle=oracle,
        feature_cols=feature_cols,
        strategy="gp_ucb",
        init_indices=init_indices,
        total_queries=3,
        evaluator_meta=evaluator_meta,
        optimizer_seed=opt_seed,
    )

    # Initial warm-ups must match exactly
    assert hist_greedy[0]["best_simulated_lifetime"] == hist_ucb[0]["best_simulated_lifetime"]
    assert hist_greedy[0]["best_reference_mean"] == hist_ucb[0]["best_reference_mean"]
    assert hist_greedy[0]["simple_regret"] == hist_ucb[0]["simple_regret"]


def test_attia_reference_landscape_isolation() -> None:
    adapter = AttiaAdapter()
    cand_pool = adapter.load_candidate_pool()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ref_path = Path(tmpdir) / "ref_test.csv"
        ref_df, meta = compute_or_load_reference_landscape(
            adapter,
            eval_seeds=list(range(5)),  # Fast subset for test
            output_path=tmp_ref_path,
            force_recompute=True,
        )

        assert len(ref_df) == 224
        assert "reference_mean_lifetime" in ref_df.columns
        assert "reference_true_lifetime" in ref_df.columns
        assert meta["global_max"] > 0
        assert meta["top_10_pct_val"] <= meta["global_max"]
        assert meta["top_5_pct_val"] <= meta["global_max"]

        # Ensure candidate pool remains uncontaminated
        for col in ["reference_mean_lifetime", "reference_true_lifetime"]:
            assert col not in cand_pool.columns


def test_attia_benchmark_end_to_end() -> None:
    adapter = AttiaAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "attia_out"

        summary = run_attia_optimization_benchmark(
            adapter=adapter,
            budgets=(7, 9),
            initial_policies=5,
            n_seeds=3,
            output_dir=out_dir,
            force_recompute=False,
        )

        assert summary["benchmark"] == "Attia et al. 2020 Fast-Charging Optimization Benchmark"
        assert summary["benchmark_nature"] == "simulator != experimental dataset"
        assert summary["total_valid_policies"] == 224
        assert (out_dir / "benchmark_summary.json").exists()
        assert (out_dir / "optimization_history.csv").exists()
        assert (out_dir / "budget_sweep.csv").exists()
        assert (out_dir / "budget_sweep_summary.json").exists()
        assert (out_dir / "reference_landscape.csv").exists()

        # Check optimization history structure
        hist_df = pd.read_csv(out_dir / "optimization_history.csv")
        assert "simple_regret" in hist_df.columns
        assert "strategy" in hist_df.columns
        assert set(hist_df["strategy"].unique()) == {"random", "greedy", "gp_ucb"}


def test_attia_registry_and_cli_routing() -> None:
    adapter = get_dataset_adapter("attia")
    assert isinstance(adapter, AttiaAdapter)
    assert adapter.spec.name == "attia"
    assert adapter.spec.supports_optimization is True
    assert adapter.spec.supports_prediction is False

    from run_pipeline import main

    # Recommendation mode without prior experiment must be rejected
    with pytest.raises(ValueError, match="offline protocol optimization benchmark"):
        main(dataset="attia", mode="recommend")
