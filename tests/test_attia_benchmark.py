from __future__ import annotations

import json
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
    compute_bootstrap_mean_ci,
    evaluate_trajectory_metrics,
    run_attia_optimization_benchmark,
    run_single_attia_optimization_trajectory,
)
from src.evaluation.attia_oracle import (
    AttiaSimulatorOracle,
    compute_or_load_reference_landscape,
    generate_attia_simulator_seed,
    simulate_attia_policy,
)


@pytest.mark.external_data
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
    forbidden_cols = [
        "simulated_lifetime",
        "lifetime",
        "target",
        "reference_mean_lifetime",
        "reference_true_lifetime",
        "cycles",
    ]
    for col in forbidden_cols:
        assert col not in cand_pool.columns


@pytest.mark.external_data
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


@pytest.mark.external_data
def test_attia_fair_stochastic_seeding() -> None:
    benchmark_seed = 42
    policy_id = "ATTIA_P010"

    # Invariant: Seed depends ONLY on benchmark_seed + policy_id
    seed1 = generate_attia_simulator_seed(benchmark_seed, policy_id)
    seed2 = generate_attia_simulator_seed(benchmark_seed, policy_id)
    assert seed1 == seed2

    # Different policy or different benchmark seed produces different simulator seeds
    diff_policy_seed = generate_attia_simulator_seed(benchmark_seed, "ATTIA_P011")
    diff_bench_seed = generate_attia_simulator_seed(43, policy_id)
    assert seed1 != diff_policy_seed
    assert seed1 != diff_bench_seed

    # Simulator evaluation with that seed gives identical stochastic lifetime
    adapter = AttiaAdapter()
    pool = adapter.load_candidate_pool()
    oracle = AttiaSimulatorOracle(pool, mode="hi", variance=True)
    cand = pool[pool["policy_id"] == policy_id].iloc[0]

    out1 = oracle.query(cand, seed=seed1)
    out2 = oracle.query(cand, seed=seed1)
    assert out1.target == out2.target
    assert out1.metadata["simulator_seed"] == seed1
    assert out1.metadata["simulated"] is True


@pytest.mark.external_data
def test_attia_strict_oracle_contract() -> None:
    adapter = AttiaAdapter()
    pool = adapter.load_candidate_pool()
    oracle = AttiaSimulatorOracle(pool, mode="hi", variance=True)

    # 1. Missing policy_id -> reject
    with pytest.raises(ValueError, match="requires 'policy_id'"):
        oracle.query({"C1": 3.6, "C2": 4.0, "C3": 4.0, "C4": 4.8})

    # 2. Unknown policy_id -> reject
    with pytest.raises(KeyError, match="Unknown policy_id"):
        oracle.query({"policy_id": "UNKNOWN_P999"})

    # 3. Conflicting coordinate with canonical definition -> reject
    valid_cand = pool.iloc[0].to_dict()
    bad_coord_cand = {**valid_cand, "C1": 9.9}
    with pytest.raises(ValueError, match="conflicts with canonical policy"):
        oracle.query(bad_coord_cand)


@pytest.mark.external_data
def test_attia_optimizer_evaluator_separation() -> None:
    adapter = AttiaAdapter()
    pool = adapter.load_candidate_pool()
    oracle = AttiaSimulatorOracle(pool, mode="hi", variance=True)
    feature_cols = list(adapter.spec.feature_columns)

    # Run optimizer trajectory: accepts NO reference landscape, NO global max, NO thresholds
    raw_hist = run_single_attia_optimization_trajectory(
        candidate_pool=pool,
        oracle=oracle,
        feature_cols=feature_cols,
        strategy="greedy",
        init_indices=[0, 1, 2, 3, 4],
        total_queries=3,
        optimizer_seed=42,
    )

    # Optimizer trajectory must NOT contain evaluator-derived fields
    evaluator_forbidden = ["simple_regret", "reference_true_lifetime", "reference_mean_lifetime", "hit_top_10_pct", "hit_top_5_pct"]
    for row in raw_hist:
        for field in evaluator_forbidden:
            assert field not in row, f"Optimizer trajectory leaked evaluator field {field!r}"

    # Verify trajectory contains standard experiment columns
    expected_keys = {
        "benchmark_seed",
        "strategy",
        "step",
        "policy_id",
        "C1",
        "C2",
        "C3",
        "C4",
        "simulator_seed",
        "simulated_lifetime",
        "best_observed_lifetime",
    }
    assert set(raw_hist[0].keys()) == expected_keys


@pytest.mark.external_data
def test_attia_evaluator_stage_metrics() -> None:
    adapter = AttiaAdapter()
    pool = adapter.load_candidate_pool()
    oracle = AttiaSimulatorOracle(pool, mode="hi", variance=True)
    feature_cols = list(adapter.spec.feature_columns)

    ref_df, ref_meta = compute_or_load_reference_landscape(adapter)
    ref_lookup = dict(zip(ref_df["policy_id"].astype(str), ref_df["reference_true_lifetime"].astype(float)))

    init_indices = [0, 1, 2, 3, 4]
    init_pids = [str(pool.iloc[i]["policy_id"]) for i in init_indices]

    raw_hist = run_single_attia_optimization_trajectory(
        candidate_pool=pool,
        oracle=oracle,
        feature_cols=feature_cols,
        strategy="gp_ucb",
        init_indices=init_indices,
        total_queries=3,
        optimizer_seed=100,
    )

    eval_hist = evaluate_trajectory_metrics(
        raw_history=raw_hist,
        init_pids=init_pids,
        ref_lookup=ref_lookup,
        global_max=ref_meta["global_max"],
        top_10_pct_val=ref_meta["top_10_pct_val"],
        top_5_pct_val=ref_meta["top_5_pct_val"],
    )

    assert len(eval_hist) == len(raw_hist)
    for row in eval_hist:
        assert "simple_regret" in row
        assert row["simple_regret"] >= 0.0
        assert "hit_top_10_pct" in row
        assert "hit_top_5_pct" in row
        assert "best_reference_true" in row


@pytest.mark.external_data
def test_attia_reference_manifest_and_cache_invalidation() -> None:
    adapter = AttiaAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = Path(tmpdir) / "reference_landscape.csv"
        tmp_manifest = Path(tmpdir) / "reference_landscape_manifest.json"

        # First build
        ref_df1, meta1 = compute_or_load_reference_landscape(
            adapter,
            eval_seeds=list(range(5)),
            output_path=tmp_csv,
            force_recompute=True,
        )
        assert meta1["cache_reused"] is False
        assert tmp_manifest.exists()

        # Second load -> should reuse cache
        ref_df2, meta2 = compute_or_load_reference_landscape(
            adapter,
            eval_seeds=list(range(5)),
            output_path=tmp_csv,
            force_recompute=False,
        )
        assert meta2["cache_reused"] is True
        assert len(ref_df2) == 224

        # Invalidate manifest (e.g. change simulator_version in manifest)
        with open(tmp_manifest, "r", encoding="utf-8") as f:
            mdata = json.load(f)
        mdata["simulator_version"] = "99.9.9"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(mdata, f)

        # Third load -> detects invalid manifest, rebuilds cache
        ref_df3, meta3 = compute_or_load_reference_landscape(
            adapter,
            eval_seeds=list(range(5)),
            output_path=tmp_csv,
            force_recompute=False,
        )
        assert meta3["cache_reused"] is False


def test_attia_deterministic_bootstrap_ci() -> None:
    data = np.array([10.0, 20.0, 15.0, 30.0, 25.0, 18.0, 22.0, 14.0] * 4, dtype=float)
    mean_val = float(np.mean(data))

    ci_low1, ci_high1 = compute_bootstrap_mean_ci(data, n_bootstraps=2000, ci=0.95, seed=42)
    ci_low2, ci_high2 = compute_bootstrap_mean_ci(data, n_bootstraps=2000, ci=0.95, seed=42)

    # Determinism
    assert ci_low1 == ci_low2
    assert ci_high1 == ci_high2
    # Consistency
    assert ci_low1 <= mean_val <= ci_high1


@pytest.mark.external_data
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
        assert summary["reference_objective"] == "reference_true_lifetime"
        assert summary["total_valid_policies"] == 224
        assert (out_dir / "benchmark_summary.json").exists()
        assert (out_dir / "optimization_history.csv").exists()
        assert (out_dir / "budget_sweep.csv").exists()
        assert (out_dir / "budget_sweep_summary.json").exists()
        assert (out_dir / "reference_landscape.csv").exists()
        assert (out_dir / "reference_landscape_manifest.json").exists()

        # Check optimization history structure
        hist_df = pd.read_csv(out_dir / "optimization_history.csv")
        assert "simple_regret" in hist_df.columns
        assert "simulator_seed" in hist_df.columns
        assert "simulated_lifetime" in hist_df.columns
        assert "reference_true_lifetime" in hist_df.columns
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
