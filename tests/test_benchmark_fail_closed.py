from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import src.evaluation.falsification_benchmark as benchmark_module
from src.evaluation.falsification_benchmark import (
    run_full_falsification_benchmark,
)


def test_sequential_and_parallel_modes_produce_identical_logical_results(tmp_path: Path) -> None:
    """Verifies that parallel and sequential benchmark runs produce identical deterministic trajectories."""
    seq_dir = tmp_path / "seq"
    par_dir = tmp_path / "par"

    # Run with small horizon and fixed seed
    df_seq, records_seq = run_full_falsification_benchmark(
        seeds=(42,),
        n_steps=2,
        output_dir=seq_dir,
        parallel=False,
    )

    df_par, records_par = run_full_falsification_benchmark(
        seeds=(42,),
        n_steps=2,
        output_dir=par_dir,
        parallel=True,
    )

    assert len(records_seq) == 30
    assert len(records_par) == 30

    # Compare deterministic discrete fields exactly
    discrete_cols = ["world", "true_hypothesis", "policy", "seed", "step", "candidate_id", "action_type", "is_top1_correct"]
    for col in discrete_cols:
        assert (df_seq[col] == df_par[col]).all(), f"Mismatch in column {col}"

    # Compare numerical trajectory fields
    numeric_cols = [
        "cost_spent",
        "true_hypothesis_weight",
        "hypothesis_entropy",
        "best_observed_k0",
        "expected_hig",
    ]
    for col in numeric_cols:
        assert np.allclose(df_seq[col], df_par[col], atol=1e-5), f"Numerical difference in column {col}"


def test_sequential_worker_failure_fail_closed(tmp_path: Path) -> None:
    """Verifies sequential fail-closed behavior when worker fails."""
    out_dir = tmp_path / "fail_seq"
    real_worker = benchmark_module._run_single_job

    def faulty_worker(args: tuple[str, str, int, int]):
        _, policy, _, _ = args
        if policy == "pure_falsification":
            raise RuntimeError("Artificial simulated sequential worker crash")
        return real_worker(args)

    with patch.object(benchmark_module, "_run_single_job", side_effect=faulty_worker):
        with pytest.raises(RuntimeError, match="Falsification benchmark failed"):
            run_full_falsification_benchmark(
                seeds=(42,),
                n_steps=2,
                output_dir=out_dir,
                parallel=False,
            )


def test_parallel_worker_failure_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies parallel ProcessPoolExecutor fail-closed behavior when a worker raises an exception."""
    out_dir = tmp_path / "fail_par"
    monkeypatch.setenv("_TEST_BENCHMARK_SIMULATE_FAILURE_POLICY", "pure_falsification")

    with pytest.raises(RuntimeError, match="Falsification benchmark failed"):
        run_full_falsification_benchmark(
            seeds=(42,),
            n_steps=2,
            output_dir=out_dir,
            parallel=True,
        )


def test_completeness_validation_without_digit_heuristics() -> None:
    """Verifies that completeness validation checks canonical tuples directly without digit heuristics."""
    jobs = [("World1", "pure_falsification", 2, 42)]
    # Incomplete records simulating missing world
    incomplete_records = [
        {
            "world": "Synthetic_World_2_H2_True",
            "policy": "pure_falsification",
            "seed": 42,
            "step": 0,
        }
    ]
    raw_df = pd.DataFrame(incomplete_records)
    expected_combinations = {(benchmark_module.WORLD_TYPE_TO_CANONICAL_NAME.get(w, w), p, s) for w, p, _, s in jobs}
    actual_combinations = set(zip(raw_df["world"], raw_df["policy"], raw_df["seed"]))
    missing = expected_combinations - actual_combinations
    assert len(missing) == 1
    assert ("Synthetic_World_1_H1_True", "pure_falsification", 42) in missing


def test_deterministic_record_ordering(tmp_path: Path) -> None:
    """Verifies that benchmark results are deterministically sorted by world, policy, seed, step."""
    out_dir = tmp_path / "order_test"
    df, records = run_full_falsification_benchmark(
        seeds=(42,),
        n_steps=2,
        output_dir=out_dir,
        parallel=False,
    )

    with open(out_dir / "benchmark_runs.json", "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    for i in range(len(loaded_json) - 1):
        r1 = loaded_json[i]
        r2 = loaded_json[i + 1]
        k1 = (r1["world"], r1["policy"], r1["seed"], r1["step"])
        k2 = (r2["world"], r2["policy"], r2["seed"], r2["step"])
        assert k1 <= k2
