from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.evaluation.falsification_benchmark import (
    run_full_falsification_benchmark,
    run_single_falsification_trajectory,
)
from src.science.falsification.synthetic_worlds import World1_CompositionSufficient


def test_sequential_and_parallel_modes_produce_equivalent_results(tmp_path: Path) -> None:
    """Verifies that parallel and sequential benchmark runs produce identical result shapes and schemas."""
    seq_dir = tmp_path / "seq"
    par_dir = tmp_path / "par"

    # Run with small horizon
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

    # 3 worlds * 5 policies * 1 seed * 2 steps = 30 records
    assert len(records_seq) == 30
    assert len(records_par) == 30
    assert set(df_seq.columns) == set(df_par.columns)

    # Verify both saved artifacts
    assert (seq_dir / "benchmark_summary.csv").exists()
    assert (par_dir / "benchmark_summary.csv").exists()
    assert (seq_dir / "benchmark_report.md").exists()
    assert (par_dir / "benchmark_report.md").exists()


def test_artificial_worker_failure_causes_fail_closed(tmp_path: Path) -> None:
    """Verifies that any worker failure causes benchmark to raise RuntimeError rather than silently succeed."""
    out_dir = tmp_path / "fail_test"

    def faulty_worker(args):
        world_type, policy, n_steps, seed = args
        if policy == "pure_falsification":
            raise RuntimeError("Artificial simulated worker crash")
        from src.evaluation.falsification_benchmark import _run_single_job
        return _run_single_job(args)

    with patch("src.evaluation.falsification_benchmark._run_single_job", side_effect=faulty_worker):
        with pytest.raises(RuntimeError, match="Falsification benchmark failed"):
            run_full_falsification_benchmark(
                seeds=(42,),
                n_steps=2,
                output_dir=out_dir,
                parallel=False,
            )


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

    # Check that sorting is monotonic
    for i in range(len(loaded_json) - 1):
        r1 = loaded_json[i]
        r2 = loaded_json[i + 1]
        k1 = (r1["world"], r1["policy"], r1["seed"], r1["step"])
        k2 = (r2["world"], r2["policy"], r2["seed"], r2["step"])
        assert k1 <= k2
