"""Tests verifying artifacts, physical correctness, and deliverables of Drakopoulos v2 rediscovery benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def v2_dir() -> Path:
    p = Path("outputs/drakopoulos_rediscovery_v2")
    if not p.is_dir():
        pytest.skip(f"Benchmark artifacts directory {p} does not exist.")
    return p


def test_v2_core_deliverables_exist(v2_dir: Path) -> None:
    expected_files = [
        "summary.json",
        "slide_summary.json",
        "policy_summary.csv",
        "recipe_table.csv",
        "trajectories.json",
        "DRAKOPOULOS_REDISCOVERY_REPORT_V2.md",
    ]
    for fname in expected_files:
        fpath = v2_dir / fname
        assert fpath.is_file(), f"Missing expected deliverable: {fname}"
        assert fpath.stat().st_size > 0, f"File {fname} is empty."


def test_v2_figures_exist_and_valid(v2_dir: Path) -> None:
    figures_dir = v2_dir / "figures"
    assert figures_dir.is_dir(), "Missing figures directory."

    expected_figures = [
        "regret_vs_experiment.png",
        "best_so_far_vs_experiment.png",
        "rediscovery_success_rate.png",
        "hidden_best_rank.png",
        "recipe_performance_landscape.png",
        "budget_sensitivity.png",
        "surrogate_calibration.png",
    ]
    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.is_file(), f"Missing figure: {fig_name}"
        assert fig_path.stat().st_size > 10_000, f"Figure {fig_name} is too small."
        with open(fig_path, "rb") as f:
            header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n", f"File {fig_name} is not a valid PNG."


def test_v2_recipe_table_physical_correctness(v2_dir: Path) -> None:
    df = pd.read_csv(v2_dir / "recipe_table.csv")
    assert len(df) == 26, f"Expected 26 candidate recipes, got {len(df)}"

    # Check controls are in physical bounds (P0-1 fix verification)
    assert (df["coating_speed_m_per_min"] >= 0.05).all()
    assert (df["coating_speed_m_per_min"] <= 1.0).all(), "Unphysical coating speeds detected!"

    assert (df["coating_gap_um"] >= 50.0).all()
    assert (df["coating_gap_um"] <= 500.0).all(), "Unphysical coating gap detected!"

    assert (df["active_material_fraction_pct"] >= 90.0).all()
    assert (df["active_material_fraction_pct"] <= 98.0).all(), "Unphysical active material fraction!"

    assert (df["drying_temperature_c"] >= 50.0).all()
    assert (df["drying_temperature_c"] <= 130.0).all(), "Unphysical drying temperature!"

    # Champion recipe check (P0-2 fix verification)
    best = df.iloc[0]
    assert best["recipe_id"] == "protocol-d3602183e567"
    assert "ASC-52" in str(best["cell_ids"])
    assert pytest.approx(best["discharge_specific_capacity_cycle30_mah_g"], abs=0.5) == 402.25
    assert best["calendering_applied"] == 1.0


def test_v2_slide_summary_scientific_guarantees(v2_dir: Path) -> None:
    with open(v2_dir / "slide_summary.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["retrospective_best_recipe"]["id"] == "protocol-d3602183e567"
    assert pytest.approx(data["retrospective_best_recipe"]["capacity_d30_mah_g"], abs=0.5) == 402.25
    assert data["zero_leakage_firewall"] is True
    assert data["mode_2_status"] == "PUBLISHED_DESIGN_REQUIRES_RESTRICTED_PARTITION_C_MAPPING"

    hit_rates = data["hit_rate_comparison"]
    assert "AICOSCIENTIST_PROCESS_ENGINE" in hit_rates
    assert "random" in hit_rates

    engine_hit5 = hit_rates["AICOSCIENTIST_PROCESS_ENGINE"]["hit_at_5_pct"]
    random_hit5 = hit_rates["random"]["hit_at_5_pct"]
    assert engine_hit5 >= random_hit5, "Engine Hit@5 should match or beat empirical random!"

    engine_regret = hit_rates["AICOSCIENTIST_PROCESS_ENGINE"]["mean_simple_regret_mah_g"]
    random_regret = hit_rates["random"]["mean_simple_regret_mah_g"]
    assert engine_regret < random_regret, "Engine simple regret must be lower than random exploration!"
