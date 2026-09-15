from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.datasets.battery_process.base import RawDatasetUnavailableError
from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosGraphiteAdapter
from src.process.benchmarks.rediscovery import (
    filter_candidate_pool_for_high_loading,
    run_rediscovery_benchmark,
    ProductionProcessRediscoveryRunner,
)

HAS_REAL_DATA = (
    Path("data/external/drakopoulos_graphite/1/processed/2/normalized_runs.json").exists()
    or (
        Path("data/external/drakopoulos_graphite/1/raw/ASC-Cell_Data-Azar-Stavros.xlsx").exists()
        and Path("data/external/drakopoulos_graphite/1/raw/ASC_results-live.xlsx").exists()
    )
)


def _load_real_pool():
    if not HAS_REAL_DATA:
        pytest.skip("Requires local Drakopoulos raw or normalized dataset")
    adapter = DrakopoulosGraphiteAdapter()
    groups = adapter.load_recipe_groups("PROSPECTIVE_MODEL_VALIDATION")
    valid_groups = [g for g in groups if g.valid_d30_replicates > 0]
    records = []
    for g in valid_groups:
        row = {
            "recipe_id": g.recipe_id,
            "discharge_specific_capacity_cycle30_mah_g": g.mean_d30_specific_capacity,
            "std_d30_specific_capacity": g.std_d30_specific_capacity,
            "replicate_count": g.replicate_count,
            "mean_active_mass_mg": g.mean_active_mass_mg,
            "cell_ids": ",".join(g.cell_ids),
            **g.controls,
        }
        records.append(row)
    df = pd.DataFrame(records).sort_values(
        by="discharge_specific_capacity_cycle30_mah_g", ascending=False
    ).reset_index(drop=True)
    non_ctrls = {
        "recipe_id",
        "discharge_specific_capacity_cycle30_mah_g",
        "std_d30_specific_capacity",
        "replicate_count",
        "mean_active_mass_mg",
        "cell_ids",
    }
    ctrl_cols = sorted([c for c in df.columns if c not in non_ctrls])
    return df, ctrl_cols


@pytest.fixture
def mock_high_loading_pool() -> tuple[pd.DataFrame, list[str]]:
    """Self-contained mock candidate pool for CI environments without raw data files."""
    records = [
        {"recipe_id": "r1", "coating_speed_m_per_min": 0.2, "coating_gap_um": 100.0, "mean_active_mass_mg": 7.5, "discharge_specific_capacity_cycle30_mah_g": 402.0},
        {"recipe_id": "r2", "coating_speed_m_per_min": 0.3, "coating_gap_um": 120.0, "mean_active_mass_mg": 9.2, "discharge_specific_capacity_cycle30_mah_g": 380.0},
        {"recipe_id": "r3", "coating_speed_m_per_min": 0.2, "coating_gap_um": 150.0, "mean_active_mass_mg": 16.5, "discharge_specific_capacity_cycle30_mah_g": 355.0},
        {"recipe_id": "r4", "coating_speed_m_per_min": 0.4, "coating_gap_um": 200.0, "mean_active_mass_mg": 17.2, "discharge_specific_capacity_cycle30_mah_g": 285.0},
        {"recipe_id": "r5", "coating_speed_m_per_min": 0.5, "coating_gap_um": 200.0, "mean_active_mass_mg": 16.8, "discharge_specific_capacity_cycle30_mah_g": 260.0},
        {"recipe_id": "r6", "coating_speed_m_per_min": 0.6, "coating_gap_um": 200.0, "mean_active_mass_mg": 18.0, "discharge_specific_capacity_cycle30_mah_g": 240.0},
    ]
    df = pd.DataFrame(records).sort_values(by="discharge_specific_capacity_cycle30_mah_g", ascending=False).reset_index(drop=True)
    ctrl_cols = ["coating_speed_m_per_min", "coating_gap_um"]
    return df, ctrl_cols


def test_filter_candidate_pool_for_high_loading_unit(mock_high_loading_pool):
    df, _ = mock_high_loading_pool
    hl_df = filter_candidate_pool_for_high_loading(df, min_active_mass_mg=16.0)
    assert len(hl_df) == 4
    assert all(hl_df["mean_active_mass_mg"] >= 16.0)
    assert hl_df.iloc[0]["recipe_id"] == "r3"
    assert hl_df.iloc[0]["discharge_specific_capacity_cycle30_mah_g"] == 355.0


def test_filter_candidate_pool_high_loading_exceeds_available_raises_unit(mock_high_loading_pool):
    df, _ = mock_high_loading_pool
    with pytest.raises(ValueError, match="No candidates satisfy high-loading threshold"):
        filter_candidate_pool_for_high_loading(df, min_active_mass_mg=25.0)


def test_production_runner_high_loading_replay_unit(mock_high_loading_pool):
    df, ctrl_cols = mock_high_loading_pool
    runner = ProductionProcessRediscoveryRunner(
        candidate_pool=df,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=ctrl_cols,
    )
    results = runner.run_high_loading(
        min_active_mass_mg=16.0,
        policies=["AICOSCIENTIST_PROCESS_SURROGATE", "random"],
        seeds=[42],
        initial_size=2,
        budget=1,
    )
    assert results["benchmark_task"] == "HIGH_LOADING_D30"
    assert results["candidate_pool_size"] == 4
    trajs = results["trajectories"]["AICOSCIENTIST_PROCESS_SURROGATE"]
    assert len(trajs) == 1
    assert trajs[0]["hidden_best_id"] == "r3"


@pytest.mark.skipif(not HAS_REAL_DATA, reason="Requires local Drakopoulos raw or normalized dataset")
def test_filter_candidate_pool_for_high_loading_16mg():
    df, ctrl_cols = _load_real_pool()
    assert len(df) == 26
    assert df.iloc[0]["recipe_id"] == "protocol-c1c280b7366f"

    hl_df = filter_candidate_pool_for_high_loading(df, min_active_mass_mg=16.0)
    assert len(hl_df) == 11
    assert all(hl_df["mean_active_mass_mg"] >= 16.0)
    assert all(hl_df["coating_gap_um"] == 200.0)

    best_hl = hl_df.sort_values(by="discharge_specific_capacity_cycle30_mah_g", ascending=False).iloc[0]
    assert best_hl["recipe_id"] == "protocol-e025bc31c00a"
    assert pytest.approx(best_hl["discharge_specific_capacity_cycle30_mah_g"], 0.01) == 282.07


@pytest.mark.skipif(not HAS_REAL_DATA, reason="Requires local Drakopoulos raw or normalized dataset")
def test_filter_candidate_pool_high_loading_exceeds_available_raises():
    df, _ = _load_real_pool()
    with pytest.raises(ValueError, match="No candidates satisfy high-loading threshold"):
        filter_candidate_pool_for_high_loading(df, min_active_mass_mg=25.0)


@pytest.mark.skipif(not HAS_REAL_DATA, reason="Requires local Drakopoulos raw or normalized dataset")
def test_high_loading_zero_leakage_controls():
    df, ctrl_cols = _load_real_pool()
    assert "mean_active_mass_mg" not in ctrl_cols
    assert "discharge_specific_capacity_cycle30_mah_g" not in ctrl_cols

    expected_controls = {
        "active_material_fraction_pct",
        "binder_cmc_fraction_pct",
        "binder_sbr_fraction_pct",
        "calendering_applied",
        "coating_gap_um",
        "coating_speed_m_per_min",
        "conductive_additive_fraction_pct",
        "drying_temperature_c",
        "mixing_solids_pct",
    }
    assert set(ctrl_cols) == expected_controls


@pytest.mark.skipif(not HAS_REAL_DATA, reason="Requires local Drakopoulos raw or normalized dataset")
def test_production_runner_high_loading_replay():
    df, ctrl_cols = _load_real_pool()
    runner = ProductionProcessRediscoveryRunner(
        candidate_pool=df,
        candidate_id_column="recipe_id",
        target_column="discharge_specific_capacity_cycle30_mah_g",
        control_columns=ctrl_cols,
    )

    results = runner.run_high_loading(
        min_active_mass_mg=16.0,
        policies=["AICOSCIENTIST_PROCESS_SURROGATE", "random"],
        seeds=[42],
        initial_size=3,
        budget=2,
    )

    assert results["benchmark_task"] == "HIGH_LOADING_D30"
    assert results["candidate_pool_size"] == 11
    trajs = results["trajectories"]["AICOSCIENTIST_PROCESS_SURROGATE"]
    assert len(trajs) == 1
    t = trajs[0]
    assert t["hidden_best_id"] == "protocol-e025bc31c00a"
    assert pytest.approx(t["hidden_best_value"], 0.01) == 282.07
    assert t["engine_path"] == "AICOSCIENTIST_PROCESS_SURROGATE"
    assert len(t["steps"]) == 2
    step1 = t["steps"][0]
    assert step1["engine_path"] == "AICOSCIENTIST_PROCESS_SURROGATE"
    assert step1["surrogate_artifact_fingerprint"] is not None
