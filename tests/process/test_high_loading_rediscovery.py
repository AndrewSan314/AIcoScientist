import pytest
import pandas as pd
import numpy as np

from src.datasets.battery_process.drakopoulos_graphite import DrakopoulosGraphiteAdapter
from src.process.benchmarks.rediscovery import (
    filter_candidate_pool_for_high_loading,
    run_rediscovery_benchmark,
    ProductionProcessRediscoveryRunner,
)


def _load_pool():
    adapter = DrakopoulosGraphiteAdapter()
    groups = adapter.load_recipe_groups("PROSPECTIVE_MODEL_VALIDATION")
    valid_groups = [g for g in groups if g.mean_d30_specific_capacity > 0]
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


def test_filter_candidate_pool_for_high_loading_16mg():
    df, ctrl_cols = _load_pool()
    assert len(df) == 26
    # Low-loading champion is protocol-d3602183e567 (~402 mAh/g, ~7.7 mg mass)
    assert df.iloc[0]["recipe_id"] == "protocol-d3602183e567"

    hl_df = filter_candidate_pool_for_high_loading(df, min_active_mass_mg=16.0)
    # Exactly 11 recipes in the high-loading regime (gap=200 um, mass 16.8-19.2 mg)
    assert len(hl_df) == 11
    assert all(hl_df["mean_active_mass_mg"] >= 16.0)
    assert all(hl_df["coating_gap_um"] == 200.0)

    # High-loading champion must be protocol-e025bc31c00a (282.07 mAh/g)
    best_hl = hl_df.sort_values(by="discharge_specific_capacity_cycle30_mah_g", ascending=False).iloc[0]
    assert best_hl["recipe_id"] == "protocol-e025bc31c00a"
    assert pytest.approx(best_hl["discharge_specific_capacity_cycle30_mah_g"], 0.01) == 282.07


def test_filter_candidate_pool_high_loading_exceeds_available_raises():
    df, _ = _load_pool()
    # ASC 300 um cells have active mass >= 20-32 mg, but all have missing D30 cycling data.
    # Therefore, threshold >= 25 mg must fail closed with informative ValueError.
    with pytest.raises(ValueError, match="No candidates satisfy high-loading threshold"):
        filter_candidate_pool_for_high_loading(df, min_active_mass_mg=25.0)


def test_high_loading_zero_leakage_controls():
    df, ctrl_cols = _load_pool()
    hl_df = filter_candidate_pool_for_high_loading(df, min_active_mass_mg=16.0)

    # mean_active_mass_mg is strictly an outcome metrology, not a pre-manufacturing control
    assert "mean_active_mass_mg" not in ctrl_cols
    assert "discharge_specific_capacity_cycle30_mah_g" not in ctrl_cols

    # Pre-manufacturing controls only
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


def test_production_runner_high_loading_replay():
    df, ctrl_cols = _load_pool()
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
