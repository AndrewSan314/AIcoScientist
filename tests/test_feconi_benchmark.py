import numpy as np
import pandas as pd
import pytest

from src.datasets.feconi import (
    FECONI_CANDIDATE_ID_COLUMN,
    FeCoNiAdapter,
    FeCoNiExperimentOracle,
    compute_derived_ni,
    load_raw_feconi_mat,
)
from src.evaluation.feconi_benchmark import run_single_aicoscientist_trajectory
from src.evaluation.feconi_reproduction_benchmark import run_single_reproduction_trajectory


def test_specification_1_mat_loader_exact_921_rows():
    raw = load_raw_feconi_mat()
    assert raw["C"].shape == (921, 3)
    assert raw["XRD"].shape == (921, 89)
    assert len(raw["Coer"]) == 921
    assert len(raw["Kerr"]) == 921


def test_specification_2_composition_sums_approximate_100():
    raw = load_raw_feconi_mat()
    sums = np.sum(raw["C"], axis=1)
    assert np.all(sums >= 99.8) and np.all(sums <= 100.2)


def test_specification_3_candidate_ids_are_unique():
    adapter = FeCoNiAdapter()
    pool = adapter.get_candidate_pool()
    assert len(pool["candidate_id"].unique()) == 921


def test_specification_4_oracle_refuses_unknown_candidate_ids():
    adapter = FeCoNiAdapter(target="Kerr")
    oracle = adapter.create_oracle()
    with pytest.raises(KeyError):
        oracle.query("NON_EXISTENT_ID")


def test_specification_5_oracle_refuses_duplicate_reveal():
    adapter = FeCoNiAdapter(target="Kerr")
    oracle = adapter.create_oracle(allow_duplicate_queries=False)
    oracle.query("FECONI_010")
    with pytest.raises(ValueError, match="Duplicate experimental measurement"):
        oracle.query("FECONI_010")


def test_specification_6_and_7_hidden_targets_and_xrd_never_in_candidate_input():
    adapter = FeCoNiAdapter(target="Kerr")
    pool = adapter.get_candidate_pool()
    for col in ["Kerr", "Coer", "XRD", "TTH"]:
        assert col not in pool.columns


def test_specification_8_and_9_selected_candidate_strictly_in_921_pool():
    adapter = FeCoNiAdapter(target="Kerr")
    pool = adapter.get_candidate_pool()
    oracle = adapter.create_oracle()

    valid_cids = set(pool["candidate_id"])

    # Run short trajectory
    traj = run_single_reproduction_trajectory(
        candidate_pool=pool,
        oracle=oracle,
        target_name="Kerr",
        strategy="gp_ucb",
        init_sample_index=5,
        total_budget=5,
        seed=123,
    )

    for step in traj:
        cid = step["selected_sample_id"]
        assert cid in valid_cids
        assert 0 <= step["sample_index"] < 921


def test_specification_10_and_11_deterministic_and_identical_initialization():
    adapter = FeCoNiAdapter(target="Coer")
    pool = adapter.get_candidate_pool()
    oracle = adapter.create_oracle()

    seed = 77
    rng = np.random.default_rng(seed)
    init_idx = int(rng.integers(0, len(pool)))

    traj_random = run_single_reproduction_trajectory(
        candidate_pool=pool, oracle=oracle, target_name="Coer", strategy="random", init_sample_index=init_idx, total_budget=3, seed=seed
    )
    traj_greedy = run_single_reproduction_trajectory(
        candidate_pool=pool, oracle=oracle, target_name="Coer", strategy="greedy", init_sample_index=init_idx, total_budget=3, seed=seed
    )
    traj_ei = run_single_reproduction_trajectory(
        candidate_pool=pool, oracle=oracle, target_name="Coer", strategy="expected_improvement", init_sample_index=init_idx, total_budget=3, seed=seed
    )

    # Initial sample is identical across all strategies
    assert traj_random[0]["selected_sample_id"] == traj_greedy[0]["selected_sample_id"] == traj_ei[0]["selected_sample_id"]
    assert traj_random[0]["observed_target"] == traj_greedy[0]["observed_target"] == traj_ei[0]["observed_target"]


def test_specification_12_target_selection_kerr_vs_coer_independent():
    kerr_oracle = FeCoNiAdapter(target="Kerr").create_oracle()
    coer_oracle = FeCoNiAdapter(target="Coer").create_oracle()

    assert kerr_oracle.target_column == "Kerr"
    assert coer_oracle.target_column == "Coer"
    assert kerr_oracle.global_best_candidate_id != coer_oracle.global_best_candidate_id


def test_specification_13_global_optimum_used_only_for_evaluation_metrics():
    adapter = FeCoNiAdapter(target="Kerr")
    pool = adapter.get_candidate_pool()
    oracle = adapter.create_oracle()

    traj = run_single_reproduction_trajectory(
        candidate_pool=pool, oracle=oracle, target_name="Kerr", strategy="greedy", init_sample_index=10, total_budget=4, seed=42
    )
    for step in traj:
        assert step["regret"] == step["global_best"] - step["best_observed"]
        assert step["regret"] >= 0.0


def test_specification_14_no_duplicate_selection_within_run():
    adapter = FeCoNiAdapter(target="Coer")
    pool = adapter.get_candidate_pool()
    oracle = adapter.create_oracle()

    traj = run_single_aicoscientist_trajectory(
        candidate_pool=pool, oracle=oracle, target_name="Coer", strategy="turbo_nei", init_sample_index=20, total_budget=10, seed=99
    )
    selected_cids = [s["selected_sample_id"] for s in traj]
    assert len(selected_cids) == len(set(selected_cids))


def test_specification_15_derived_composition_reconstructs_accurately():
    adapter = FeCoNiAdapter(target="Kerr")
    df = adapter.load_data()
    for _, row in df.iterrows():
        reconstructed_ni = compute_derived_ni(row["Co"], row["Fe"])
        assert abs(reconstructed_ni - row["Ni"]) < 0.2
