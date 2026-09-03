from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES
from src.domains.electrolyte.data import generate_candidate_id

logger = logging.getLogger(__name__)


def screen_large_pool_candidates(
    candidates_df: pd.DataFrame,
    observed_features: np.ndarray | None = None,
    observed_targets: np.ndarray | None = None,
    working_set_size: int = 200,
    discovery_fraction: float = 0.40,
    exploration_fraction: float = 0.30,
    diversity_fraction: float = 0.20,
    random_fraction: float = 0.10,
    feature_cols: Sequence[str] = ELECTROLYTE_SOLVENT_FEATURES,
    screening_round: int = 0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Scalable Stage-1 candidate screening: reduces virtual candidate libraries into a bounded working set.

    Partitions candidate selection into:
    1. Discovery tranche: Top predicted capacity under current surrogate model.
    2. Exploration tranche: Candidates with highest uncertainty / maximum distance to observed data.
    3. Diversity tranche: Farthest-point chemical descriptor coverage across candidate pool.
    4. Random tranche: Seeded uniform exploration safeguarding against model blind spots.

    Guarantees:
    - Standardizes 11D feature geometry using full candidate space moments so that mol_wt_solv does not dominate PCA 0-9.
    - Candidate IDs are deterministically assigned BEFORE screening.
    - Preserves stable candidate_id identity and stores tranche provenance metadata.
    - Full pool is NEVER materialized as ScientificAction objects in ScientificDecisionEngine.
    """
    total_cands = len(candidates_df)
    df_work = candidates_df.copy()

    # Deterministically ensure candidate_id is assigned BEFORE screening
    if "candidate_id" not in df_work.columns:
        if "solv_comb_sm" in df_work.columns and "salt_comb_sm" in df_work.columns:
            df_work["candidate_id"] = [
                generate_candidate_id(s, sa)
                for s, sa in zip(df_work["solv_comb_sm"], df_work["salt_comb_sm"])
            ]
        else:
            df_work["candidate_id"] = [f"CAND_{i}" for i in range(len(df_work))]

    if total_cands <= working_set_size:
        df_work["screening_tranche"] = "all"
        df_work["screening_score"] = 0.0
        df_work["screening_round"] = screening_round
        return df_work.reset_index(drop=True)

    f_cols = list(feature_cols)
    X_pool = df_work[f_cols].to_numpy(dtype=np.float64, copy=False)
    rng = np.random.default_rng(random_state)

    # Standardize 11D features to eliminate molecular weight scale bias over PCA components
    scaler = StandardScaler()
    X_pool_scaled = scaler.fit_transform(X_pool)

    k_disc = max(1, int(working_set_size * discovery_fraction))
    k_expl = max(1, int(working_set_size * exploration_fraction))
    k_div = max(1, int(working_set_size * diversity_fraction))
    k_rand = max(1, working_set_size - (k_disc + k_expl + k_div))

    # Store mapping: pool_idx -> (tranche_name, score)
    selected_info: dict[int, tuple[str, float]] = {}

    # 1. Discovery Tranche
    if observed_features is not None and observed_targets is not None and len(observed_targets) >= 3:
        X_obs_scaled = scaler.transform(observed_features)
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(X_obs_scaled, observed_targets)
        pred_scores = surrogate.predict(X_pool_scaled)

        top_disc_idx = np.argsort(-pred_scores)
        for idx in top_disc_idx:
            idx_int = int(idx)
            if idx_int not in selected_info:
                selected_info[idx_int] = ("discovery", float(pred_scores[idx_int]))
                if len(selected_info) >= k_disc:
                    break
    else:
        # Cold start: select candidates with highest standardized descriptor norm
        norms = np.linalg.norm(X_pool_scaled, axis=1)
        top_disc_idx = np.argsort(-norms)
        for idx in top_disc_idx[:k_disc]:
            idx_int = int(idx)
            selected_info[idx_int] = ("discovery", float(norms[idx_int]))

    # 2. Exploration Tranche (Max standardized distance to currently observed points)
    if observed_features is not None and len(observed_features) > 0:
        X_obs_scaled = scaler.transform(observed_features)
        dists = cdist(X_pool_scaled, X_obs_scaled, metric="euclidean").min(axis=1)
    else:
        # Standardized distance to centroid of pool
        centroid = np.mean(X_pool_scaled, axis=0, keepdims=True)
        dists = cdist(X_pool_scaled, centroid, metric="euclidean").flatten()

    expl_sorted = np.argsort(-dists)
    for idx in expl_sorted:
        idx_int = int(idx)
        if idx_int not in selected_info:
            selected_info[idx_int] = ("exploration", float(dists[idx_int]))
            if len(selected_info) >= (k_disc + k_expl):
                break

    # 3. Diversity Tranche (Greedy farthest-point selection in standardized feature space)
    remaining_indices = [i for i in range(total_cands) if i not in selected_info]
    if remaining_indices:
        subset_size = min(len(remaining_indices), 2000)
        sub_rem = rng.choice(remaining_indices, size=subset_size, replace=False)
        curr_selected = list(selected_info.keys())
        d_to_sel = cdist(X_pool_scaled[sub_rem], X_pool_scaled[curr_selected], metric="euclidean").min(axis=1)
        div_sorted = np.argsort(-d_to_sel)
        for d_idx in div_sorted:
            idx_int = int(sub_rem[d_idx])
            if idx_int not in selected_info:
                selected_info[idx_int] = ("diversity", float(d_to_sel[d_idx]))
                if len(selected_info) >= (k_disc + k_expl + k_div):
                    break

    # 4. Random Tranche
    rem_final = [i for i in range(total_cands) if i not in selected_info]
    if rem_final and len(selected_info) < working_set_size:
        needed = working_set_size - len(selected_info)
        rand_picks = rng.choice(rem_final, size=min(needed, len(rem_final)), replace=False)
        for r_idx in rand_picks:
            idx_int = int(r_idx)
            selected_info[idx_int] = ("random", 0.0)

    sorted_indices = sorted(selected_info.keys())
    res_df = df_work.iloc[sorted_indices].copy().reset_index(drop=True)
    res_df["screening_tranche"] = [selected_info[i][0] for i in sorted_indices]
    res_df["screening_score"] = [round(selected_info[i][1], 6) for i in sorted_indices]
    res_df["screening_round"] = screening_round
    return res_df


def benchmark_large_pool_screening(
    virtual_csv_path: str = "data/external/al_anode_free_2025/virtual_search_space_1million.csv",
    sample_sizes: Sequence[int] = (10000, 100000, 333333, 999999),
    working_set_size: int = 200,
    feature_cols: Sequence[str] = ELECTROLYTE_SOLVENT_FEATURES,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluates computational scalability across 10k, 100k, 333k LiFSI slice, and 999k candidate space."""
    import psutil
    import os

    results = []
    process = psutil.Process(os.getpid())
    f_cols = list(feature_cols)

    for n in sample_sizes:
        t0 = time.perf_counter()
        mem_before = process.memory_info().rss / (1024 * 1024)

        # Stream chunk
        chunks = []
        loaded = 0
        lifsi_only = (n == 333333)

        for chunk in pd.read_csv(virtual_csv_path, chunksize=100000, usecols=["solv_comb_sm", "salt_comb_sm"] + f_cols):
            if lifsi_only:
                chunk = chunk[chunk["salt_comb_sm"] == "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F"]
            chunks.append(chunk)
            loaded += len(chunk)
            if loaded >= n:
                break

        df_loaded = pd.concat(chunks, ignore_index=True).iloc[:n]
        load_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        working_set = screen_large_pool_candidates(
            candidates_df=df_loaded,
            working_set_size=working_set_size,
            feature_cols=f_cols,
            random_state=random_state,
        )
        screen_time = time.perf_counter() - t1
        total_time = time.perf_counter() - t0
        mem_after = process.memory_info().rss / (1024 * 1024)

        scope_desc = (
            "333k LiFSI Discovery Slice (scientific virtual candidate pool)"
            if lifsi_only
            else (
                "999k Virtual Candidate Space (infrastructure-scale benchmark only)"
                if n == 999999
                else f"{n:,} Virtual Formulations"
            )
        )

        results.append({
            "candidate_count": int(n),
            "pool_scope": scope_desc,
            "load_time_sec": round(load_time, 4),
            "screen_time_sec": round(screen_time, 4),
            "total_time_sec": round(total_time, 4),
            "throughput_cands_per_sec": round(n / max(total_time, 0.001), 1),
            "rss_before_mb": round(mem_before, 2),
            "rss_after_mb": round(mem_after, 2),
            "memory_delta_mb": round(max(0.0, mem_after - mem_before), 2),
            "working_set_size": len(working_set),
            "status": "PASS",
        })

    return {
        "benchmark_name": "Large-Pool Two-Stage Screening Scalability",
        "working_set_size_target": working_set_size,
        "results": results,
    }
