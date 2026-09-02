from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.domains.electrolyte.config import (
    ELECTROLYTE_SOLVENT_FEATURES,
    ELECTROLYTE_VIRTUAL_FEATURES,
)

logger = logging.getLogger(__name__)

DEFAULT_CONTRACT_PATH: str = "outputs/electrolyte/audit/electrolyte_data_contract.json"
DEFAULT_COMPATIBLE_DERIVED_PATH: str = "outputs/electrolyte/audit/pool_compatible_deexpanded_outcomes.csv"
DEFAULT_ALL_DEEXPANDED_PATH: str = "outputs/electrolyte/audit/deexpanded_campaign_outcomes.csv"
DEFAULT_VIRTUAL_1M_PATH: str = "data/external/al_anode_free_2025/virtual_search_space_1million.csv"

# Pre-experiment columns allowed through the candidate information firewall
ALLOWED_CANDIDATE_COLUMNS: set[str] = {
    "candidate_id",
    "solv_comb_sm",
    "salt_comb_sm",
    "canonical_salt",
    "conc_salt_1",
    "theor_capacity",
    "amt_electrolyte",
} | set(ELECTROLYTE_VIRTUAL_FEATURES)

# Forbidden post-experiment / target / future columns strictly blocked by the firewall
FORBIDDEN_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "norm_capacity_3",
    "C_norm_20",
    "act_capacity_20",
    "act_capacity_1",
    "batch",
    "historical_outcome_id",
    "de_expansion_status",
    "provenance",
    "test_status",
)


def generate_candidate_id(solv_smiles: str, salt_smiles: str) -> str:
    """Generates a deterministic, unique, collision-resistant candidate ID."""
    key = f"{solv_smiles.strip()}_{salt_smiles.strip()}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"ELEC_{h}"


def load_electrolyte_data_contract(contract_path: str = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Loads the frozen electrolyte data contract."""
    if not os.path.exists(contract_path):
        raise FileNotFoundError(f"Electrolyte data contract not found at {contract_path}. Run audit first.")
    with open(contract_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_derived_historical_outcomes(
    derived_path: str = DEFAULT_COMPATIBLE_DERIVED_PATH,
) -> pd.DataFrame:
    """Loads the row-level derived de-expanded historical outcomes."""
    if not os.path.exists(derived_path):
        raise FileNotFoundError(f"Derived historical outcomes not found at {derived_path}. Run audit first.")
    df = pd.read_csv(derived_path)
    if "candidate_id" not in df.columns:
        df["candidate_id"] = [
            generate_candidate_id(r["solv_comb_sm"], r.get("canonical_salt", r["salt_comb_sm"]))
            for _, r in df.iterrows()
        ]
    return df


def extract_candidate_pool_from_derived(
    df_derived: pd.DataFrame,
    feature_cols: Sequence[str] = ELECTROLYTE_SOLVENT_FEATURES,
) -> pd.DataFrame:
    """Constructs a strictly firewalled candidate pool DataFrame from derived outcomes.

    Guarantees:
    1. Zero ground-truth targets (C_norm_20, norm_capacity_3) in the returned candidate pool.
    2. Zero future batch indicators.
    3. Exactly 1 row per unique candidate_id.
    """
    candidates = []
    seen_ids = set()

    for _, row in df_derived.iterrows():
        cid = row["candidate_id"]
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        cand_rec = {
            "candidate_id": cid,
            "solv_comb_sm": row["solv_comb_sm"],
            "salt_comb_sm": row.get("canonical_salt", row["salt_comb_sm"]),
            "conc_salt_1": float(row.get("conc_salt_1", 1.0)),
            "theor_capacity": float(row.get("theor_capacity", 150.0)),
            "amt_electrolyte": float(row.get("amt_electrolyte", 50.0)),
        }
        for f in feature_cols:
            cand_rec[f] = float(row[f])
        candidates.append(cand_rec)

    pool_df = pd.DataFrame(candidates)

    # Enforce firewall assertions
    for col in FORBIDDEN_CANDIDATE_COLUMNS:
        assert col not in pool_df.columns, f"Information firewall breach: forbidden column '{col}' in candidate pool!"

    return pool_df


def load_lifsi_virtual_candidate_chunk(
    virtual_csv_path: str = DEFAULT_VIRTUAL_1M_PATH,
    nrows: int | None = None,
    chunksize: int = 50000,
    feature_cols: Sequence[str] = ELECTROLYTE_SOLVENT_FEATURES,
) -> pd.DataFrame:
    """Streams and loads candidates from the physically aligned LiFSI virtual slice (~333,333 rows)."""
    if not os.path.exists(virtual_csv_path):
        raise FileNotFoundError(f"Virtual candidate space file not found at {virtual_csv_path}")

    chunks = []
    loaded = 0
    lifsi_smiles = "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F"

    cols_to_read = list(set(["solv_comb_sm", "salt_comb_sm"] + list(feature_cols)))

    for chunk in pd.read_csv(virtual_csv_path, chunksize=chunksize, usecols=cols_to_read):
        lifsi_chunk = chunk[chunk["salt_comb_sm"] == lifsi_smiles].copy()
        if len(lifsi_chunk) > 0:
            lifsi_chunk["candidate_id"] = [
                generate_candidate_id(s, lifsi_smiles) for s in lifsi_chunk["solv_comb_sm"]
            ]
            lifsi_chunk["conc_salt_1"] = 1.0
            lifsi_chunk["theor_capacity"] = 150.0
            lifsi_chunk["amt_electrolyte"] = 50.0
            chunks.append(lifsi_chunk)
            loaded += len(lifsi_chunk)
            if nrows is not None and loaded >= nrows:
                break

    if not chunks:
        return pd.DataFrame()

    full_df = pd.concat(chunks, ignore_index=True)
    if nrows is not None:
        full_df = full_df.iloc[:nrows].copy()

    # Enforce firewall
    for col in FORBIDDEN_CANDIDATE_COLUMNS:
        assert col not in full_df.columns, f"Firewall breach: '{col}' present in virtual pool!"

    return full_df
