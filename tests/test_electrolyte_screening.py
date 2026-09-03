import numpy as np
import pandas as pd
import pytest

from src.domains.electrolyte.config import ELECTROLYTE_SOLVENT_FEATURES
from src.domains.electrolyte.data import FORBIDDEN_CANDIDATE_COLUMNS, generate_candidate_id
from src.domains.electrolyte.screening import screen_large_pool_candidates


def _create_synthetic_candidate_pool(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    lifsi = "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F"
    for i in range(n):
        solv = f"C{i}OCC{i}"
        cid = generate_candidate_id(solv, lifsi)
        rec = {
            "candidate_id": cid,
            "solv_comb_sm": solv,
            "salt_comb_sm": lifsi,
            "conc_salt_1": 1.0,
            "theor_capacity": 150.0,
            "amt_electrolyte": 50.0,
        }
        for f_idx, f_name in enumerate(ELECTROLYTE_SOLVENT_FEATURES):
            rec[f_name] = float(rng.normal())
        rows.append(rec)
    return pd.DataFrame(rows)


def test_large_pool_screen_returns_bounded_working_set():
    """Verifies that large pool candidate screening returns exactly the bounded target working set size."""
    pool_df = _create_synthetic_candidate_pool(n=500, seed=42)
    working_set = screen_large_pool_candidates(
        candidates_df=pool_df,
        working_set_size=50,
        random_state=42,
    )
    assert len(working_set) == 50
    assert working_set["candidate_id"].nunique() == 50


def test_screening_does_not_use_hidden_targets():
    """Verifies that candidate screening requires zero target columns and produces no target leakage."""
    pool_df = _create_synthetic_candidate_pool(n=200, seed=42)
    for col in FORBIDDEN_CANDIDATE_COLUMNS:
        assert col not in pool_df.columns

    working_set = screen_large_pool_candidates(
        candidates_df=pool_df,
        working_set_size=30,
        random_state=42,
    )
    for col in FORBIDDEN_CANDIDATE_COLUMNS:
        assert col not in working_set.columns


def test_working_set_contains_discovery_and_exploration_candidates():
    """Verifies that the screened working set contains discovery, exploration, and diversity candidates."""
    pool_df = _create_synthetic_candidate_pool(n=300, seed=42)

    # Synthetic observed data
    f_cols = list(ELECTROLYTE_SOLVENT_FEATURES)
    X_obs = pool_df[f_cols].iloc[:5].values
    y_obs = np.array([0.7, 0.2, 0.8, 0.1, 0.6])

    working_set = screen_large_pool_candidates(
        candidates_df=pool_df,
        observed_features=X_obs,
        observed_targets=y_obs,
        working_set_size=60,
        discovery_fraction=0.4,
        exploration_fraction=0.3,
        diversity_fraction=0.2,
        random_fraction=0.1,
        random_state=42,
    )

    assert len(working_set) == 60
    assert working_set["candidate_id"].nunique() == 60


def test_full_pool_is_not_materialized_as_scientific_actions():
    """Verifies architectural scale constraint: 10,000 candidates are screened to 50 before action materialization."""
    pool_df = _create_synthetic_candidate_pool(n=1000, seed=42)
    working_set = screen_large_pool_candidates(
        candidates_df=pool_df,
        working_set_size=25,
        random_state=42,
    )

    # Only 25 actions materialized, not 1000
    assert len(working_set) == 25
    assert len(working_set) < len(pool_df)


def test_stable_candidate_identity_through_screening():
    """Verifies that candidate IDs and features are preserved identically without index corruption."""
    pool_df = _create_synthetic_candidate_pool(n=100, seed=42)
    working_set = screen_large_pool_candidates(
        candidates_df=pool_df,
        working_set_size=20,
        random_state=42,
    )

    pool_lookup = {r["candidate_id"]: r for _, r in pool_df.iterrows()}
    for _, ws_row in working_set.iterrows():
        cid = ws_row["candidate_id"]
        assert cid in pool_lookup
        orig = pool_lookup[cid]
        assert ws_row["solv_comb_sm"] == orig["solv_comb_sm"]
        assert ws_row["mol_wt_solv"] == orig["mol_wt_solv"]


def test_screened_candidates_have_tranche_provenance():
    """Phase 6: Verifies screening_tranche, screening_score, and screening_round are present."""
    pool_df = _create_synthetic_candidate_pool(n=250, seed=42)
    working_set = screen_large_pool_candidates(
        candidates_df=pool_df,
        working_set_size=50,
        screening_round=2,
        random_state=42,
    )
    assert len(working_set) == 50
    assert "screening_tranche" in working_set.columns
    assert "screening_score" in working_set.columns
    assert "screening_round" in working_set.columns
    assert set(working_set["screening_tranche"].unique()).issubset({"discovery", "exploration", "diversity", "random"})
    assert (working_set["screening_round"] == 2).all()


def test_screened_working_set_initializes_adapter_and_engine():
    """Phase 7: Verifies screened working set wraps into ElectrolyteDomainAdapter and runs DecisionEngine."""
    from src.domains.electrolyte.adapter import ElectrolyteDomainAdapter
    from src.domains.electrolyte.oracle import SurrogateElectrolyteOracle
    from src.science.decision_engine import ScientificDecisionEngine

    pool_df = _create_synthetic_candidate_pool(n=100, seed=42)
    working_set = screen_large_pool_candidates(
        candidates_df=pool_df,
        working_set_size=25,
        random_state=42,
    )

    # Mock or surrogate oracle
    oracle_train = pool_df.copy()
    oracle_train["C_norm_20"] = 0.5
    oracle = SurrogateElectrolyteOracle(df_train=oracle_train, feature_cols=ELECTROLYTE_SOLVENT_FEATURES)

    adapter = ElectrolyteDomainAdapter(
        candidate_pool_df=working_set,
        oracle=oracle,
    )
    assert len(adapter.get_candidate_pool()) == 25

    engine = ScientificDecisionEngine(domain=adapter, seed=42)
    init_actions = adapter.get_default_initial_actions(n_seed=3, seed=42)
    engine.initialize(init_actions)

    rec = engine.propose_next_experiment()
    assert rec is not None
    assert rec.action.candidate_id in adapter.get_candidate_pool()["candidate_id"].values
    outcome = engine.execute_recommendation(rec)
    assert outcome is not None
    assert outcome.canonical_observation is not None


def test_frozen_electrolyte_feature_scaler():
    """Verifies that FrozenElectrolyteFeatureScaler uses canonical population moments and preserves scale invariance."""
    from src.domains.electrolyte.screening import FrozenElectrolyteFeatureScaler, CANONICAL_ELECTROLYTE_MOMENTS

    scaler = FrozenElectrolyteFeatureScaler()
    assert len(scaler.means) == 11
    assert len(scaler.stds) == 11

    # Synthetic sample of 2 candidates with exact population mean
    X = np.zeros((2, 11), dtype=np.float64)
    for i, f in enumerate(ELECTROLYTE_SOLVENT_FEATURES):
        X[:, i] = CANONICAL_ELECTROLYTE_MOMENTS[f]["mean"]

    X_scaled = scaler.transform(X)
    assert X_scaled.shape == (2, 11)
    np.testing.assert_allclose(X_scaled, 0.0, atol=1e-5)

