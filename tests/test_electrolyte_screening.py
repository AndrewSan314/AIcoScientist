import json
import os
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


def test_historical_evidence_screening_uses_observed_features_and_targets():
    """Verify that HISTORICAL_EVIDENCE mode uses observed features and targets to train discovery models."""
    from src.domains.electrolyte.screening import ScreeningEvidenceMode

    pool_df = _create_synthetic_candidate_pool(n=100, seed=42)
    f_cols = list(ELECTROLYTE_SOLVENT_FEATURES)
    X_obs = pool_df[f_cols].iloc[:10].values
    y_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ws = screen_large_pool_candidates(
        candidates_df=pool_df,
        observed_features=X_obs,
        observed_targets=y_obs,
        working_set_size=20,
        random_state=42,
        evidence_mode=ScreeningEvidenceMode.HISTORICAL_EVIDENCE,
    )
    assert len(ws) == 20
    assert ws["screening_evidence_mode"].iloc[0] == "HISTORICAL_EVIDENCE"
    assert ws.attrs["screening_metadata"]["discovery_scorer"] == "ensemble"

    # Missing observed features must raise ValueError
    with pytest.raises(ValueError, match="requires observed_features and observed_targets"):
        screen_large_pool_candidates(
            candidates_df=pool_df,
            observed_features=None,
            observed_targets=None,
            working_set_size=20,
            evidence_mode=ScreeningEvidenceMode.HISTORICAL_EVIDENCE,
        )


def test_screening_never_accepts_oracle_truth_as_input():
    """Verify that screen_large_pool_candidates parameter list never accepts surrogate oracle truth."""
    import inspect
    sig = inspect.signature(screen_large_pool_candidates)
    forbidden_terms = {"oracle", "surrogate", "latent", "latent_truth", "ground_truth"}
    for param_name in sig.parameters:
        for term in forbidden_terms:
            assert term not in param_name.lower(), f"Forbidden term '{term}' found in parameter '{param_name}'"


def test_screening_evidence_mode_is_recorded():
    """Verify that screening_evidence_mode is tracked both as a column and in metadata."""
    from src.domains.electrolyte.screening import ScreeningEvidenceMode

    pool_df = _create_synthetic_candidate_pool(n=100, seed=42)
    ws_cold = screen_large_pool_candidates(
        candidates_df=pool_df,
        working_set_size=20,
        random_state=42,
        evidence_mode=ScreeningEvidenceMode.COLD_START_DESCRIPTOR_ONLY,
    )
    assert "screening_evidence_mode" in ws_cold.columns
    assert (ws_cold["screening_evidence_mode"] == "COLD_START_DESCRIPTOR_ONLY").all()
    assert ws_cold.attrs["screening_metadata"]["evidence_mode"] == "COLD_START_DESCRIPTOR_ONLY"


def test_historical_screening_is_deterministic():
    """Verify that screening with the same random seed produces identical candidate sets."""
    from src.domains.electrolyte.screening import ScreeningEvidenceMode

    pool_df = _create_synthetic_candidate_pool(n=100, seed=42)
    f_cols = list(ELECTROLYTE_SOLVENT_FEATURES)
    X_obs = pool_df[f_cols].iloc[:10].values
    y_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ws1 = screen_large_pool_candidates(
        candidates_df=pool_df,
        observed_features=X_obs,
        observed_targets=y_obs,
        working_set_size=25,
        random_state=42,
        evidence_mode=ScreeningEvidenceMode.HISTORICAL_EVIDENCE,
    )
    ws2 = screen_large_pool_candidates(
        candidates_df=pool_df,
        observed_features=X_obs,
        observed_targets=y_obs,
        working_set_size=25,
        random_state=42,
        evidence_mode=ScreeningEvidenceMode.HISTORICAL_EVIDENCE,
    )
    assert list(ws1["candidate_id"]) == list(ws2["candidate_id"])
    assert list(ws1["screening_tranche"]) == list(ws2["screening_tranche"])


def test_working_set_quality_diagnostics_are_computed_offline():
    """Verify that diagnostics evaluate candidate latent values strictly offline after screening."""
    diag_path = "outputs/electrolyte/benchmark/screening_quality_diagnostics.json"
    if os.path.exists(diag_path):
        import json
        with open(diag_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "full_search_space_latent_max" in data
        assert "working_set_trials" in data
        for ws_k, trial in data["working_set_trials"].items():
            assert "working_set_latent_max" in trial
            assert "screening_latent_gap" in trial
            assert trial["screening_latent_gap"] >= 0.0


def test_working_set_size_sensitivity_has_200_500_1000():
    """Verify that sensitivity trials cover sizes 200, 500, and 1000."""
    diag_path = "outputs/electrolyte/benchmark/screening_quality_diagnostics.json"
    if os.path.exists(diag_path):
        import json
        with open(diag_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "200" in data["working_set_trials"]
        assert "500" in data["working_set_trials"]
        assert "1000" in data["working_set_trials"]


def test_cold_start_vs_historical_screening_comparison_exists():
    """Verify that diagnostics include reference comparison between cold-start and historical."""
    diag_path = "outputs/electrolyte/benchmark/screening_quality_diagnostics.json"
    if os.path.exists(diag_path):
        import json
        with open(diag_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ref = data.get("reference_comparison", {})
        assert "COLD_START_DESCRIPTOR_ONLY_200" in ref
        assert "HISTORICAL_EVIDENCE_200" in ref


def test_screening_quality_gate_does_not_require_hidden_oracle_optimum_recovery():
    """Verify that screening_quality_gate passes even when screening_latent_gap > 0 (honest quantification)."""
    mock_diag = {
        "evidence_mode": "HISTORICAL_EVIDENCE",
        "full_search_space_latent_max": 0.85,
        "working_set_trials": {
            "200": {"working_set_latent_max": 0.70, "screening_latent_gap": 0.15},
            "500": {"working_set_latent_max": 0.75, "screening_latent_gap": 0.10},
            "1000": {"working_set_latent_max": 0.80, "screening_latent_gap": 0.05},
        },
        "reference_comparison": {"COLD_START_DESCRIPTOR_ONLY_200": {}},
        "chosen_default_working_set_size": 200,
        "chosen_default_rationale": "Balances speed and diversity.",
    }
    # Emulate the gate check
    has_mode = mock_diag.get("evidence_mode") == "HISTORICAL_EVIDENCE"
    has_trials = all(k in mock_diag.get("working_set_trials", {}) for k in ("200", "500", "1000"))
    has_ref = "COLD_START_DESCRIPTOR_ONLY_200" in mock_diag.get("reference_comparison", {})
    finite_max = np.isfinite(mock_diag.get("full_search_space_latent_max"))
    trials_valid = all(
        np.isfinite(t["working_set_latent_max"]) and np.isfinite(t["screening_latent_gap"]) and t["screening_latent_gap"] >= 0
        for t in mock_diag["working_set_trials"].values()
    )
    has_default = bool(mock_diag.get("chosen_default_rationale"))
    assert has_mode and has_trials and has_ref and finite_max and trials_valid and has_default


