from __future__ import annotations

import pandas as pd
import pytest

from src.science.coordinator import ScientificClosedLoopCoordinator
from src.science.synthetic import SyntheticExperimentOracle, SyntheticScienceAdapter


def test_critical_future_characterization_leakage_invariance() -> None:
    """CRITICAL TEST: Verifies that changing future post-experiment characterization physics

    or oracle characterization functions before experiment proposal has ZERO effect
    on the proposed candidate, process parameters, or scientific rationale.
    """
    seed = 42
    adapter = SyntheticScienceAdapter()

    # Initial history is identical for both runs
    init_df = adapter.load_initial_dataset(n_samples=10, seed=seed)
    cand_pool = adapter.candidate_space(observed=init_df, n_candidates=50, seed=seed)

    # Coordinator 1: Standard oracle (z1 = sin(x1) + 0.05*x2)
    oracle_standard = SyntheticExperimentOracle(char_noise=0.05)
    coord_1 = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        strategy="expected_improvement",
        random_state=seed,
    )

    # Coordinator 2: RADICALLY ALTERED future characterization physics (e.g. z1 = 1000 * cos(x1))
    class RadicallyAlteredOracle(SyntheticExperimentOracle):
        def evaluate_characterization(self, candidate, seed=None):
            return {"z1": 99999.0, "z2": -99999.0}

    coord_2 = ScientificClosedLoopCoordinator.initialize_new(
        spec=adapter.spec,
        two_stage_spec=adapter.two_stage_spec,
        initial_data=init_df,
        candidate_pool=cand_pool,
        strategy="expected_improvement",
        random_state=seed,
    )

    # Step 1 proposal must be EXACTLY IDENTICAL
    rec_1, rat_1 = coord_1.propose_next(n_mc_samples=32)
    rec_2, rat_2 = coord_2.propose_next(n_mc_samples=32)

    assert rec_1.candidate_id == rec_2.candidate_id
    assert rec_1.pre_experiment_features == rec_2.pre_experiment_features
    assert rat_1.predicted_performance_mean == pytest.approx(rat_2.predicted_performance_mean, rel=1e-5)
    assert rat_1.acquisition_score == pytest.approx(rat_2.acquisition_score, rel=1e-5)
    assert rat_1.expected_learning_value == pytest.approx(rat_2.expected_learning_value, rel=1e-5)
