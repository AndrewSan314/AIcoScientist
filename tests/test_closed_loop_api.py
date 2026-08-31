from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.legacy.native_optimizer.closed_loop import (
    ClosedLoopOptimizer,
    ExperimentOracle,
    ExperimentProposal,
    ExperimentResult,
)
from src.optimization.search_space import ContinuousVariable, SearchSpace


class MockSimulatedOracle(ExperimentOracle):
    """Simple 2D quadratic sphere oracle: f(x1, x2) = -(x1 - 3)^2 - (x2 - 4)^2 + 100 + noise."""

    def __init__(self, noise_std: float = 0.5, seed: int = 42) -> None:
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def evaluate(self, proposal: ExperimentProposal) -> ExperimentResult:
        x1 = float(proposal.design_variables["x1"])
        x2 = float(proposal.design_variables["x2"])
        latent = -(x1 - 3.0) ** 2 - (x2 - 4.0) ** 2 + 100.0
        noisy_val = latent + self.rng.normal(0.0, self.noise_std)
        return ExperimentResult(
            candidate_id=proposal.candidate_id,
            design_variables=proposal.design_variables,
            target_value=float(noisy_val),
            observations={"latent_lifetime": latent},
        )


def test_closed_loop_optimizer_workflow() -> None:
    space = SearchSpace(
        variables=[
            ContinuousVariable("x1", lower=0.0, upper=10.0),
            ContinuousVariable("x2", lower=0.0, upper=10.0),
        ],
        name="quadratic_space",
    )

    oracle = MockSimulatedOracle(noise_std=0.2, seed=42)

    init_df = pd.DataFrame([
        {"x1": 1.0, "x2": 1.0, "performance": 80.0, "candidate_id": "INIT_0"},
        {"x1": 8.0, "x2": 8.0, "performance": 50.0, "candidate_id": "INIT_1"},
        {"x1": 2.0, "x2": 5.0, "performance": 95.0, "candidate_id": "INIT_2"},
    ])

    optimizer = ClosedLoopOptimizer(
        search_space=space,
        feature_cols=["x1", "x2"],
        target_col="performance",
        strategy="turbo_nei",
        n_candidates_per_step=100,
        random_state=42,
    )

    state = optimizer.initialize(init_df)
    assert state.step == 0
    assert state.current_best == 95.0
    assert state.trust_region is not None
    assert state.trust_region.state.radius == 0.8

    # Propose -> Observe -> Update loop for 5 steps
    for step in range(1, 6):
        proposal = optimizer.propose(state)
        assert isinstance(proposal, ExperimentProposal)
        assert proposal.step == step
        assert "x1" in proposal.design_variables
        assert "x2" in proposal.design_variables
        assert np.isfinite(proposal.predicted_performance)
        assert np.isfinite(proposal.prediction_uncertainty)
        assert proposal.recommendation_reason != ""

        result = oracle.evaluate(proposal)
        assert isinstance(result, ExperimentResult)

        state = optimizer.observe(state, proposal, result)
        assert state.step == step
        assert len(state.observed_records) == 3 + step
        assert len(state.history) == step


def test_closed_loop_firewall_no_latent_oracle_leakage() -> None:
    space = SearchSpace(
        variables=[
            ContinuousVariable("x1", lower=0.0, upper=5.0),
            ContinuousVariable("x2", lower=0.0, upper=5.0),
        ],
        name="test_space",
    )

    init_df = pd.DataFrame([
        {"x1": 1.0, "x2": 1.0, "target": 10.0, "candidate_id": "P0"},
        {"x1": 2.0, "x2": 2.0, "target": 15.0, "candidate_id": "P1"},
    ])

    optimizer = ClosedLoopOptimizer(
        search_space=space,
        feature_cols=["x1", "x2"],
        target_col="target",
        strategy="turbo_nei",
        n_candidates_per_step=50,
        random_state=123,
    )

    state = optimizer.initialize(init_df)
    proposal = optimizer.propose(state)

    proposal_dict = proposal.to_dict()
    # Check that no hidden simulator oracle fields leaked into proposal
    assert "latent_lifetime" not in proposal_dict
    assert "oracle" not in proposal_dict
    assert "reference_true_lifetime" not in proposal_dict
    assert "reason_code" in proposal_dict
    assert "distance_to_nearest_observed" in proposal_dict


def test_closed_loop_state_save_and_load(tmp_path: Path) -> None:
    from pathlib import Path

    space = SearchSpace(
        variables=[
            ContinuousVariable("x1", lower=0.0, upper=10.0),
            ContinuousVariable("x2", lower=0.0, upper=10.0),
        ],
        name="quadratic_space",
    )

    init_df = pd.DataFrame([
        {"x1": 1.0, "x2": 1.0, "performance": 80.0, "candidate_id": "INIT_0"},
        {"x1": 2.0, "x2": 5.0, "performance": 95.0, "candidate_id": "INIT_1"},
    ])

    optimizer = ClosedLoopOptimizer(
        search_space=space,
        feature_cols=["x1", "x2"],
        target_col="performance",
        strategy="turbo_nei",
        n_candidates_per_step=50,
        random_state=42,
    )

    state = optimizer.initialize(init_df)
    proposal = optimizer.propose(state)
    oracle = MockSimulatedOracle(noise_std=0.1, seed=42)
    result = oracle.evaluate(proposal)
    state = optimizer.observe(state, proposal, result)

    save_path = tmp_path / "optimizer_checkpoint.json"
    optimizer.save_state(state, save_path)
    assert save_path.is_file()

    # Load into fresh optimizer instance
    restored_optimizer = ClosedLoopOptimizer(
        search_space=space,
        feature_cols=["x1", "x2"],
        target_col="performance",
        strategy="turbo_nei",
        n_candidates_per_step=50,
        random_state=42,
    )
    restored_state = restored_optimizer.load_state(save_path)

    assert restored_state.step == state.step
    assert restored_state.current_best == state.current_best
    assert len(restored_state.observed_records) == len(state.observed_records)
    assert len(restored_state.history) == len(state.history)

    # Next proposal from restored state should function identically
    next_proposal = restored_optimizer.propose(restored_state)
    assert next_proposal.step == state.step + 1


def test_closed_loop_candidate_id_deterministic_sha256() -> None:
    import hashlib
    space = SearchSpace(
        variables=[
            ContinuousVariable("x1", lower=0.0, upper=10.0),
            ContinuousVariable("x2", lower=0.0, upper=10.0),
        ],
        name="test_space",
    )
    init_df = pd.DataFrame([
        {"x1": 1.0, "x2": 1.0, "performance": 80.0},
        {"x1": 2.0, "x2": 5.0, "performance": 95.0},
    ])
    optimizer = ClosedLoopOptimizer(
        search_space=space,
        feature_cols=["x1", "x2"],
        target_col="performance",
        strategy="random",
        n_candidates_per_step=10,
        random_state=42,
    )
    state = optimizer.initialize(init_df)
    proposal = optimizer.propose(state)

    assert proposal.candidate_id.startswith("EXP_")
    assert len(proposal.candidate_id) == 16

    # Verify deterministic digest calculation
    canonical = "|".join(f"{k}={float(proposal.design_variables[k]):.8f}" for k in sorted(["x1", "x2"]))
    expected_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    assert proposal.candidate_id == f"EXP_{expected_digest}"


def test_closed_loop_duplicate_exhaustion_raises_runtime_error() -> None:
    space = SearchSpace(
        variables=[
            ContinuousVariable("x1", lower=1.0, upper=1.00001),
            ContinuousVariable("x2", lower=1.0, upper=1.00001),
        ],
        name="tiny_space",
    )
    init_df = pd.DataFrame([
        {"x1": 1.0, "x2": 1.0, "performance": 80.0, "candidate_id": "INIT_0"},
    ])
    optimizer = ClosedLoopOptimizer(
        search_space=space,
        feature_cols=["x1", "x2"],
        target_col="performance",
        strategy="random",
        duplicate_tol=0.1,  # Large duplicate tolerance ensures everything is a duplicate
        n_candidates_per_step=10,
        random_state=42,
    )
    state = optimizer.initialize(init_df)
    with pytest.raises(RuntimeError, match="No novel candidates found in search space"):
        optimizer.propose(state)


