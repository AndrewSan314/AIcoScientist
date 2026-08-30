from __future__ import annotations

import numpy as np
import pytest

from src.optimization.search_space import (
    Constraint,
    ContinuousVariable,
    DerivedVariable,
    SearchSpace,
)
from src.optimization.trust_region import TuRBOTrustRegion


@pytest.fixture
def simple_search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            ContinuousVariable("x1", lower=0.0, upper=10.0),
            ContinuousVariable("x2", lower=-5.0, upper=5.0),
        ],
        derived_variables=[
            DerivedVariable("sum_x", compute_fn=lambda c: float(c["x1"]) + float(c["x2"]), depends_on=("x1", "x2")),
        ],
        constraints=[
            Constraint(name="non_negative_sum", predicate=lambda c: float(c.get("sum_x", float(c["x1"]) + float(c["x2"]))) >= 0.0),
        ],
        name="test_space",
    )


def test_trust_region_initialization(simple_search_space: SearchSpace) -> None:
    turbo = TuRBOTrustRegion(search_space=simple_search_space, init_radius=0.8)
    init_cand = {"x1": 5.0, "x2": 0.0, "sum_x": 5.0}
    state = turbo.initialize(init_cand, initial_best_value=100.0)

    assert state.radius == 0.8
    assert state.best_value == 100.0
    assert state.success_counter == 0
    assert state.failure_counter == 0
    assert state.restarts_count == 0
    assert state.center["x1"] == 5.0
    assert state.center["x2"] == 0.0


def test_trust_region_expansion_on_success(simple_search_space: SearchSpace) -> None:
    turbo = TuRBOTrustRegion(search_space=simple_search_space, init_radius=0.5, success_tolerance=3)
    turbo.initialize({"x1": 5.0, "x2": 0.0}, initial_best_value=100.0)

    # 1st success
    u1 = turbo.update({"x1": 5.5, "x2": 0.5}, observed_value=105.0)
    assert not u1["expanded"]
    assert turbo.state.success_counter == 1
    assert turbo.state.radius == 0.5

    # 2nd success
    u2 = turbo.update({"x1": 6.0, "x2": 0.5}, observed_value=110.0)
    assert not u2["expanded"]
    assert turbo.state.success_counter == 2

    # 3rd success -> triggers expansion
    u3 = turbo.update({"x1": 6.5, "x2": 0.5}, observed_value=115.0)
    assert u3["expanded"]
    assert turbo.state.success_counter == 0
    assert np.isclose(turbo.state.radius, 1.0)
    assert turbo.state.expansions_count == 1


def test_trust_region_shrink_on_failure(simple_search_space: SearchSpace) -> None:
    turbo = TuRBOTrustRegion(search_space=simple_search_space, init_radius=0.8, failure_tolerance=3)
    turbo.initialize({"x1": 5.0, "x2": 0.0}, initial_best_value=100.0)

    # 1st failure
    u1 = turbo.update({"x1": 4.0, "x2": -1.0}, observed_value=90.0)
    assert not u1["contracted"]
    assert turbo.state.failure_counter == 1
    assert turbo.state.radius == 0.8

    # 2nd failure
    u2 = turbo.update({"x1": 3.0, "x2": -1.0}, observed_value=85.0)
    assert not u2["contracted"]
    assert turbo.state.failure_counter == 2

    # 3rd failure -> triggers contraction
    u3 = turbo.update({"x1": 2.0, "x2": -1.0}, observed_value=80.0)
    assert u3["contracted"]
    assert turbo.state.failure_counter == 0
    assert np.isclose(turbo.state.radius, 0.4)
    assert turbo.state.contractions_count == 1


def test_trust_region_restart_with_fallback_center(simple_search_space: SearchSpace) -> None:
    turbo = TuRBOTrustRegion(
        search_space=simple_search_space,
        init_radius=0.8,
        min_radius=0.1,
        failure_tolerance=1,
    )
    turbo.initialize({"x1": 5.0, "x2": 0.0}, initial_best_value=100.0)

    # Contract down: 0.8 -> 0.4 -> 0.2 -> 0.1 -> 0.05 (< 0.1 min_radius) -> restart
    turbo.update({"x1": 4.0, "x2": 0.0}, observed_value=90.0)  # 0.4
    turbo.update({"x1": 4.0, "x2": 0.0}, observed_value=90.0)  # 0.2
    turbo.update({"x1": 4.0, "x2": 0.0}, observed_value=90.0)  # 0.1
    u_restart = turbo.update(
        {"x1": 4.0, "x2": 0.0},
        observed_value=90.0,
        fallback_center={"x1": 8.5, "x2": 2.5},
        fallback_candidate_id="GLOBAL_RESTART_42",
    )

    assert u_restart["restarted"]
    assert turbo.state.restarts_count == 1
    assert np.isclose(turbo.state.radius, 0.8)
    assert turbo.state.center["x1"] == 8.5
    assert turbo.state.center["x2"] == 2.5
    assert u_restart["restart_reason"] == "trust_region_contracted_below_min_length"
    assert u_restart["restart_candidate_id"] == "GLOBAL_RESTART_42"


def test_noise_aware_turbo_rejects_noisy_outlier_without_posterior_evidence(simple_search_space: SearchSpace) -> None:
    turbo = TuRBOTrustRegion(
        search_space=simple_search_space,
        init_radius=0.8,
        success_delta=1.0,
        success_probability_threshold=0.6,
    )
    turbo.initialize({"x1": 5.0, "x2": 0.0}, initial_best_value=100.0)

    # A noisy spike of 200.0 observed, but posterior mean is only 95.0 vs incumbent 100.0 with high uncertainty
    u = turbo.update(
        {"x1": 6.0, "x2": 0.0},
        observed_value=200.0,
        posterior_candidate_mean=95.0,
        posterior_incumbent_mean=100.0,
        posterior_candidate_std=10.0,
        posterior_incumbent_std=2.0,
    )

    # Must NOT count as success because posterior evidence is weak
    assert turbo.state.success_counter == 0
    assert turbo.state.failure_counter == 1
    assert turbo.state.center["x1"] == 5.0  # Center remains unchanged


def test_noise_aware_turbo_expands_on_verified_posterior_improvement(simple_search_space: SearchSpace) -> None:
    turbo = TuRBOTrustRegion(
        search_space=simple_search_space,
        init_radius=0.5,
        success_tolerance=2,
        success_delta=1.0,
        success_probability_threshold=0.6,
    )
    turbo.initialize({"x1": 5.0, "x2": 0.0}, initial_best_value=100.0)

    # 1st strong posterior improvement
    u1 = turbo.update(
        {"x1": 5.5, "x2": 0.5},
        observed_value=110.0,
        posterior_candidate_mean=108.0,
        posterior_incumbent_mean=100.0,
        posterior_candidate_std=2.0,
        posterior_incumbent_std=2.0,
    )
    assert turbo.state.success_counter == 1
    assert turbo.state.center["x1"] == 5.5

    # 2nd strong posterior improvement -> triggers expansion
    u2 = turbo.update(
        {"x1": 6.0, "x2": 0.5},
        observed_value=120.0,
        posterior_candidate_mean=118.0,
        posterior_incumbent_mean=108.0,
        posterior_candidate_std=2.0,
        posterior_incumbent_std=2.0,
    )
    assert u2["expanded"]
    assert turbo.state.expansions_count == 1
    assert np.isclose(turbo.state.radius, 1.0)


def test_trust_region_sampling_within_bounds_and_constraints(simple_search_space: SearchSpace) -> None:
    turbo = TuRBOTrustRegion(search_space=simple_search_space, init_radius=0.2)
    turbo.initialize({"x1": 5.0, "x2": 0.0}, initial_best_value=100.0)

    cands = turbo.sample_candidates(n=100, seed=42)
    assert len(cands) == 100

    box = turbo.get_bounding_box()
    for _, row in cands.iterrows():
        # Check within box
        assert box["x1"][0] <= row["x1"] <= box["x1"][1]
        assert box["x2"][0] <= row["x2"] <= box["x2"][1]
        # Check feasible
        assert simple_search_space.is_feasible(row)


def test_trust_region_global_escape_trigger(simple_search_space: SearchSpace) -> None:
    turbo = TuRBOTrustRegion(search_space=simple_search_space, global_escape_frequency=4)
    turbo.initialize({"x1": 5.0, "x2": 0.0}, initial_best_value=100.0)

    assert not turbo.should_global_escape(step=0)
    assert not turbo.should_global_escape(step=1)
    assert not turbo.should_global_escape(step=2)
    assert not turbo.should_global_escape(step=3)
    assert turbo.should_global_escape(step=4)
    assert not turbo.should_global_escape(step=5)
    assert turbo.should_global_escape(step=8)


def test_trust_region_serialization_and_restore(simple_search_space: SearchSpace) -> None:
    from src.optimization.trust_region import TrustRegionState

    turbo = TuRBOTrustRegion(search_space=simple_search_space, init_length=0.6)
    state = turbo.initialize({"x1": 5.0, "x2": 0.0}, initial_best_value=100.0)
    turbo.update({"x1": 6.0, "x2": 1.0}, observed_value=105.0)

    state_dict = state.to_dict()
    restored_state = TrustRegionState.from_dict(state_dict)

    assert restored_state.step == state.step
    assert restored_state.best_value == state.best_value
    assert restored_state.length == state.length
    assert restored_state.center == state.center


def test_trust_region_covariance_aware_success_probability(simple_search_space: SearchSpace) -> None:
    turbo = TuRBOTrustRegion(search_space=simple_search_space, success_delta=0.5, success_probability_threshold=0.6)
    turbo.initialize({"x1": 5.0, "x2": 0.0}, initial_best_value=100.0)

    # Candidate with small mean improvement: delta_mu = 1.0 (delta = 0.5 -> delta_mu - delta = 0.5)
    # Independent variances: var_cand = 4.0, var_inc = 4.0 -> delta_std = sqrt(8.0) ~ 2.828 -> z = 0.5 / 2.828 ~ 0.1768 -> p ~ 0.570 (below 0.60 -> failure)
    u_uncorr = turbo.update(
        {"x1": 5.5, "x2": 0.0},
        observed_value=101.0,
        posterior_candidate_mean=101.0,
        posterior_incumbent_mean=100.0,
        posterior_candidate_variance=4.0,
        posterior_incumbent_variance=4.0,
        posterior_candidate_incumbent_covariance=0.0,
    )
    assert u_uncorr["success_probability"] < 0.60
    assert turbo.state.failure_counter == 1

    # Correlated with positive covariance: cov = 3.5 -> delta_var = 4 + 4 - 2(3.5) = 1.0 -> delta_std = 1.0 -> z = 0.5 / 1.0 = 0.5 -> p = Phi(0.5) ~ 0.691 (above 0.60 -> success!)
    u_corr = turbo.update(
        {"x1": 5.5, "x2": 0.0},
        observed_value=101.0,
        posterior_candidate_mean=101.0,
        posterior_incumbent_mean=100.0,
        posterior_candidate_variance=4.0,
        posterior_incumbent_variance=4.0,
        posterior_candidate_incumbent_covariance=3.5,
    )
    assert u_corr["success_probability"] >= 0.60
    assert turbo.state.success_counter == 1
    assert turbo.state.failure_counter == 0

