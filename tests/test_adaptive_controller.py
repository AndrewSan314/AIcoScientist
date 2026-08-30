from __future__ import annotations

import numpy as np

from src.legacy.native_optimizer.adaptive_controller import AdaptiveBOController


def test_adaptive_controller_mode_switching() -> None:
    controller = AdaptiveBOController()

    # Case 1: Early step (step 1/25) -> should choose gp_ucb with exploration rationale
    dec_early = controller.decide(
        step=1,
        total_queries=25,
        observed_targets=[900.0, 950.0],
        pred_mean=np.array([920.0, 940.0, 960.0]),
        pred_std=np.array([50.0, 40.0, 60.0]),
    )
    assert dec_early.chosen_method == "gp_ucb"
    assert "Early exploration phase" in dec_early.controller_reason
    assert dec_early.beta > 1.0

    # Case 2: High Expected Improvement -> should choose expected_improvement
    dec_ei = controller.decide(
        step=10,
        total_queries=25,
        observed_targets=[900.0, 950.0, 960.0],
        pred_mean=np.array([920.0, 1020.0, 940.0]),  # Strong predicted improvement (1020 vs 960)
        pred_std=np.array([10.0, 25.0, 15.0]),
    )
    assert dec_ei.chosen_method == "expected_improvement"
    assert "High Expected Improvement" in dec_ei.controller_reason

    # Case 3: Late budget (step 24/25) -> should choose greedy
    dec_late = controller.decide(
        step=24,
        total_queries=25,
        observed_targets=[900.0, 950.0, 960.0, 980.0],
        pred_mean=np.array([970.0, 990.0, 985.0]),
        pred_std=np.array([5.0, 4.0, 6.0]),
    )
    assert dec_late.chosen_method == "greedy"
    assert "Late budget phase" in dec_late.controller_reason


def test_adaptive_controller_stopping_signal() -> None:
    controller = AdaptiveBOController()

    # Saturated / depleted uncertainty + prolonged stagnation -> should_stop = True
    dec_stop = controller.decide(
        step=20,
        total_queries=25,
        observed_targets=[900.0, 980.0, 980.0, 980.0, 980.0, 980.0, 980.0, 980.0],  # 6 steps stagnation
        pred_mean=np.array([960.0, 970.0, 975.0]),
        pred_std=np.array([1.0, 2.0, 1.5]),  # Depleted epistemic variance
    )
    assert dec_stop.should_stop is True
    assert "below threshold" in dec_stop.stop_reason

    # Active exploration state -> should_stop = False
    dec_active = controller.decide(
        step=5,
        total_queries=25,
        observed_targets=[900.0, 950.0],
        pred_mean=np.array([920.0, 980.0]),
        pred_std=np.array([30.0, 40.0]),
    )
    assert dec_active.should_stop is False
    assert "Active search warranted" in dec_active.stop_reason


def test_adaptive_controller_deterministic_output() -> None:
    controller = AdaptiveBOController()
    dec_1 = controller.decide(
        step=8,
        total_queries=25,
        observed_targets=[900.0, 950.0, 970.0],
        pred_mean=np.array([950.0, 980.0]),
        pred_std=np.array([20.0, 15.0]),
    )
    dec_2 = controller.decide(
        step=8,
        total_queries=25,
        observed_targets=[900.0, 950.0, 970.0],
        pred_mean=np.array([950.0, 980.0]),
        pred_std=np.array([20.0, 15.0]),
    )
    assert dec_1.chosen_method == dec_2.chosen_method
    assert dec_1.beta == dec_2.beta
    assert dec_1.exploration_score == dec_2.exploration_score
    assert dec_1.exploitation_score == dec_2.exploitation_score
    assert dec_1.controller_reason == dec_2.controller_reason
