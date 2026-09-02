from __future__ import annotations

from unittest.mock import MagicMock
import pytest
import pandas as pd
import numpy as np

from src.science.actions import ScientificAction, ExperimentOutcome
from src.domains.alab.adapter import ALabDomainAdapter
from src.science.domain import ModalityDefinition, ObjectiveDefinition, ObjectiveDirection
from scripts.run_alab_validation import run_single_simulation, run_multi_policy_benchmark


@pytest.fixture
def alab_fixture_adapter(tmp_path):
    fixture_dir = "tests/fixtures/alab"
    return ALabDomainAdapter(data_dir=fixture_dir, cache_dir=str(tmp_path / "cache"))


def test_discovery_metric_not_attributed_to_policy_when_bootstrap_already_crossed_threshold(alab_fixture_adapter):
    """Verifies that if bootstrap reaches threshold (utility >= 0.8), first_autonomous_threshold_cost is None."""
    res = run_single_simulation(alab_fixture_adapter, "PURE_FALSIFICATION", seed=42, budget=15.0)

    assert "bootstrap_threshold_reached" in res
    assert "threshold_already_reached_in_bootstrap" in res
    assert "first_autonomous_threshold_cost" in res

    if res["bootstrap_threshold_reached"]:
        # Discovery must NOT be credited as a policy achievement
        assert res["threshold_already_reached_in_bootstrap"] is True
        assert res["first_autonomous_threshold_cost"] is None
        assert res["bootstrap_best_utility"] >= 0.8


def test_autonomous_improvement_metrics_are_separate_from_bootstrap(alab_fixture_adapter):
    """Verifies that autonomous phase metrics strictly isolate post-bootstrap actions."""
    res = run_single_simulation(alab_fixture_adapter, "PURE_FALSIFICATION", seed=42, budget=15.0)

    assert res["bootstrap_cost"] == 12.0
    assert res["bootstrap_objective_observations"] == 4
    assert res["autonomous_cost"] == res["total_cost"] - res["bootstrap_cost"]
    assert res["autonomous_steps"] == len(res["steps"])

    # Sum of counts in autonomous_action_counts equals autonomous_steps
    assert sum(res["autonomous_action_counts"].values()) == res["autonomous_steps"]

    if res["autonomous_best_utility"] is not None:
        if res["autonomous_best_utility"] > res["bootstrap_best_utility"]:
            assert res["autonomous_improved_over_bootstrap"] is True
            assert res["autonomous_improvement_amount"] > 0.0
        else:
            assert res["autonomous_improved_over_bootstrap"] is False
            assert res["autonomous_improvement_amount"] == 0.0
    else:
        assert res["autonomous_improved_over_bootstrap"] is False
        assert res["autonomous_improvement_amount"] == 0.0


def test_first_autonomous_threshold_cost_tracked_when_bootstrap_below_threshold(alab_fixture_adapter):
    """Verifies that first_autonomous_threshold_cost tracks autonomous cost when bootstrap is below threshold."""
    orig_reveal = alab_fixture_adapter.execute_or_reveal
    call_count = [0]

    import dataclasses

    def mock_reveal(action):
        res = orig_reveal(action)
        call_count[0] += 1
        if action.action_type == "OUTCOME_TEST":
            # For bootstrap objective actions, cap utility at 0.5 (< 0.8)
            if call_count[0] <= 8:
                return dataclasses.replace(res, canonical_observation=0.5)
            else:
                return dataclasses.replace(res, canonical_observation=1.0)
        return res

    alab_fixture_adapter.execute_or_reveal = mock_reveal
    res = run_single_simulation(alab_fixture_adapter, "DISCOVERY_ONLY", seed=42, budget=16.0)

    assert res["bootstrap_threshold_reached"] is False
    assert res["threshold_already_reached_in_bootstrap"] is False
    assert res["bootstrap_best_utility"] == 0.5
    assert res["first_autonomous_threshold_cost"] == 2.0
    assert res["autonomous_best_utility"] == 1.0
    assert res["autonomous_improved_over_bootstrap"] is True
    assert abs(res["autonomous_improvement_amount"] - 0.5) < 1e-4
