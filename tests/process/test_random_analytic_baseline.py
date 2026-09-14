"""Tests verifying hypergeometric closed-form baseline calculations."""

from __future__ import annotations
import pytest
from src.process.benchmarks.rediscovery import calculate_hypergeometric_baseline

def test_hypergeometric_curve_properties() -> None:
    curve = calculate_hypergeometric_baseline(
        total_candidates=26,
        initial_size=3,
        budget=10,
        top_k=1,
    )
    # At step 1: 1 / 23 = ~0.04348
    assert pytest.approx(curve[1], abs=1e-4) == 1.0 / 23.0
    # At step 5: 5 / 23 = ~0.21739
    assert pytest.approx(curve[5], abs=1e-4) == 5.0 / 23.0
    # At step 10: 10 / 23 = ~0.43478
    assert pytest.approx(curve[10], abs=1e-4) == 10.0 / 23.0
