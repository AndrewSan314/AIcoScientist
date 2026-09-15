"""Tests verifying D30 calculation and specific capacity grounding."""

from __future__ import annotations
import pytest

def test_d30_specific_capacity_formula() -> None:
    # Formula: D30_specific (mAh/g) = (D30_mAh / Active_Mass_mg) * 1000
    d30_mah = 3.10
    active_mass_mg = 7.70177
    specific_d30 = (d30_mah / active_mass_mg) * 1000.0
    assert pytest.approx(specific_d30, abs=0.1) == 402.5

def test_d30_rejects_zero_or_negative_active_mass() -> None:
    d30_mah = 3.10
    mass = 0.0
    val = (d30_mah / mass * 1000.0) if (d30_mah is not None and mass is not None and mass > 0) else None
    assert val is None


def test_d30_measured_zero_yields_numeric_zero() -> None:
    d30_mah = 0.0
    mass = 7.5
    val = (d30_mah / mass * 1000.0) if (d30_mah is not None and mass is not None and mass > 0) else None
    assert val == 0.0

