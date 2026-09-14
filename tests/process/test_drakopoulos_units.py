"""Tests verifying physical units and ranges for Drakopoulos dataset."""

from __future__ import annotations
import pytest
from src.datasets.battery_process.drakopoulos_graphite import (
    PHYSICAL_SANITY_BOUNDS,
    _validate_sanity_bounds,
    SourceSemanticValidationError,
)

def test_physical_sanity_bounds_defined() -> None:
    assert "coating_speed_m_per_min" in PHYSICAL_SANITY_BOUNDS
    assert "coating_gap_um" in PHYSICAL_SANITY_BOUNDS
    assert "drying_temperature_c" in PHYSICAL_SANITY_BOUNDS
    assert "active_material_fraction_pct" in PHYSICAL_SANITY_BOUNDS

def test_sanity_validation_rejects_unphysical_speed() -> None:
    with pytest.raises(SourceSemanticValidationError):
        _validate_sanity_bounds("coating_speed_m_per_min", 120.0, "test_ctx")

def test_sanity_validation_rejects_unphysical_gap() -> None:
    with pytest.raises(SourceSemanticValidationError):
        _validate_sanity_bounds("coating_gap_um", 0.3, "test_ctx")

def test_sanity_validation_rejects_unphysical_active_material() -> None:
    with pytest.raises(SourceSemanticValidationError):
        _validate_sanity_bounds("active_material_fraction_pct", 300.0, "test_ctx")

def test_sanity_validation_accepts_physical_values() -> None:
    # These should not raise
    _validate_sanity_bounds("coating_speed_m_per_min", 0.20, "test_ctx")
    _validate_sanity_bounds("coating_gap_um", 100.0, "test_ctx")
    _validate_sanity_bounds("drying_temperature_c", 60.0, "test_ctx")
    _validate_sanity_bounds("active_material_fraction_pct", 95.2, "test_ctx")
