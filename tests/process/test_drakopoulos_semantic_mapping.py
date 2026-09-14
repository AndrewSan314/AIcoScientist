"""Tests verifying semantic header mapping for Drakopoulos dataset."""

from __future__ import annotations
import pytest
from src.datasets.battery_process.drakopoulos_graphite import (
    PHYSICAL_SANITY_BOUNDS,
    _validate_sanity_bounds,
    _find_header_col,
)

def test_find_header_col_matches_semantic_variations() -> None:
    headers = [
        (1, "Case", ""),
        (2, "Cell ID", ""),
        (3, "Coating Speed (m/min)", ""),
        (4, "Gap size", "um"),
        (5, "D30", "mAh"),
    ]
    assert _find_header_col(headers, r"speed") == 3
    assert _find_header_col(headers, r"gap\s*size") == 4
    assert _find_header_col(headers, r"d30") == 5
    assert _find_header_col(headers, r"nonexistent") is None

def test_permutation_invariance_semantic_mapping() -> None:
    # Column order in workbook does not affect semantic matching
    headers_permuted = [
        (1, "D30", "mAh"),
        (2, "Gap size", "um"),
        (3, "Coating Speed (m/min)", ""),
    ]
    assert _find_header_col(headers_permuted, r"speed") == 3
    assert _find_header_col(headers_permuted, r"gap\s*size") == 2
    assert _find_header_col(headers_permuted, r"d30") == 1
