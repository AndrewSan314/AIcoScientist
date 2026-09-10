from __future__ import annotations

import numpy as np

from src.process.models.uncertainty import conformal_interval, empirical_coverage


def test_conformal_interval_reports_empirical_coverage() -> None:
    low, high = conformal_interval(np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.9)
    assert empirical_coverage(np.array([1.1, 2.1]), low, high) == 1.0
