from __future__ import annotations

from src.process.optimization.process_objective import ConstraintSpec


def test_constraint_directions_are_explicit() -> None:
    assert ConstraintSpec("porosity", "range", (0.3, 0.4)).satisfied(0.35)
    assert not ConstraintSpec("temperature", "upper", 120).satisfied(121)
    assert ConstraintSpec("manufacturing_success", "feasibility", 0.9).satisfied(0.9)
