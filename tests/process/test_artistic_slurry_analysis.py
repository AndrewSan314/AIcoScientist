from scripts.analyze_artistic_slurry import _tail_stability_row


def test_tail_stability_zero_crossing_and_near_zero_are_not_relative_percentages() -> None:
    crossing = _tail_stability_row("pressure", [-1.0, -0.5, 0.5, 1.0], [0, 1, 2, 3])
    near_zero = _tail_stability_row("rotation", [1e-16, 2e-16, 3e-16, 0.0], [0, 1, 2, 3])
    assert crossing["crosses_zero"] is True
    assert crossing["relative_span_to_last"] is None
    assert crossing["relative_span_status"] == "ZERO_CROSSING_NOT_APPLICABLE"
    assert near_zero["relative_span_to_last"] is None
    assert near_zero["relative_span_status"] == "NEAR_ZERO_DENOMINATOR"


def test_tail_stability_nonzero_metric_and_block_diagnostic_are_deterministic() -> None:
    row = _tail_stability_row("temperature", [10.0, 11.0, 12.0, 13.0], [0, 10, 20, 30])
    assert row["relative_span_status"] == "VALID"
    assert row["relative_span_to_last"] == 3 / 13
    assert row["recent_window_diagnostic"] == "RECENT_WINDOW_DIAGNOSTIC"
    assert row["first_block_mean"] == 10.5
    assert row["second_block_mean"] == 12.5
    assert row["absolute_block_mean_shift"] == 2.0
