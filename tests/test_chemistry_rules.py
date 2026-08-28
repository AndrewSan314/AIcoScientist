"""Tests for src/chemistry_rules.py — physics and chemistry rule filters."""
import pytest

from src.chemistry_rules import (
    COMP,
    MECH,
    PROCESS,
    SYNERGY,
    TRANSPORT,
    ValidationResult,
    check_alginate_range,
    check_binder_si_ratio,
    check_carbon_minimum,
    check_drying_temperature,
    check_mass_balance,
    check_mxene_range,
    check_percolation_threshold,
    check_si_mxene_synergy,
    check_si_range,
    estimate_volume_expansion_risk,
    score_recipe_quality,
    validate_candidate,
)


# ── Helper ──────────────────────────────────────────────────────────────────

def _good_recipe(**overrides):
    """Return a recipe that passes all hard rules by default."""
    recipe = {
        "si_content": 65.0,
        "mxene_content": 15.0,
        "alginate_content": 10.0,
        "carbon_content": 10.0,
        "drying_temp": 80.0,
        "mixing_time": 45.0,
    }
    recipe.update(overrides)
    return recipe


# ── Rule 1: Mass balance ────────────────────────────────────────────────────

class TestMassBalance:
    def test_pass(self):
        ok, _ = check_mass_balance(_good_recipe())
        assert ok

    def test_fail_over(self):
        ok, reason = check_mass_balance(_good_recipe(carbon_content=15.0))  # total=105
        assert not ok
        assert "Mass balance" in reason

    def test_fail_under(self):
        ok, _ = check_mass_balance(_good_recipe(carbon_content=5.0))  # total=95
        assert not ok

    def test_edge_within_tolerance(self):
        # 100.05 is within ±0.1
        ok, _ = check_mass_balance(_good_recipe(carbon_content=10.05))
        assert ok


# ── Rule 2: Si range ────────────────────────────────────────────────────────

class TestSiRange:
    def test_pass(self):
        ok, _ = check_si_range(_good_recipe())
        assert ok

    def test_fail_too_low(self):
        ok, reason = check_si_range(_good_recipe(si_content=35.0))
        assert not ok
        assert "insufficient capacity" in reason.lower() or "Si=" in reason

    def test_fail_too_high(self):
        ok, reason = check_si_range(_good_recipe(si_content=85.0))
        assert not ok
        assert "volume expansion" in reason.lower() or "fracture" in reason.lower()

    def test_boundary_min(self):
        ok, _ = check_si_range(_good_recipe(si_content=40.0))
        assert ok

    def test_boundary_max(self):
        ok, _ = check_si_range(_good_recipe(si_content=80.0))
        assert ok


# ── Rule 3: MXene range ────────────────────────────────────────────────────

class TestMXeneRange:
    def test_pass(self):
        ok, _ = check_mxene_range(_good_recipe())
        assert ok

    def test_fail_too_low(self):
        ok, _ = check_mxene_range(_good_recipe(mxene_content=2.0))
        assert not ok

    def test_fail_too_high(self):
        ok, _ = check_mxene_range(_good_recipe(mxene_content=40.0))
        assert not ok


# ── Rule 4: Alginate range ──────────────────────────────────────────────────

class TestAlginateRange:
    def test_pass(self):
        ok, _ = check_alginate_range(_good_recipe())
        assert ok

    def test_fail_too_low(self):
        ok, _ = check_alginate_range(_good_recipe(alginate_content=3.0))
        assert not ok

    def test_fail_too_high(self):
        ok, _ = check_alginate_range(_good_recipe(alginate_content=22.0))
        assert not ok


# ── Rule 5: Carbon minimum ─────────────────────────────────────────────────

class TestCarbonMinimum:
    def test_pass(self):
        ok, _ = check_carbon_minimum(_good_recipe())
        assert ok

    def test_fail(self):
        ok, _ = check_carbon_minimum(_good_recipe(carbon_content=1.0))
        assert not ok


# ── Rule 6: Percolation threshold ──────────────────────────────────────────

class TestPercolationThreshold:
    def test_pass(self):
        ok, _ = check_percolation_threshold(_good_recipe())  # 15+10=25
        assert ok

    def test_fail(self):
        ok, reason = check_percolation_threshold(
            _good_recipe(mxene_content=5.0, carbon_content=5.0)  # 10 < 15
        )
        assert not ok
        assert "percolation" in reason.lower()

    def test_boundary(self):
        ok, _ = check_percolation_threshold(
            _good_recipe(mxene_content=10.0, carbon_content=5.0)  # exactly 15
        )
        assert ok


# ── Rule 7: Binder-to-Si ratio ─────────────────────────────────────────────

class TestBinderSiRatio:
    def test_pass(self):
        # 10/65 ≈ 0.154 > 0.08
        ok, _ = check_binder_si_ratio(_good_recipe())
        assert ok

    def test_fail(self):
        # 5/80 = 0.0625 < 0.08
        ok, reason = check_binder_si_ratio(
            _good_recipe(si_content=80.0, alginate_content=5.0)
        )
        assert not ok
        assert "buffering" in reason.lower() or "Binder/Si" in reason


# ── Rule 8: Si/MXene synergy ──────────────────────────────────────────────

class TestSiMXeneSynergy:
    def test_pass(self):
        # 65/15 ≈ 4.33 within [1.5, 12.0]
        ok, _ = check_si_mxene_synergy(_good_recipe())
        assert ok

    def test_fail_too_low(self):
        # 40/35 ≈ 1.14 < 1.5
        ok, _ = check_si_mxene_synergy(
            _good_recipe(si_content=40.0, mxene_content=35.0)
        )
        assert not ok

    def test_fail_too_high(self):
        # 75/5 = 15 > 12
        ok, _ = check_si_mxene_synergy(
            _good_recipe(si_content=75.0, mxene_content=5.0)
        )
        assert not ok


# ── Rule 9: Drying temperature ─────────────────────────────────────────────

class TestDryingTemperature:
    def test_pass(self):
        ok, _ = check_drying_temperature(_good_recipe())
        assert ok

    def test_fail_too_low(self):
        ok, _ = check_drying_temperature(_good_recipe(drying_temp=50.0))
        assert not ok

    def test_fail_too_high(self):
        ok, reason = check_drying_temperature(_good_recipe(drying_temp=150.0))
        assert not ok
        assert "decomposition" in reason.lower() or "MXene" in reason


# ── Rule 10: Volume expansion risk (soft) ──────────────────────────────────

class TestVolumeExpansionRisk:
    def test_low_risk_recipe(self):
        risk, _ = estimate_volume_expansion_risk(
            _good_recipe(si_content=50.0, alginate_content=12.0, mxene_content=20.0)
        )
        assert 0 <= risk <= 0.4, f"Expected low risk, got {risk}"

    def test_high_risk_recipe(self):
        risk, _ = estimate_volume_expansion_risk(
            _good_recipe(si_content=80.0, alginate_content=5.0, mxene_content=5.0)
        )
        assert risk > 0.5, f"Expected high risk, got {risk}"

    def test_range(self):
        risk, _ = estimate_volume_expansion_risk(_good_recipe())
        assert 0.0 <= risk <= 1.0


# ── Composite validation ───────────────────────────────────────────────────

class TestValidateCandidate:
    def test_valid_recipe(self):
        result = validate_candidate(_good_recipe())
        assert result.valid
        assert result.violations == []
        assert result.chemistry_score > 0

    def test_invalid_recipe_reports_all_violations(self):
        bad = _good_recipe(
            si_content=85.0,     # violates si_range + mass_balance + synergy
            carbon_content=10.0, # mass ≠ 100
        )
        result = validate_candidate(bad)
        assert not result.valid
        assert len(result.violations) >= 1

    def test_multiple_violations(self):
        bad = {
            "si_content": 90.0,
            "mxene_content": 2.0,
            "alginate_content": 2.0,
            "carbon_content": 1.0,
            "drying_temp": 200.0,
        }
        result = validate_candidate(bad)
        assert not result.valid
        # Should catch: mass balance, si range, mxene range, alginate range,
        # carbon min, percolation, binder/si, synergy, drying temp
        assert len(result.violations) >= 5

    def test_chemistry_score_range(self):
        result = validate_candidate(_good_recipe())
        assert 0.0 <= result.chemistry_score <= 1.0


# ── Score recipe quality ───────────────────────────────────────────────────

class TestScoreRecipeQuality:
    def test_sweet_spot_scores_high(self):
        """A recipe near the literature sweet spot should score highly."""
        score = score_recipe_quality(_good_recipe())
        assert score >= 0.5

    def test_extreme_recipe_scores_lower(self):
        """A recipe at the boundary should score lower."""
        extreme = _good_recipe(si_content=80.0, mxene_content=5.0,
                               alginate_content=5.0, carbon_content=10.0)
        extreme_score = score_recipe_quality(extreme)
        sweet_score = score_recipe_quality(_good_recipe())
        assert extreme_score < sweet_score

    def test_always_in_unit_interval(self):
        for si in range(40, 81, 10):
            for mx in range(5, 36, 10):
                alg = 10
                c = 100 - si - mx - alg
                if c < 0:
                    continue
                s = score_recipe_quality({
                    "si_content": si, "mxene_content": mx,
                    "alginate_content": alg, "carbon_content": c,
                })
                assert 0.0 <= s <= 1.0, f"Score {s} out of range for Si={si}, MXene={mx}"
