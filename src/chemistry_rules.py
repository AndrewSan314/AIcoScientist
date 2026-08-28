"""
chemistry_rules.py — Physics and Chemistry Rules for Battery Material Recommender

This module encodes domain knowledge from the battery materials literature into
concrete, testable rules that constrain the search space of the AI recommender.

Rule sources:
  - Mass balance: fundamental stoichiometry.
  - Percolation threshold: continuum percolation theory for 2D fillers (MXene)
    combined with 0D fillers (carbon black).  See Franco et al., Energy Storage
    Materials 55, 336–349 (2023).
  - Binder-to-Si ratio: sodium alginate adhesion studies.  See J. Power Sources
    595, 235745 (2025).
  - Si/MXene synergy range: crumpled MXene encapsulation studies.  See ACS
    Appl. Energy Mater. 4, 12 (2021).
  - Drying temperature limits: MXene surface termination stability (-F, =O
    groups decompose above ~120 °C in air).
  - Volume expansion risk model: empirical heuristic combining Si content,
    binder buffering, and MXene network coverage.

Each rule returns ``(passed, reason)`` for traceability.  The module also
provides a composite ``validate_candidate`` function and a soft scoring
function ``score_recipe_quality`` for acquisition-score boosting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants — sourced from literature review (see docstring above)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositionLimits:
    """Hard limits on wt% for Si/MXene/Alginate/Carbon composite anodes."""
    si_min: float = 40.0
    si_max: float = 80.0
    mxene_min: float = 5.0
    mxene_max: float = 35.0
    alginate_min: float = 5.0
    alginate_max: float = 18.0
    carbon_min: float = 3.0
    total_wt: float = 100.0
    total_tolerance: float = 0.1  # ±0.1 wt%


@dataclass(frozen=True)
class TransportLimits:
    """Percolation and conductivity thresholds."""
    # MXene (2D) + Carbon (0D) must exceed this combined wt% for a
    # continuous electron-conductive network through the electrode.
    min_conductive_wt_pct: float = 15.0


@dataclass(frozen=True)
class MechanicalLimits:
    """Mechanical integrity constraints."""
    # Minimum binder-to-Si weight ratio to buffer >300% volume expansion
    # of nano-Si during lithiation.  Below this, rapid capacity fade
    # from particle cracking and loss of electrical contact is expected.
    min_binder_si_ratio: float = 0.08


@dataclass(frozen=True)
class SynergyLimits:
    """Empirical synergy windows from literature."""
    # Si/MXene weight ratio for effective encapsulation.
    si_mxene_ratio_min: float = 1.5
    si_mxene_ratio_max: float = 12.0


@dataclass(frozen=True)
class ProcessLimits:
    """Process-condition safety envelopes."""
    # MXene surface groups (-F, =O, -OH) degrade above ~120 °C in air.
    drying_temp_min: float = 60.0
    drying_temp_max: float = 120.0


# Singleton instances
COMP = CompositionLimits()
TRANSPORT = TransportLimits()
MECH = MechanicalLimits()
SYNERGY = SynergyLimits()
PROCESS = ProcessLimits()


# ---------------------------------------------------------------------------
# Individual rule functions
# ---------------------------------------------------------------------------

def check_mass_balance(recipe: Dict[str, float]) -> Tuple[bool, str]:
    """Rule 1: Total wt% must equal 100 ± tolerance."""
    si = recipe.get("si_content", 0.0)
    mxene = recipe.get("mxene_content", 0.0)
    alginate = recipe.get("alginate_content", 0.0)
    carbon = recipe.get("carbon_content", 0.0)
    total = si + mxene + alginate + carbon
    if abs(total - COMP.total_wt) > COMP.total_tolerance:
        return False, f"Mass balance violated: total={total:.2f}% (must be 100±{COMP.total_tolerance}%)"
    return True, "Mass balance OK"


def check_si_range(recipe: Dict[str, float]) -> Tuple[bool, str]:
    """Rule 2: Si content must be within [40, 80] wt%."""
    si = recipe.get("si_content", 0.0)
    if si < COMP.si_min:
        return False, f"Si={si:.1f}% < {COMP.si_min}%: insufficient capacity contribution"
    if si > COMP.si_max:
        return False, f"Si={si:.1f}% > {COMP.si_max}%: extreme volume expansion, electrode fracture"
    return True, "Si range OK"


def check_mxene_range(recipe: Dict[str, float]) -> Tuple[bool, str]:
    """Rule 3: MXene content must be within [5, 35] wt%."""
    mxene = recipe.get("mxene_content", 0.0)
    if mxene < COMP.mxene_min:
        return False, f"MXene={mxene:.1f}% < {COMP.mxene_min}%: no conductive 2D network"
    if mxene > COMP.mxene_max:
        return False, f"MXene={mxene:.1f}% > {COMP.mxene_max}%: displaces active Si, lowers capacity"
    return True, "MXene range OK"


def check_alginate_range(recipe: Dict[str, float]) -> Tuple[bool, str]:
    """Rule 4: Alginate binder content must be within [5, 18] wt%."""
    alginate = recipe.get("alginate_content", 0.0)
    if alginate < COMP.alginate_min:
        return False, f"Alginate={alginate:.1f}% < {COMP.alginate_min}%: weak adhesion, particle delamination"
    if alginate > COMP.alginate_max:
        return False, f"Alginate={alginate:.1f}% > {COMP.alginate_max}%: excessive insulating binder, high Rct"
    return True, "Alginate range OK"


def check_carbon_minimum(recipe: Dict[str, float]) -> Tuple[bool, str]:
    """Rule 5: Carbon additive must be ≥ 3 wt%."""
    carbon = recipe.get("carbon_content", 0.0)
    if carbon < COMP.carbon_min:
        return False, f"Carbon={carbon:.1f}% < {COMP.carbon_min}%: gaps between MXene sheets lack filler"
    return True, "Carbon minimum OK"


def check_percolation_threshold(recipe: Dict[str, float]) -> Tuple[bool, str]:
    """Rule 6: Combined conductive phase (MXene + Carbon) must exceed the
    electron-percolation threshold for continuous transport.

    For composites mixing 2D MXene sheets (low percolation ~1-3 vol%) with
    0D carbon black (~10-16 vol%), the combined threshold in wt% is
    approximately 15 wt% to ensure a robust network that survives Si
    volume cycling.
    """
    mxene = recipe.get("mxene_content", 0.0)
    carbon = recipe.get("carbon_content", 0.0)
    conductive = mxene + carbon
    if conductive < TRANSPORT.min_conductive_wt_pct:
        return False, (
            f"Conductive phase (MXene+Carbon)={conductive:.1f}% "
            f"< {TRANSPORT.min_conductive_wt_pct}%: below percolation threshold, "
            f"high Rct and loss of contact expected"
        )
    return True, "Percolation threshold OK"


def check_binder_si_ratio(recipe: Dict[str, float]) -> Tuple[bool, str]:
    """Rule 7: Binder-to-Si ratio must be sufficient to buffer volume
    expansion (~300%) of nano-silicon during lithiation/delithiation.

    Alginate's carboxyl groups form hydrogen bonds with SiO₂ native oxide;
    at ratio < 0.08, the binder network is too sparse to maintain contact.
    """
    si = recipe.get("si_content", 0.0)
    alginate = recipe.get("alginate_content", 0.0)
    if si <= 0:
        return False, "Si content must be positive"
    ratio = alginate / si
    if ratio < MECH.min_binder_si_ratio:
        return False, (
            f"Binder/Si ratio={ratio:.3f} < {MECH.min_binder_si_ratio}: "
            f"insufficient mechanical buffering for Si volume expansion"
        )
    return True, "Binder-to-Si ratio OK"


def check_si_mxene_synergy(recipe: Dict[str, float]) -> Tuple[bool, str]:
    """Rule 8: Si/MXene weight ratio must be in the effective encapsulation
    window [1.5, 12.0].

    Below 1.5, MXene dominates and capacity is wasted on inactive mass.
    Above 12.0, MXene is too sparse to wrap Si particles effectively —
    leading to unprotected expansion and rapid SEI growth.
    """
    si = recipe.get("si_content", 0.0)
    mxene = recipe.get("mxene_content", 0.0)
    if mxene <= 0:
        return False, "MXene content must be positive"
    ratio = si / mxene
    if ratio < SYNERGY.si_mxene_ratio_min:
        return False, (
            f"Si/MXene={ratio:.2f} < {SYNERGY.si_mxene_ratio_min}: "
            f"excess MXene, capacity penalty"
        )
    if ratio > SYNERGY.si_mxene_ratio_max:
        return False, (
            f"Si/MXene={ratio:.2f} > {SYNERGY.si_mxene_ratio_max}: "
            f"insufficient MXene wrapping, unprotected Si expansion"
        )
    return True, "Si/MXene synergy OK"


def check_drying_temperature(recipe: Dict[str, float]) -> Tuple[bool, str]:
    """Rule 9: Drying temperature must preserve MXene surface chemistry.

    Ti₃C₂Tₓ MXene surface groups (-F, =O, -OH) are critical for:
    - Electronic conductivity at the MXene/Si interface
    - Hydrogen bonding with alginate binder
    - Formation of a stable, thin SEI layer

    Above ~120 °C in air, these groups begin to decompose, degrading
    conductivity and adhesion.  Below 60 °C, solvent removal is too slow.
    """
    temp = recipe.get("drying_temp", 80.0)
    if temp < PROCESS.drying_temp_min:
        return False, f"Drying temp={temp}°C < {PROCESS.drying_temp_min}°C: incomplete solvent removal"
    if temp > PROCESS.drying_temp_max:
        return False, (
            f"Drying temp={temp}°C > {PROCESS.drying_temp_max}°C: "
            f"MXene surface groups (-F, =O) decomposition risk"
        )
    return True, "Drying temperature OK"


def estimate_volume_expansion_risk(recipe: Dict[str, float]) -> Tuple[float, str]:
    """Rule 10 (soft): Estimate relative volume-expansion risk.

    Higher Si → higher expansion.
    More binder and MXene → better buffering.

    Returns a risk score in [0, 1] where:
      0.0 = lowest risk (well-buffered)
      1.0 = highest risk (extreme Si, minimal buffer)

    This is a heuristic model based on the following factors:
      - Si fraction relative to its safe maximum
      - Binder adequacy (alginate/Si ratio vs. recommended minimum)
      - MXene coverage (MXene wt% relative to Si)
    """
    si = recipe.get("si_content", 0.0)
    alginate = recipe.get("alginate_content", 0.0)
    mxene = recipe.get("mxene_content", 0.0)

    # Normalize Si aggressiveness: 0 at si_min, 1 at si_max
    si_norm = np.clip((si - COMP.si_min) / max(COMP.si_max - COMP.si_min, 1), 0, 1)

    # Binder buffering factor: 1 (no buffering) → 0 (well buffered)
    binder_ratio = alginate / max(si, 1)
    binder_factor = np.clip(1.0 - binder_ratio / 0.20, 0, 1)  # 0.20 = generous binder target

    # MXene wrapping factor: 1 (no wrapping) → 0 (good wrapping)
    mxene_ratio = mxene / max(si, 1)
    mxene_factor = np.clip(1.0 - mxene_ratio / 0.40, 0, 1)   # 0.40 = ideal coverage target

    # Weighted combination
    risk = 0.50 * si_norm + 0.25 * binder_factor + 0.25 * mxene_factor
    risk = float(np.clip(risk, 0, 1))

    if risk <= 0.3:
        label = "low risk"
    elif risk <= 0.6:
        label = "moderate risk"
    else:
        label = "high risk"

    return risk, f"Volume expansion risk={risk:.2f} ({label})"


# ---------------------------------------------------------------------------
# Composite validation
# ---------------------------------------------------------------------------

# All hard rules in execution order
HARD_RULES = [
    check_mass_balance,
    check_si_range,
    check_mxene_range,
    check_alginate_range,
    check_carbon_minimum,
    check_percolation_threshold,
    check_binder_si_ratio,
    check_si_mxene_synergy,
    check_drying_temperature,
]


@dataclass
class ValidationResult:
    """Result of validating a candidate recipe against all chemistry rules."""
    valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    volume_expansion_risk: float = 0.0
    chemistry_score: float = 0.0


def validate_candidate(recipe: Dict[str, float]) -> ValidationResult:
    """Run all hard rules and the soft risk estimator on a candidate recipe.

    Parameters
    ----------
    recipe : dict
        Must contain at minimum: si_content, mxene_content, alginate_content,
        carbon_content.  May also contain drying_temp.

    Returns
    -------
    ValidationResult
        .valid is True only if all hard rules pass.
        .violations lists the reasons for any failures.
        .volume_expansion_risk is the soft risk score [0, 1].
        .chemistry_score is a composite quality score [0, 1] for ranking.
    """
    violations = []
    warnings = []

    for rule_fn in HARD_RULES:
        passed, reason = rule_fn(recipe)
        if not passed:
            violations.append(reason)

    # Soft checks (always computed, even if hard rules fail)
    risk, risk_msg = estimate_volume_expansion_risk(recipe)
    if risk > 0.6:
        warnings.append(risk_msg)

    # Compute chemistry quality score
    chem_score = score_recipe_quality(recipe)

    return ValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        volume_expansion_risk=risk,
        chemistry_score=chem_score,
    )


# ---------------------------------------------------------------------------
# Soft scoring for acquisition-score boosting
# ---------------------------------------------------------------------------

def score_recipe_quality(recipe: Dict[str, float]) -> float:
    """Compute a chemistry-informed quality score in [0, 1].

    This score rewards recipes that sit in the "physical sweet spot"
    identified by literature:
      - Si around 60-70 wt% (high capacity, manageable expansion)
      - MXene+Carbon around 20-30 wt% (robust conductive network)
      - Alginate around 8-12 wt% (good adhesion without excess resistance)
      - Si/MXene ratio around 3-5 (optimal wrapping coverage)

    The score is the average of four Gaussian-bell sub-scores, each
    centered on the literature-optimal value with a width reflecting
    the acceptable range.
    """
    si = recipe.get("si_content", 0.0)
    mxene = recipe.get("mxene_content", 0.0)
    alginate = recipe.get("alginate_content", 0.0)
    carbon = recipe.get("carbon_content", 0.0)

    def _bell(value: float, center: float, width: float) -> float:
        """Gaussian bell: 1.0 at center, decays with distance."""
        return float(np.exp(-0.5 * ((value - center) / max(width, 1e-6)) ** 2))

    # Sub-scores with (center, width) from literature sweet spots
    si_score = _bell(si, center=65.0, width=10.0)
    conductive_score = _bell(mxene + carbon, center=25.0, width=8.0)
    alginate_score = _bell(alginate, center=10.0, width=4.0)

    si_mxene_ratio = si / max(mxene, 1e-6)
    synergy_score = _bell(si_mxene_ratio, center=4.0, width=2.5)

    # Volume expansion risk penalty
    risk, _ = estimate_volume_expansion_risk(recipe)
    risk_penalty = 1.0 - risk  # 1.0 = no risk, 0.0 = max risk

    # Weighted average
    score = (
        0.25 * si_score
        + 0.25 * conductive_score
        + 0.15 * alginate_score
        + 0.20 * synergy_score
        + 0.15 * risk_penalty
    )
    return float(np.clip(score, 0.0, 1.0))
