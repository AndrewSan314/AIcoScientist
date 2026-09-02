from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping, Sequence


def parse_chemical_formula(formula: str) -> dict[str, float]:
    """Parses a chemical formula (with nested parentheses/brackets and float/int amounts) into element counts.

    Handles space-group suffixes (e.g. 'BaCO3_62_(icsd_166091)-0' -> 'BaCO3') and stoichiometric multipliers.
    """
    cleaned = formula.strip().replace(" ", "")
    if not cleaned:
        return {}

    # Strip space group / ICSD suffixes if present
    if "_" in cleaned:
        cleaned = cleaned.split("_")[0]

    stack: list[defaultdict[str, float]] = [defaultdict(float)]
    i = 0
    n = len(cleaned)

    while i < n:
        ch = cleaned[i]
        if ch in ("(", "[", "{"):
            stack.append(defaultdict(float))
            i += 1
        elif ch in (")", "]", "}"):
            if len(stack) <= 1:
                raise ValueError(f"Unmatched closing bracket in formula '{formula}'")
            i += 1
            # Read multiplier after closing bracket
            m = re.match(r"^[0-9]+(?:\.[0-9]+)?", cleaned[i:])
            mult = 1.0
            if m:
                mult = float(m.group(0))
                i += len(m.group(0))
            popped = stack.pop()
            target = stack[-1]
            for el, count in popped.items():
                target[el] += count * mult
        else:
            # Read element (Capital letter followed by optional lowercase letter)
            m = re.match(r"^([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)?", cleaned[i:])
            if m:
                el = m.group(1)
                count = float(m.group(2)) if m.group(2) else 1.0
                stack[-1][el] += count
                i += len(m.group(0))
            else:
                raise ValueError(f"Unrecognized chemical symbol or syntax at '{cleaned[i:]}' in formula '{formula}'")

    if len(stack) != 1:
        raise ValueError(f"Unclosed opening bracket in formula '{formula}'")

    res = dict(stack[0])
    if not res and cleaned:
        raise ValueError(f"Could not parse valid chemical elements from formula '{formula}'")
    return res


def get_fractional_composition(formula: str) -> dict[str, float]:
    """Computes normalized elemental fractions summing to 1.0."""
    counts = parse_chemical_formula(formula)
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {el: c / total for el, c in counts.items()}


def are_chemically_equivalent(formula_a: str, formula_b: str, tol: float = 0.02) -> bool:
    """Checks whether two chemical formulas represent equivalent elemental stoichiometry within tolerance."""
    comp_a = get_fractional_composition(formula_a)
    comp_b = get_fractional_composition(formula_b)

    if not comp_a or not comp_b:
        return False
    if set(comp_a.keys()) != set(comp_b.keys()):
        return False

    return all(abs(comp_a[el] - comp_b[el]) <= tol for el in comp_a)


from src.domains.alab.canonical import normalize_phase_weights


def parse_refinement_phases(
    phase_weights: Mapping[str, float],
    target_formula: str,
    precursor_formulas: Sequence[str],
    rwp: float,
) -> dict[str, Any]:
    """Performs deterministic chemical phase matching on Rietveld refinement results.

    Calculates:
    - target_phase_fraction: fraction of phases chemically matching target compound
    - precursor_phase_fraction: fraction of unreacted precursor phases
    - other_identified_phase_fraction: intermediate or side phases
    - unknown_phase_fraction: unassigned residual fraction
    - num_identified_phases: total phase count
    - rwp_scaled: Rwp scaled to [0, 1] range (rwp / 10.0)
    """
    norm_weights, residual, unit_detected = normalize_phase_weights(phase_weights)

    target_frac = 0.0
    precursor_frac = 0.0
    other_frac = 0.0

    for phase_name, weight in norm_weights.items():
        w = float(weight)
        if are_chemically_equivalent(phase_name, target_formula):
            target_frac += w
        elif any(are_chemically_equivalent(phase_name, prec) for prec in precursor_formulas):
            precursor_frac += w
        else:
            other_frac += w

    target_frac = float(max(0.0, min(1.0, target_frac)))
    precursor_frac = float(max(0.0, min(1.0, precursor_frac)))
    other_frac = float(max(0.0, min(1.0, other_frac)))
    unknown_frac = float(max(0.0, min(1.0, 1.0 - (target_frac + precursor_frac + other_frac))))
    rwp_scaled = float(max(0.0, min(2.0, float(rwp) / 10.0)))

    # Canonical 4-dimensional standardized refinement vector:
    # [target_phase_fraction, precursor_phase_fraction, other_phase_fraction, rwp_scaled]
    feature_vector = [target_frac, precursor_frac, other_frac, rwp_scaled]

    return {
        "target_phase_fraction": target_frac,
        "precursor_phase_fraction": precursor_frac,
        "other_identified_phase_fraction": other_frac,
        "unknown_phase_fraction": unknown_frac,
        "num_identified_phases": len(norm_weights),
        "rwp": float(rwp),
        "rwp_scaled": rwp_scaled,
        "feature_vector": feature_vector,
        "phase_weights": norm_weights,
        "phase_weight_unit": unit_detected,
    }
