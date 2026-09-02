from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


def get_canonical_scan(sample: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, int | None, str]:
    """Retrieves the canonical powder XRD scan entry for an A-Lab sample.

    Upstream A-Lab Ledger Semantics:
    1. SampleEntry.active_scan_index specifies the verified/canonical scan if present and valid.
    2. Fallback:
       a. First scan marked is_active is True or status == "valid"
       b. First scan containing refinement cases
       c. First scan in list (index 0)

    Returns:
        tuple of (scan_dict, scan_index, selection_method)
    """
    scans = sample.get("characterization", {}).get("xrd", {}).get("scans", [])
    if not scans:
        return None, None, "no_scans_available"

    # 1. Active scan index from ledger
    asi = sample.get("active_scan_index")
    if isinstance(asi, int) and 0 <= asi < len(scans):
        scan = scans[asi]
        return scan, asi, "ledger_active_scan_index"

    # 2. Fallback: active or valid scan
    for idx, sc in enumerate(scans):
        if sc.get("is_active") is True or sc.get("status") == "valid":
            return sc, idx, "status_active_or_valid"

    # 3. Fallback: scan with refinement cases
    for idx, sc in enumerate(scans):
        if sc.get("refinement_cases"):
            return sc, idx, "has_refinement_cases"

    # 4. Final deterministic fallback: scan 0
    return scans[0], 0, "fallback_first_scan"


def get_canonical_refinement_case(scan: Mapping[str, Any] | None) -> tuple[Mapping[str, Any] | None, int | None, str]:
    """Retrieves the canonical Rietveld refinement case for an XRD scan.

    Upstream A-Lab Ledger Semantics:
    1. ScanEntry.active_case_index specifies the verified/canonical refinement case if present and valid.
    2. Fallback Priority:
       a. Manual refinement (rank == -1 or origin == "manual")
       b. Verified / accepted refinement (verification.is_accepted is True)
       c. Best human quality score (lower is better, e.g. score 1 < score 2 < score 3)
       d. Lowest Rwp (lower is better)
       e. Case index tie-break

    Returns:
        tuple of (case_dict, case_index, selection_method)
    """
    if not scan:
        return None, None, "no_scan_provided"

    cases = scan.get("refinement_cases", [])
    if not cases:
        return None, None, "no_refinement_cases"

    # 1. Active case index from ledger
    aci = scan.get("active_case_index")
    if isinstance(aci, int) and 0 <= aci < len(cases):
        return cases[aci], aci, "ledger_active_case_index"

    # 2. Priority scoring function for fallback
    best_idx = None
    best_case = None
    best_key = None

    for idx, c in enumerate(cases):
        is_manual = 1 if (c.get("rank") == -1 or c.get("origin") == "manual") else 0
        verif = c.get("verification") or {}
        is_accepted = 1 if verif.get("is_accepted") is True else 0

        # Human quality score: lower is better; unverified gets large penalty
        quality_score = verif.get("human_quality_score")
        q_val = float(quality_score) if isinstance(quality_score, (int, float)) else 999.0

        rwp = float(c.get("rwp", 999.0) or 999.0)

        # Priority tuple: (higher manual, higher accepted, lower quality score, lower rwp, lower idx)
        # We invert so that higher is better for max()
        sort_key = (is_manual, is_accepted, -q_val, -rwp, -idx)

        if best_key is None or sort_key > best_key:
            best_key = sort_key
            best_idx = idx
            best_case = c

    method = "fallback_priority"
    if best_case and (best_case.get("rank") == -1 or best_case.get("origin") == "manual"):
        method = "fallback_manual_preferred"
    elif best_case and best_case.get("verification", {}).get("is_accepted") is True:
        method = "fallback_human_accepted"

    return best_case, best_idx, method


def normalize_phase_weights(
    phase_weights: Mapping[str, Any],
) -> tuple[dict[str, float], float, str]:
    """Normalizes raw Rietveld refinement phase weights to standardized fractions in [0, 1].

    Handles:
    1. Detecting whether values are percentage (0-100) vs fraction (0-1).
       If sum of weights > 1.5 or any weight > 1.0, scales by 1/100.
    2. Validating weight values (non-negative, finite).
    3. Exposing residual/unknown fraction: max(0.0, 1.0 - sum(weights)).

    Returns:
        tuple of (normalized_weights_dict, residual_fraction, unit_detected)
    """
    if not phase_weights:
        return {}, 1.0, "empty"

    raw_parsed: dict[str, float] = {}
    for k, v in phase_weights.items():
        try:
            val = float(v)
            if val < 0.0:
                logger.warning("Negative phase weight for phase '%s': %f. Clamping to 0.", k, val)
                val = 0.0
            raw_parsed[str(k)] = val
        except (ValueError, TypeError):
            continue

    if not raw_parsed:
        return {}, 1.0, "invalid"

    total_weight = sum(raw_parsed.values())
    any_over_one = any(v > 1.0 for v in raw_parsed.values())

    if total_weight > 1.5 or any_over_one:
        unit_detected = "percentage"
        scale = 0.01
    else:
        unit_detected = "fraction"
        scale = 1.0

    normalized = {k: float(v * scale) for k, v in raw_parsed.items()}
    norm_total = sum(normalized.values())

    # If sum slightly exceeds 1.0 (e.g. 1.002 due to rounding), renormalize
    if norm_total > 1.02:
        logger.debug("Phase weights sum to %f (> 1.02). Renormalizing to 1.0.", norm_total)
        normalized = {k: float(v / norm_total) for k, v in normalized.items()}
        residual = 0.0
    else:
        residual = float(max(0.0, 1.0 - norm_total))

    return normalized, residual, unit_detected
