from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


class ScanSelectionResult:
    """Explicit container distinguishing ledger-canonical vs replay-fallback XRD scans."""

    def __init__(
        self,
        scan: Mapping[str, Any] | None,
        scan_index: int | None,
        selection_method: str,
        is_ledger_canonical: bool,
        is_replay_fallback: bool,
        is_canonical: bool,
    ):
        self.scan = scan
        self.scan_index = scan_index
        self.selection_method = selection_method
        self.is_ledger_canonical = is_ledger_canonical
        self.is_replay_fallback = is_replay_fallback
        self.is_canonical = is_canonical

    def __iter__(self):
        # Enables backwards-compatible 3-tuple unpacking: scan, idx, method = get_canonical_scan(sample)
        yield self.scan
        yield self.scan_index
        yield self.selection_method

    def __getitem__(self, index: int):
        return (self.scan, self.scan_index, self.selection_method)[index]

    def __len__(self):
        return 3

    def __repr__(self) -> str:
        return (
            f"ScanSelectionResult(scan_index={self.scan_index}, method={self.selection_method!r}, "
            f"is_ledger_canonical={self.is_ledger_canonical}, is_replay_fallback={self.is_replay_fallback}, "
            f"is_canonical={self.is_canonical})"
        )


def recompute_upstream_active_scan(sample: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, int | None, str]:
    """Recomputes upstream active scan mirroring precursor-genome SampleEntry.update_active_scan().

    Rules:
    1. Candidate scans must contain refinement cases.
    2. Refresh/select each scan's active refinement using upstream rules.
    3. Manual refinement beats automated (rank == -1 or origin == "manual").
    4. Human quality score determines best when available (lower is better).
    5. Lower Rwp tie-break.
    6. If no scan has refinement cases, active scan remains None.

    Returns:
        tuple of (scan_dict, scan_index, selection_method)
    """
    scans = sample.get("characterization", {}).get("xrd", {}).get("scans", [])
    if not scans:
        return None, None, "no_scans_available"

    best_scan_idx = None
    best_scan = None
    best_scan_key = None

    for idx, sc in enumerate(scans):
        cases = sc.get("refinement_cases", [])
        if not cases:
            continue
        best_case, _, _ = get_canonical_refinement_case(sc)
        if best_case is None:
            continue

        is_manual = 1 if (best_case.get("rank") == -1 or best_case.get("origin") == "manual") else 0
        verif = best_case.get("verification") or {}
        quality_score = verif.get("human_quality_score")
        q_val = float(quality_score) if isinstance(quality_score, (int, float)) else 999.0
        rwp = float(best_case.get("rwp", 999.0) or 999.0)

        sort_key = (is_manual, -q_val, -rwp, -idx)
        if best_scan_key is None or sort_key > best_scan_key:
            best_scan_key = sort_key
            best_scan_idx = idx
            best_scan = sc

    if best_scan is not None:
        return best_scan, best_scan_idx, "upstream_recomputed_active_scan"

    return None, None, "no_refinements_for_active_scan"


def get_canonical_scan(sample: Mapping[str, Any]) -> ScanSelectionResult:
    """Retrieves the canonical or deterministic replay fallback powder XRD scan for an A-Lab sample.

    Upstream A-Lab Ledger Semantics:
    1. SampleEntry.active_scan_index specifies the verified ledger-canonical scan if present and valid.
       -> is_ledger_canonical = True, is_replay_fallback = False, is_canonical = True.
    2. If missing/invalid, recompute upstream active scan using precursor-genome rules:
       -> is_ledger_canonical = False, is_replay_fallback = False, is_canonical = True.
    3. If no refinement-based active scan can be determined upstream, select a deterministic
       replay fallback scan:
       a. First scan marked is_active is True or status == "valid"
          -> selection_method = "replay_fallback_valid_scan"
       b. First scan in list (index 0)
          -> selection_method = "replay_fallback_first_scan"
       -> is_ledger_canonical = False, is_replay_fallback = True, is_canonical = False.
    """
    scans = sample.get("characterization", {}).get("xrd", {}).get("scans", [])
    if not scans:
        return ScanSelectionResult(None, None, "no_scans_available", False, False, False)

    # 1. Active scan index from ledger
    asi = sample.get("active_scan_index")
    if isinstance(asi, int) and 0 <= asi < len(scans):
        return ScanSelectionResult(
            scan=scans[asi],
            scan_index=asi,
            selection_method="ledger_active_scan_index",
            is_ledger_canonical=True,
            is_replay_fallback=False,
            is_canonical=True,
        )

    # 2. Upstream active scan recomputation
    recomputed_scan, recomputed_idx, method = recompute_upstream_active_scan(sample)
    if recomputed_scan is not None:
        return ScanSelectionResult(
            scan=recomputed_scan,
            scan_index=recomputed_idx,
            selection_method=method,
            is_ledger_canonical=False,
            is_replay_fallback=False,
            is_canonical=True,
        )

    # 3. Deterministic replay fallback (non-canonical)
    for idx, sc in enumerate(scans):
        if sc.get("is_active") is True or sc.get("status") == "valid":
            return ScanSelectionResult(
                scan=sc,
                scan_index=idx,
                selection_method="replay_fallback_valid_scan",
                is_ledger_canonical=False,
                is_replay_fallback=True,
                is_canonical=False,
            )

    return ScanSelectionResult(
        scan=scans[0],
        scan_index=0,
        selection_method="replay_fallback_first_scan",
        is_ledger_canonical=False,
        is_replay_fallback=True,
        is_canonical=False,
    )


def get_canonical_refinement_case(scan: Mapping[str, Any] | None) -> tuple[Mapping[str, Any] | None, int | None, str]:
    """Retrieves the canonical Rietveld refinement case for an XRD scan.

    # Mirrors precursor-genome Scan.active_refinement property selection priority:
    1. ScanEntry.active_case_index specifies the verified/canonical refinement case if present and valid.
    2. Fallback Priority:
       a. Manual refinement (rank == -1 or origin == "manual")
       b. Best human quality score (lower is better: score 1 < score 2 < score 3; unverified = 999.0)
       c. Lowest Rwp (lower is better)
       d. Lowest case index tie-break

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

    # 2. Priority scoring function for fallback (Mirrors precursor-genome Scan.active_refinement)
    best_idx = None
    best_case = None
    best_key = None

    for idx, c in enumerate(cases):
        is_manual = 1 if (c.get("rank") == -1 or c.get("origin") == "manual") else 0
        verif = c.get("verification") or {}

        # Human quality score: lower is better; unverified gets large penalty (999.0)
        quality_score = verif.get("human_quality_score")
        q_val = float(quality_score) if isinstance(quality_score, (int, float)) else 999.0

        rwp = float(c.get("rwp", 999.0) or 999.0)

        # Priority tuple: (higher manual, lower quality score, lower rwp, lower idx)
        sort_key = (is_manual, -q_val, -rwp, -idx)

        if best_key is None or sort_key > best_key:
            best_key = sort_key
            best_idx = idx
            best_case = c

    method = "upstream_fallback_lowest_rwp"
    if best_case:
        if best_case.get("rank") == -1 or best_case.get("origin") == "manual":
            method = "upstream_fallback_manual"
        elif best_case.get("verification", {}).get("human_quality_score") is not None:
            method = "upstream_fallback_quality_score"

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
