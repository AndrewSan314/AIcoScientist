from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class ParsedXRDMeasurement:
    """Scientific XRD measurement record parsed from raw bytes and scan metadata."""

    normalized_intensity: np.ndarray
    two_theta_start: float
    two_theta_end: float
    axis_source: str  # "xml_positions_2theta", "xml_datapoints", "canonical_xrd_settings", "ledger_metadata"


def parse_alab_xrd(
    raw_bytes: bytes,
    scan_metadata: Mapping[str, Any] | None = None,
    target_grid_points: int = 450,
    two_theta_min: float = 10.0,
    two_theta_max: float = 100.0,
) -> ParsedXRDMeasurement:
    """Enforces scientific XRD parsing and physical 2theta interpolation rules.

    Rules:
    1. Parse XML.
    2. Parse physical 2Theta from XML if available.
    3. Otherwise use explicit scan_metadata['xrd_settings']['range_2theta'].
    4. If both missing -> fail with ValueError.
    5. Require intensity/count vector (fail if empty or missing).
    6. Resample on physical 2Theta grid (e.g. 450 points, 10.0 to 100.0 degrees).
    7. Normalize intensity (max = 1.0).
    8. Return deterministic ParsedXRDMeasurement.
    """
    try:
        root = ET.fromstring(raw_bytes)
    except Exception as e:
        raise ValueError(f"Malformed XRD XML file: {e}") from e

    # Extract physical 2theta axis start and end positions
    start_pos: float | None = None
    end_pos: float | None = None
    axis_source = "xml_positions_2theta"
    for elem in root.iter():
        if elem.tag.endswith("positions") and elem.attrib.get("axis") == "2Theta":
            for child in elem:
                if child.tag.endswith("startPosition") and child.text:
                    try:
                        start_pos = float(child.text.strip())
                    except (ValueError, TypeError):
                        pass
                elif child.tag.endswith("endPosition") and child.text:
                    try:
                        end_pos = float(child.text.strip())
                    except (ValueError, TypeError):
                        pass
        elif elem.tag.endswith("startPosition") and elem.text and start_pos is None:
            try:
                start_pos = float(elem.text.strip())
                axis_source = "xml_datapoints"
            except (ValueError, TypeError):
                pass
        elif elem.tag.endswith("endPosition") and elem.text and end_pos is None:
            try:
                end_pos = float(elem.text.strip())
                axis_source = "xml_datapoints"
            except (ValueError, TypeError):
                pass

    # If 2theta axis missing in XML, check explicit scan metadata from canonical scan
    if start_pos is None or end_pos is None:
        xrd_settings = (scan_metadata or {}).get("xrd_settings", {})
        r2t = xrd_settings.get("range_2theta")
        if isinstance(r2t, (list, tuple)) and len(r2t) == 2:
            try:
                start_pos = float(r2t[0])
                end_pos = float(r2t[1])
                axis_source = "canonical_xrd_settings"
            except (ValueError, TypeError):
                pass

    if start_pos is None or end_pos is None:
        raise ValueError(
            "Missing physical 2Theta axis metadata for XRD scan. "
            "Neither XML positions nor xrd_settings.range_2theta provide axis limits."
        )

    # Extract raw intensities / counts
    intensities: list[float] = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in ("intensities", "counts") and elem.text:
            try:
                intensities = [float(x) for x in elem.text.split()]
            except (ValueError, TypeError):
                intensities = []
            break

    if not intensities:
        raise ValueError("Empty or missing XRD intensity counts in scan.")

    raw_arr = np.asarray(intensities, dtype=np.float64)
    if len(raw_arr) < 2:
        raise ValueError("XRD intensity array must have at least 2 points for interpolation.")

    # Physical 2theta axis interpolation onto canonical grid
    phys_2theta = np.linspace(start_pos, end_pos, len(raw_arr))
    canonical_grid = np.linspace(two_theta_min, two_theta_max, target_grid_points)
    norm_spec = np.interp(canonical_grid, phys_2theta, raw_arr)

    # Normalize intensity
    max_val = float(np.max(norm_spec))
    if max_val > 0:
        norm_spec = norm_spec / max_val

    return ParsedXRDMeasurement(
        normalized_intensity=norm_spec,
        two_theta_start=start_pos,
        two_theta_end=end_pos,
        axis_source=axis_source,
    )
