"""Tests verifying published Alchemite design supplement status."""

from __future__ import annotations
import json
from pathlib import Path
import pytest

def test_published_design_specifications() -> None:
    status_path = Path("outputs/drakopoulos_rediscovery_v2/published_design/status.json")
    assert status_path.is_file(), "Missing published design status.json"
    with open(status_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "PUBLISHED_DESIGN_REQUIRES_RESTRICTED_PARTITION_C_MAPPING"
    assert "firewall_compliance" in data

def test_supplement_audit_exists() -> None:
    supp_path = Path("outputs/drakopoulos_source_reaudit/supplement_audit.json")
    assert supp_path.is_file(), "Missing supplement_audit.json"
    with open(supp_path, "r", encoding="utf-8") as f:
        supp = json.load(f)
    part_c = supp["partitions"]["Partition_C_PUBLISHED_ALCHEMITE_DESIGN_VALIDATION"]
    assert part_c["fail_closed_code"] == "PUBLISHED_DESIGN_REQUIRES_RESTRICTED_PARTITION_C_MAPPING"
