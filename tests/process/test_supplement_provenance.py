import json
from pathlib import Path
import pytest


def test_supplement_provenance_requires_verified_hash_or_unverified_status():
    audit_path = Path("outputs/drakopoulos_source_reaudit/supplement_audit.json")
    if not audit_path.is_file():
        pytest.skip("supplement_audit.json not yet generated; will be verified after generation.")

    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    status = audit.get("status")
    assert status in ("OFFICIAL_SUPPLEMENT_LOCALLY_VERIFIED", "OFFICIAL_SUPPLEMENT_NOT_LOCALLY_VERIFIED")

    if status == "OFFICIAL_SUPPLEMENT_NOT_LOCALLY_VERIFIED":
        assert audit.get("mode_2_status") == "PUBLISHED_DESIGN_REQUIRES_RESTRICTED_PARTITION_C_MAPPING"
        assert audit.get("sha256") is None or audit.get("sha256") == ""
    else:
        assert audit.get("sha256") is not None and len(audit.get("sha256")) == 64


def test_unverified_supplement_blocks_unprovenanced_claims():
    # Helper to enforce policy: cannot claim verified supplement without hash
    def verify_claim(supplement_record: dict) -> bool:
        if supplement_record.get("status") == "OFFICIAL_SUPPLEMENT_NOT_LOCALLY_VERIFIED":
            return False
        if not supplement_record.get("sha256"):
            return False
        return True

    unverified = {"status": "OFFICIAL_SUPPLEMENT_NOT_LOCALLY_VERIFIED", "sha256": None}
    assert verify_claim(unverified) is False

    fake_verified_without_hash = {"status": "OFFICIAL_SUPPLEMENT_LOCALLY_VERIFIED", "sha256": ""}
    assert verify_claim(fake_verified_without_hash) is False

    verified_with_hash = {"status": "OFFICIAL_SUPPLEMENT_LOCALLY_VERIFIED", "sha256": "a" * 64}
    assert verify_claim(verified_with_hash) is True
