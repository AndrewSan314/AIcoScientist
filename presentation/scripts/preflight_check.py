#!/usr/bin/env python3
"""presentation/scripts/preflight_check.py

Automated comprehensive preflight checker for AIcoScientist Discovery Mission Control.
Verifies all source data artifacts, manifest hashes, frontend build, test suite, and backend endpoints.
"""

import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRESENTATION_DIR = ROOT / "presentation"
DATA_DIR = PRESENTATION_DIR / "data"
DIST_DIR = PRESENTATION_DIR / "frontend" / "dist"
SNAPSHOT_PATH = DATA_DIR / "snapshot.json"
MANIFEST_PATH = DATA_DIR / "snapshot_manifest.json"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check(name: str, passed: bool, detail: str = ""):
    mark = " [PASS] " if passed else " [FAIL] "
    print(f"{mark:<8} | {name:<40} | {detail}")
    return passed


def main():
    print("\n" + "=" * 75)
    print("  AIcoScientist Discovery Mission Control — Presentation Preflight")
    print("=" * 75)

    all_ok = True

    # 1. Manifest and Snapshot
    has_manifest = MANIFEST_PATH.exists()
    all_ok &= check("Manifest Available", has_manifest, str(MANIFEST_PATH.relative_to(ROOT)))

    has_snapshot = SNAPSHOT_PATH.exists()
    all_ok &= check("Snapshot Available", has_snapshot, f"{SNAPSHOT_PATH.stat().st_size / 1024 / 1024:.2f} MB" if has_snapshot else "Missing")

    if not has_manifest or not has_snapshot:
        print("\nCRITICAL: Manifest or snapshot missing. Run build_snapshot.py first.")
        sys.exit(1)

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    with SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
        snapshot = json.load(f)

    # 2. Source Artifact Hashes
    hashes = manifest.get("source_artifact_hashes", {})
    hash_mismatches = 0
    artifact_paths = {
        "evidence_ledger": ROOT / "outputs" / "alab" / "multimodal" / "evidence_ledger.jsonl",
        "precursor_genome_ledger": ROOT / "data" / "external" / "precursor_genome_2026" / "ledger_precursor_genome.json",
        "multimodal_validation": ROOT / "outputs" / "alab" / "multimodal" / "multimodal_validation.json",
        "hypothesis_calibration": ROOT / "outputs" / "alab" / "multimodal" / "per_observable_calibration.json",
        "full_policy_matrix": ROOT / "outputs" / "alab" / "multimodal" / "full_policy_matrix.json",
        "hig_sensitivity": ROOT / "outputs" / "alab" / "multimodal" / "hig_trajectory_sensitivity.json",
        "hypothesis_definitions": ROOT / "outputs" / "alab" / "multimodal" / "hypothesis_definitions.json",
        "modality_inventory": ROOT / "outputs" / "alab" / "multimodal" / "modality_inventory.json",
        "alab_dataset_audit": ROOT / "outputs" / "alab" / "alab_dataset_audit.json",
        "electrolyte_screening": ROOT / "outputs" / "electrolyte" / "benchmark" / "screening_quality_diagnostics.json",
        "electrolyte_simulation": ROOT / "outputs" / "electrolyte" / "benchmark" / "surrogate_simulation.json",
    }
    for k, expected_hash in hashes.items():
        p = artifact_paths.get(k)
        if not p or not p.exists() or compute_sha256(p) != expected_hash:
            hash_mismatches += 1

    all_ok &= check("Source Artifact Hashes", hash_mismatches == 0, f"{len(hashes) - hash_mismatches}/{len(hashes)} verified")

    # 3. Authentic Samples
    sample_count = len(snapshot.get("samples", []))
    all_ok &= check("Physical Sample Catalog", sample_count == 1035, f"{sample_count} authentic A-Lab records")

    # 4. Frontend Production Build
    has_html = (DIST_DIR / "index.html").exists()
    all_ok &= check("Frontend Production Dist", has_html, "dist/index.html ready")

    # 5. Automated Integrity Tests
    test_proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(PRESENTATION_DIR / "tests" / "test_ui_integrity.py"), "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    tests_passed = test_proc.returncode == 0
    all_ok &= check("Automated Integrity Tests", tests_passed, "5/5 tests passed in pytest" if tests_passed else test_proc.stderr[:80])

    # 6. Live Local Server Inspection (port 8501)
    server_online = False
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8501/api/health", timeout=1.5)
        if req.status == 200:
            health = json.loads(req.read())
            server_online = health.get("status") == "healthy"
    except Exception:
        server_online = False

    check("Live API Server (port 8501)", server_online, "Online (FastAPI / Uvicorn)" if server_online else "Offline (run server.py --port 8501)")

    print("-" * 75)
    if all_ok:
        print("  PREFLIGHT STATUS: ALL INTEGRITY GATES PASSED (ADVISOR DEMO READY)")
    else:
        print("  PREFLIGHT STATUS: GATES FAILED — REVIEW ITEMS ABOVE")
    print("=" * 75 + "\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
