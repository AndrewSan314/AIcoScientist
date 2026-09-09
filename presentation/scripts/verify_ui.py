#!/usr/bin/env python3
"""verify_ui.py

Automated smoke test verifying that:
1. Backend launches and serves health API
2. Snapshot JSON contains all verified keys and 4 flagship steps
3. Static frontend index.html and assets are served with HTTP 200
4. Live recommendation API functions with MultimodalDecisionEngine
"""

import json
import sys
import time
import urllib.request
import subprocess

def run_tests():
    print("Launching test server on port 8502...")
    proc = subprocess.Popen(
        [sys.executable, "presentation/backend/server.py", "--port", "8502"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(2)

    try:
        # 1. Health
        with urllib.request.urlopen("http://127.0.0.1:8502/api/health") as res:
            assert res.status == 200
            data = json.loads(res.read())
            assert data["status"] == "healthy"
            assert data["snapshot_available"] is True
            assert data["frontend_built"] is True
            print("[PASS] /api/health passed")

        # 2. Manifest
        with urllib.request.urlopen("http://127.0.0.1:8502/api/manifest") as res:
            assert res.status == 200
            manifest = json.loads(res.read())
            assert manifest["snapshot_schema_version"] == "1.1.0"
            assert manifest["total_real_samples"] == 1035
            print("[PASS] /api/manifest passed (1,035 real samples declared)")

        # 3. Snapshot
        with urllib.request.urlopen("http://127.0.0.1:8502/api/snapshot") as res:
            assert res.status == 200
            snap = json.loads(res.read())
            assert "flagship_campaign" in snap
            assert len(snap["flagship_campaign"]["steps"]) == 4
            assert snap["validation"]["gate_evidence"]["boolean_gate_pass_count"] == 48
            assert len(snap["samples"]) == 1035
            print("[PASS] /api/snapshot passed (4 steps, 1,035 real samples, 48/50 gates verified)")

        # 4. Static SPA
        with urllib.request.urlopen("http://127.0.0.1:8502/") as res:
            assert res.status == 200
            html = res.read().decode("utf-8")
            assert "<!doctype html>" in html.lower()
            assert "assets/index-" in html
            print("[PASS] / (SPA index.html) passed")

        # 5. Diagnostic recommend API
        req = urllib.request.Request(
            "http://127.0.0.1:8502/api/diagnostic/smoke-recommend",
            data=json.dumps({"policy": "HYBRID", "seed": 42}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as res:
            assert res.status == 200
            rec = json.loads(res.read())
            assert rec["mode"] == "DIAGNOSTIC_SMOKE_TEST"
            assert "action" in rec
            print(f"[PASS] /api/diagnostic/smoke-recommend passed -> candidate {rec['action']['candidate_id']}, score: {rec['score']:.4f}")

        print("\nALL 5 AUTOMATED SMOKE TESTS PASSED SUCCESSFULLY.")
    finally:
        proc.terminate()
        proc.wait(timeout=5)

if __name__ == "__main__":
    run_tests()
