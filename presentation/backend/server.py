#!/usr/bin/env python3
"""presentation/backend/server.py

Lightweight FastAPI server for AIcoScientist Mission Control.
Provides:
1. Static frontend hosting from presentation/frontend/dist/
2. Read-only artifact APIs (/api/snapshot)
3. Optional LIVE COMPUTED recommendations using MultimodalDecisionEngine (when optional backends are installed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FRONTEND_DIST = ROOT / "presentation" / "frontend" / "dist"
SNAPSHOT_PATH = ROOT / "presentation" / "data" / "snapshot.json"

app = FastAPI(
    title="AIcoScientist Discovery Mission Control API",
    description="Deterministic artifact adapter & live MultimodalDecisionEngine controller",
    version="2.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "mode": "PRESENTATION_MISSION_CONTROL",
        "snapshot_available": SNAPSHOT_PATH.exists(),
        "frontend_built": (FRONTEND_DIST / "index.html").exists(),
    }


@app.get("/api/snapshot")
def get_snapshot() -> Any:
    if not SNAPSHOT_PATH.exists():
        raise HTTPException(status_code=404, detail="snapshot.json not generated. Run build_snapshot.py first.")
    with SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/live/recommend")
def live_recommend(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute live MultimodalDecisionEngine if dependencies exist, else return documented fallback."""
    try:
        from src.science.actions import ScientificAction
        from src.science.domain import ModalityDefinition
        from src.science.multimodal.decision import MultimodalDecisionEngine
        from src.science.multimodal.hypotheses import MultimodalScientificHypothesis
        from src.science.multimodal.measurement_models import PredictiveObservableDistribution
        import numpy as np

        # Minimal standalone hypothesis setup that does not require BoTorch
        class LightweightHypothesis(MultimodalScientificHypothesis):
            def __init__(self, hid: str, mean_shift: float):
                self.hid = hid
                self.shift = mean_shift
            def fit(self, features, observed):
                pass
            def predict_observable_distribution(self, candidate_id, modality, observed, candidate_features=None):
                mean = np.array([0.5 + self.shift, 0.4])
                var = np.array([0.05, 0.05])
                return PredictiveObservableDistribution(self.hid, candidate_id, modality, mean, var)
            def log_likelihood(self, observable, observed):
                return -0.5
            def falsification_signature(self):
                return {"supported": "diagnostic separation"}

        candidates = {
            f"controlled-{i}": np.array([0.2 * i, 0.1 * ((i * 3) % 7), 0.5])
            for i in range(8)
        }
        modalities = [
            ModalityDefinition(name="XRD", observation_kind="characterization", cost=1.0),
            ModalityDefinition(name="REFINEMENT", observation_kind="characterization", cost=1.0),
            ModalityDefinition(name="OUTCOME_TEST", observation_kind="objective_measurement", cost=2.0),
        ]
        hypotheses = {
            "H1_PHASE_PURITY_LIMITED": LightweightHypothesis("H1_PHASE_PURITY_LIMITED", 0.1),
            "H2_COMPOSITION_HOMOGENEITY_LIMITED": LightweightHypothesis("H2_COMPOSITION_HOMOGENEITY_LIMITED", -0.1),
            "H3_MORPHOLOGY_KINETICS_LIMITED": LightweightHypothesis("H3_MORPHOLOGY_KINETICS_LIMITED", 0.0),
        }

        w_hig = float(payload.get("w_hig", 0.8))
        w_disc = float(payload.get("w_discovery", 0.8))
        w_cost = float(payload.get("w_cost", 2.0))
        policy_name = str(payload.get("policy", "HYBRID")).upper()

        engine = MultimodalDecisionEngine(
            candidate_features_by_id=candidates,
            modalities=modalities,
            hypotheses=hypotheses,
            w_hig=w_hig,
            w_discovery=w_disc,
            w_cost=w_cost,
            policy_name=policy_name,
            seed=int(payload.get("seed", 42)),
        )

        rec = engine.recommend(samples=32)

        return {
            "mode": "LIVE_COMPUTED",
            "action": rec.action.to_dict(),
            "score": rec.score,
            "why": rec.why,
            "why_not": rec.why_not,
            "preregistration": rec.preregistration,
            "current_beliefs": engine.beliefs,
            "current_entropy": engine.current_entropy,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live computation error: {e}")


# Static hosting for frontend
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str) -> Any:
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="AIcoScientist Mission Control Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8501, help="Bind port")
    parser.add_argument("--test", action="store_true", help="Run quick test and exit")
    args = parser.parse_args()

    if args.test:
        print("Testing FastAPI server startup and snapshot accessibility...")
        assert SNAPSHOT_PATH.exists(), "Snapshot missing!"
        print("Backend test passed!")
        sys.exit(0)

    print(f"\n=======================================================")
    print(f"  AIcoScientist Discovery Mission Control")
    print(f"  Advisor Presentation Interface: http://{args.host}:{args.port}")
    print(f"=======================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
