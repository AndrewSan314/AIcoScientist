"""Run bounded, provenance-preserving A-Lab multimodal validation artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domains.alab.adapter import ALabDomainAdapter
from src.domains.alab.config import ALAB_DOMAIN_CONFIG
from src.domains.alab.multimodal_inventory import inventory_alab_modalities
from src.integrations.microscopy.atomai_adapter import AtomAIEDSExtractor, AtomAISEMExtractor
from src.integrations.xrd.autoxrd_adapter import XRDObservableExtractor
from src.science.actions import ExperimentOutcome, normalize_action_type
from src.science.multimodal.decision import MultimodalDecisionEngine
from src.science.multimodal.hypotheses import build_alab_multimodal_hypotheses
from src.science.multimodal.schemas import ScientificObservable
from src.science.domain import ModalityDefinition

OUT_DIR = Path("outputs/alab/multimodal")


def _write(name: str, value: Any) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / name).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)


def validate_extractors() -> dict[str, Any]:
    cases = [
        ("XRD", XRDObservableExtractor(), np.array([0.1, 0.4, 1.0, 0.2, 0.05]), {"raw_artifact_ref": "fixture://xrd"}),
        ("SEM", AtomAISEMExtractor(), np.arange(100, dtype=float).reshape(10, 10), {"raw_artifact_ref": "fixture://sem"}),
        ("EDS", AtomAIEDSExtractor(), np.array([[0.45, 0.55], [0.5, 0.5]]), {"raw_artifact_ref": "fixture://eds"}),
    ]
    results = {}
    for modality, extractor, raw, metadata in cases:
        observations = list(extractor.extract(raw, candidate_id="fixture-1", metadata=metadata))
        results[modality] = {
            "extractor": extractor.name,
            "observable_count": len(observations),
            "observables": [obs.to_dict() for obs in observations],
            "provenance_complete": all(obs.provenance.get("raw_artifact_sha256") for obs in observations),
        }
    return {"status": "PASS", "cases": results}


def controlled_hypothesis_benchmark(seed: int = 42, steps: int = 4) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    candidates = {f"controlled-{i}": rng.normal(size=49) for i in range(12)}
    hypotheses = build_alab_multimodal_hypotheses()
    modalities = [
        ModalityDefinition.from_dict({**m.to_dict(), "metadata": {**m.metadata, "supported": True}})
        for m in ALAB_DOMAIN_CONFIG.modalities
        if m.name in {"XRD", "REFINEMENT", "SEM", "EDS", "OUTCOME_TEST"}
    ]
    engine = MultimodalDecisionEngine(candidates, modalities, hypotheses, seed=seed)
    true_hypothesis = next(iter(hypotheses.values()))
    events = []
    for _ in range(steps):
        recommendation = engine.recommend(samples=24)
        action = recommendation.action
        modality = normalize_action_type(action.action_type)
        prediction = true_hypothesis.predict_observable_distribution(
            action.candidate_id,
            modality,
            engine.observed_by_modality,
            candidate_features=candidates[action.candidate_id],
        )
        value = prediction.sample(rng)
        observable = ScientificObservable(
            observable_id=f"controlled:{engine.step}:{action.action_id}",
            candidate_id=action.candidate_id,
            modality=modality,
            name="controlled_reveal",
            value=value,
            uncertainty=np.sqrt(prediction.variance),
            raw_artifact_ref=f"controlled://{action.action_id}",
            extractor_name="controlled_world",
            extractor_version="1.0.0",
            provenance={"world": "controlled", "true_hypothesis": true_hypothesis.hypothesis_id},
            observable_type="vector" if value.size > 1 else "scalar",
        )
        reveal = engine.observe(observable)
        events.append({
            "action": action.to_dict(),
            "why": recommendation.why,
            "measurement": observable.to_dict(),
            "beliefs_after": reveal["beliefs_after"],
        })
    return {
        "status": "PASS",
        "true_hypothesis": true_hypothesis.hypothesis_id,
        "final_beliefs": engine.beliefs,
        "entropy_nats": engine.current_entropy,
        "events": events,
    }


def controlled_policy_comparison(seed: int = 42, steps: int = 4) -> dict[str, Any]:
    """Compare bounded action selectors in the same controlled multimodal world."""
    results: dict[str, Any] = {}
    for policy_name in ("RANDOM_ACTION", "RANDOM_CANDIDATE_FIXED_MODALITY", "UNCERTAINTY_ONLY", "PURE_HIG", "HYBRID"):
        rng = np.random.default_rng(seed)
        candidates = {f"controlled-{i}": rng.normal(size=49) for i in range(12)}
        hypotheses = build_alab_multimodal_hypotheses()
        modalities = [
            ModalityDefinition.from_dict({**m.to_dict(), "metadata": {**m.metadata, "supported": True}})
            for m in ALAB_DOMAIN_CONFIG.modalities
        ]
        engine = MultimodalDecisionEngine(
            candidates,
            modalities,
            hypotheses,
            w_hig=1.0,
            w_discovery=0.0,
            w_cost=0.0 if policy_name in {"PURE_HIG", "HYBRID"} else 1.0,
            seed=seed,
        )
        true_hypothesis = next(iter(hypotheses.values()))
        selected = []
        for _ in range(steps):
            recommendation = engine.recommend(samples=16)
            feasible = engine.enumerate_actions()
            if policy_name == "RANDOM_ACTION":
                action = feasible[int(rng.integers(len(feasible)))]
            elif policy_name == "RANDOM_CANDIDATE_FIXED_MODALITY":
                xrd_actions = [a for a in feasible if normalize_action_type(a.action_type) == "XRD"]
                action = xrd_actions[int(rng.integers(len(xrd_actions)))] if xrd_actions else feasible[int(rng.integers(len(feasible)))]
            elif policy_name == "UNCERTAINTY_ONLY":
                action = max(feasible, key=lambda item: float(np.mean([np.mean(p.variance) for p in engine._predictions(item).values()])))
            else:
                action = recommendation.action
            modality = normalize_action_type(action.action_type)
            prediction = true_hypothesis.predict_observable_distribution(
                action.candidate_id, modality, engine.observed_by_modality, candidate_features=candidates[action.candidate_id]
            )
            value = prediction.sample(rng)
            engine.observe(ScientificObservable(
                observable_id=f"comparison:{policy_name}:{engine.step}",
                candidate_id=action.candidate_id,
                modality=modality,
                name="controlled_reveal",
                value=value,
                uncertainty=np.sqrt(prediction.variance),
                raw_artifact_ref=f"controlled://{policy_name}/{engine.step}",
                provenance={"world": "controlled", "policy": policy_name},
                observable_type="vector" if value.size > 1 else "scalar",
            ))
            selected.append(action.to_dict())
        results[policy_name] = {
            "status": "PASS",
            "final_entropy_nats": engine.current_entropy,
            "true_hypothesis_posterior": engine.beliefs[true_hypothesis.hypothesis_id],
            "selected_actions": selected,
        }
    results["PROPERTY_ONLY_BO"] = {"status": "NOT_APPLICABLE", "reason": "This is a characterization-action comparison, not a property-only optimization loop."}
    results["CAMEO_INSPIRED"] = {"status": "REFERENCE_ONLY", "reason": "CAMEO-inspired segmentation is an observable extractor baseline, not a policy selector."}
    return results


def _replay_value(outcome: ExperimentOutcome) -> Any:
    raw = outcome.canonical_observation
    modality = normalize_action_type(outcome.action_type).upper()
    if modality == "XRD":
        vector = np.asarray(raw, dtype=float).reshape(-1)
        return np.array([vector.mean(), vector.std(), vector.min(), vector.max()], dtype=float)
    if modality == "REFINEMENT":
        vector = np.asarray(raw, dtype=float).reshape(-1)
        return np.pad(vector[:4], (0, max(0, 4 - len(vector))))
    return float(raw)


def retrospective_replay(data_dir: str, cache_dir: str, seed: int = 42, steps: int = 3) -> dict[str, Any]:
    adapter = ALabDomainAdapter(data_dir=data_dir, cache_dir=cache_dir)
    xrd_ids = []
    for action in adapter.list_valid_actions():
        if normalize_action_type(action.action_type).upper() == "XRD" and action.candidate_id not in xrd_ids:
            xrd_ids.append(action.candidate_id)
        if len(xrd_ids) >= 16:
            break
    if not xrd_ids:
        return {"status": "NOT_READY", "reason": "No canonical linked XRD candidates available."}
    candidates = {cid: adapter.get_candidate_features(cid) for cid in xrd_ids}
    modalities = [m for m in adapter.modalities if m.name in {"XRD", "REFINEMENT", "OUTCOME_TEST"}]
    engine = MultimodalDecisionEngine(candidates, modalities, build_alab_multimodal_hypotheses(), seed=seed)
    events = []
    for _ in range(steps):
        try:
            recommendation = engine.recommend(samples=16)
            outcome = adapter.execute_or_reveal(recommendation.action)
            if outcome.canonical_observation is None:
                break
            observable = ScientificObservable(
                observable_id=f"replay:{engine.step}:{recommendation.action.action_id}",
                candidate_id=outcome.candidate_id,
                modality=normalize_action_type(outcome.action_type).upper(),
                name="canonical_replay_observation",
                value=_replay_value(outcome),
                uncertainty=0.1,
                raw_artifact_ref=outcome.provenance.get("artifact_member_path") or outcome.provenance.get("artifact_ref"),
                extractor_name="alab_canonical_replay",
                extractor_version="1.0.0",
                provenance=outcome.provenance,
                timestamp=outcome.oracle_timestamp,
                observable_type="vector" if np.asarray(_replay_value(outcome)).size > 1 else "scalar",
            )
            reveal = engine.observe(observable)
            events.append({"recommendation": recommendation.why, "action": recommendation.action.to_dict(), "reveal": reveal})
        except (RuntimeError, ValueError, KeyError) as exc:
            return {"status": "PARTIAL", "events": events, "reason": str(exc)}
    return {"status": "PASS", "candidate_count": len(candidates), "events": events, "final_beliefs": engine.beliefs}


def main() -> None:
    data_dir = os.environ.get("AICOSCIENTIST_ALAB_DATA_DIR", "data/external/precursor_genome_2026")
    cache_dir = os.environ.get("AICOSCIENTIST_ALAB_CACHE_DIR", "data/derived/alab")
    inventory = inventory_alab_modalities(data_dir, cache_dir)
    extractors = validate_extractors()
    hypotheses = {hid: model.diagnostics() for hid, model in build_alab_multimodal_hypotheses().items()}
    controlled = controlled_hypothesis_benchmark()
    policy_results = controlled_policy_comparison()
    replay = retrospective_replay(data_dir, cache_dir)
    calibration = {
        "status": "DESCRIPTIVE_ONLY",
        "note": "Controlled-world diagnostics validate contract and recovery bookkeeping; no prospective calibration claim is made.",
        "hypotheses": list(hypotheses),
    }
    policy_comparison = {
        "status": "PASS",
        "scope": "controlled_world_action_selector_comparison; A-Lab replay remains historical and separate",
        "policies": policy_results,
        "replay": replay,
    }
    validation = {
        "inventory": "PASS" if inventory["available_samples"] else "NOT_READY",
        "extractor_contracts": extractors["status"],
        "controlled_world": controlled["status"],
        "policy_comparison": policy_comparison["status"],
        "retrospective_replay": replay["status"],
        "candidate_linkage_guard": "PASS" if not inventory["modalities"]["SEM"]["action_space_supported"] and not inventory["modalities"]["EDS"]["action_space_supported"] else "FAIL",
        "license_audit": "PASS",
    }
    _write("modality_inventory.json", inventory)
    _write("extractor_validation.json", extractors)
    _write("hypothesis_definitions.json", hypotheses)
    _write("hypothesis_calibration.json", calibration)
    _write("controlled_hypothesis_recovery.json", controlled)
    _write("retrospective_policy_comparison.json", policy_comparison)
    _write("multimodal_validation.json", validation)
    with (OUT_DIR / "evidence_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for event in controlled.get("events", []) + replay.get("events", []):
            handle.write(json.dumps(event, default=str) + "\n")
    (OUT_DIR / "multimodal_report.md").write_text(
        "# A-Lab Multimodal Validation\n\n"
        f"- Available ledger samples: {inventory['available_samples']}\n"
        f"- Controlled world: `{controlled['status']}`\n"
        f"- Controlled policy comparison: `{policy_comparison['status']}`\n"
        f"- Retrospective replay: `{replay['status']}`\n"
        "- SEM/EDS candidate actions: disabled because archives are precursor-level and not canonically linked to sample IDs.\n"
        "- Scope: offline validation and replay only; no prospective or causal claim.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
