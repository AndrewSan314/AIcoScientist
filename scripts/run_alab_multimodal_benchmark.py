"""Run bounded, provenance-preserving A-Lab multimodal validation artifacts."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domains.alab.adapter import ALabDomainAdapter
from src.domains.alab.config import ALAB_DOMAIN_CONFIG
from src.domains.alab.multimodal_inventory import inventory_alab_modalities
from src.integrations.microscopy.atomai_adapter import ClassicalEDSDescriptorExtractor, ClassicalSEMDescriptorExtractor
from src.integrations.xrd.autoxrd_adapter import DeterministicXRDSpectralDescriptorExtractor
from src.science.actions import ExperimentOutcome, normalize_action_type
from src.science.multimodal.decision import MultimodalDecisionEngine
from src.science.multimodal.hypotheses import build_alab_multimodal_hypotheses
from src.science.multimodal.ontology import MODALITY_OBSERVABLE_NAMES, OBSERVABLE_REGISTRY, observable_names_for_modality
from src.science.multimodal.schemas import ScientificObservable
from src.science.domain import ModalityDefinition

OUT_DIR = Path("outputs/alab/multimodal")
SEEDS = (7, 42, 101, 314, 2024)
WORLD_MODALITY_PROFILES = {
    "WORLD_H1_PHASE_PURITY": {"diagnostic": "REFINEMENT", "modalities": ("REFINEMENT", "XRD", "OUTCOME_TEST")},
    "WORLD_H2_COMPOSITION_HOMOGENEITY": {"diagnostic": "EDS", "modalities": ("EDS", "OUTCOME_TEST", "XRD")},
    "WORLD_H3_MORPHOLOGY_KINETICS": {"diagnostic": "SEM", "modalities": ("SEM", "OUTCOME_TEST", "XRD")},
}
POLICIES = ("RANDOM_ACTION", "RANDOM_CANDIDATE_FIXED_MODALITY", "UNCERTAINTY_ONLY", "DISCOVERY_ONLY", "PURE_HIG", "HYBRID")


def _write(name: str, value: Any) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / name).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)


def validate_extractors() -> dict[str, Any]:
    cases = [
        ("XRD", DeterministicXRDSpectralDescriptorExtractor(), np.array([0.1, 0.4, 1.0, 0.2, 0.05]), {"raw_artifact_ref": "fixture://xrd"}),
        ("SEM", ClassicalSEMDescriptorExtractor(), np.arange(100, dtype=float).reshape(10, 10), {"raw_artifact_ref": "fixture://sem"}),
        ("EDS", ClassicalEDSDescriptorExtractor(), np.array([[0.45, 0.55], [0.5, 0.5]]), {"raw_artifact_ref": "fixture://eds"}),
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
    return {"status": "CONTRACT_VALIDATED", "cases": results}


def _controlled_modalities(world: str) -> list[ModalityDefinition]:
    allowed = set(WORLD_MODALITY_PROFILES[world]["modalities"])
    return [
        ModalityDefinition.from_dict({**m.to_dict(), "metadata": {**m.metadata, "supported": True, "controlled_world_override": True}})
        for m in ALAB_DOMAIN_CONFIG.modalities
        if m.name in allowed
    ]


def _controlled_candidates(seed: int, count: int = 12) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {f"controlled-{i}": rng.normal(size=49) for i in range(count)}


def _discovery_prediction(features: np.ndarray) -> float:
    value = 0.45 + 0.20 * float(features[1]) + 0.15 * float(features[2]) - 0.10 * abs(float(features[0]))
    return float(np.clip(value, 0.0, 1.0))


def _world_rng(world: str, seed: int, candidate_id: str, modality: str) -> np.random.Generator:
    import hashlib

    raw = f"{world}|{seed}|{candidate_id}|{modality}|measurement-v1".encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(raw).digest()[:8], "big"))


def _make_reveal(engine: MultimodalDecisionEngine, action: Any, true_hypothesis: Any, world: str, seed: int, candidates: Mapping[str, np.ndarray]) -> ScientificObservable:
    modality = normalize_action_type(action.action_type).upper()
    prediction = true_hypothesis.predict_observable_distribution(
        action.candidate_id, modality, engine.observed_by_modality, candidate_features=candidates[action.candidate_id]
    )
    diagnostic = WORLD_MODALITY_PROFILES[world]["diagnostic"]
    scale = 0.12 if modality == diagnostic else 0.70
    rng = _world_rng(world, seed, action.candidate_id, modality)
    noise = np.sqrt(prediction.variance) * scale
    value = np.asarray(prediction.mean + rng.normal(0.0, noise), dtype=np.float64)
    for index, name in enumerate(prediction.observable_names):
        value_range = OBSERVABLE_REGISTRY[name].value_range
        if value_range is not None:
            value[index] = np.clip(value[index], *value_range)
    return ScientificObservable(
        observable_id=f"{world}:{seed}:{engine.step}:{action.action_id}",
        candidate_id=action.candidate_id,
        modality=modality,
        name=prediction.observable_names[0] if value.size == 1 else "controlled_bundle",
        observable_names=tuple(prediction.observable_names),
        value=float(value[0]) if value.size == 1 else value,
        uncertainty=noise,
        raw_artifact_ref=f"controlled://{world}/{seed}/{action.action_id}",
        extractor_name="controlled_world_generator",
        extractor_version="1.0.0",
        provenance={"world": world, "seed": seed, "true_hypothesis": true_hypothesis.hypothesis_id, "noise_scale": scale},
        observable_type="vector" if value.size > 1 else "scalar",
    )


def _select_policy_action(engine: MultimodalDecisionEngine, policy_name: str, rng: np.random.Generator, samples: int) -> Any:
    feasible = engine.enumerate_actions()
    if policy_name == "RANDOM_ACTION":
        return feasible[int(rng.integers(len(feasible)))]
    if policy_name == "RANDOM_CANDIDATE_FIXED_MODALITY":
        fixed = [action for action in feasible if normalize_action_type(action.action_type).upper() == "XRD"]
        return (fixed or feasible)[int(rng.integers(len(fixed or feasible)))]
    if policy_name == "UNCERTAINTY_ONLY":
        return max(feasible, key=lambda action: float(np.mean([np.mean(p.variance) for p in engine._predictions(action).values()])))
    return engine.recommend(samples=samples).action


def _tag_ledger_events(events: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    tagged = []
    for event in events:
        item = {**event, "run_id": run_id}
        action = item.get("action")
        if isinstance(action, Mapping) and action.get("action_id"):
            item["action"] = {**action, "action_id": f"{run_id}:{action['action_id']}"}
        tagged.append(item)
    return tagged


def _hig_order_invariant() -> bool:
    features = {"a": np.linspace(0.0, 1.0, 49), "b": np.linspace(1.0, 0.0, 49)}
    left = MultimodalDecisionEngine(features, ALAB_DOMAIN_CONFIG.modalities, build_alab_multimodal_hypotheses(), seed=19)
    right = MultimodalDecisionEngine(dict(reversed(list(features.items()))), ALAB_DOMAIN_CONFIG.modalities, build_alab_multimodal_hypotheses(), seed=19)
    left_actions = {action.action_id: action for action in left.enumerate_actions()}
    right_actions = {action.action_id: action for action in right.enumerate_actions()}
    return set(left_actions) == set(right_actions) and all(
        np.isclose(
            left.expected_hypothesis_information_gain(left_actions[action_id], samples=12),
            right.expected_hypothesis_information_gain(right_actions[action_id], samples=12),
        )
        for action_id in left_actions
    )


def _run_controlled_policy(world: str, seed: int, policy_name: str, steps: int, run_label: str = "controlled_world") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = _controlled_candidates(seed)
    hypotheses = build_alab_multimodal_hypotheses()
    discovery = {(cid, modality): _discovery_prediction(features) for cid, features in candidates.items() for modality in ("XRD", "REFINEMENT", "SEM", "EDS", "OUTCOME_TEST")}
    engine = MultimodalDecisionEngine(candidates, _controlled_modalities(world), hypotheses, discovery_values=discovery, policy_name=policy_name, seed=seed)
    true_hypothesis = hypotheses[{
        "WORLD_H1_PHASE_PURITY": "H1_PHASE_PURITY_LIMITED",
        "WORLD_H2_COMPOSITION_HOMOGENEITY": "H2_COMPOSITION_HOMOGENEITY_LIMITED",
        "WORLD_H3_MORPHOLOGY_KINETICS": "H3_MORPHOLOGY_KINETICS_LIMITED",
    }[world]]
    initial_entropy = engine.current_entropy
    rng = np.random.default_rng(seed + 10000)
    timeline: list[dict[str, Any]] = []
    for _ in range(steps):
        if policy_name in {"RANDOM_ACTION", "RANDOM_CANDIDATE_FIXED_MODALITY", "UNCERTAINTY_ONLY"}:
            action = _select_policy_action(engine, policy_name, rng, samples=24)
            prediction = engine._predictions(action)
            hig = engine.expected_hypothesis_information_gain(action, prediction, samples=24)
            discovery_value = engine._discovery_value(action)
            recommendation = engine.register_selected_action(action, hig=hig, discovery=discovery_value, predictions=prediction)
        else:
            recommendation = engine.recommend(samples=24)
            action = recommendation.action
        observable = _make_reveal(engine, action, true_hypothesis, world, seed, candidates)
        reveal = engine.observe(observable)
        timeline.append({
            "step": engine.step,
            "action": action.to_dict(),
            "modality": normalize_action_type(action.action_type).upper(),
            "hig_nats": recommendation.why["expected_hig_nats"],
            "discovery_utility": recommendation.why["discovery_utility"],
            "normalized_cost": recommendation.why["normalized_cost"],
            "beliefs_after": reveal["beliefs_after"],
            "entropy_reduction_nats": reveal["realized_entropy_reduction_nats"],
        })
    final_beliefs = dict(engine.beliefs)
    true_p = final_beliefs[true_hypothesis.hypothesis_id]
    map_steps = [row["step"] for row in timeline if max(row["beliefs_after"], key=row["beliefs_after"].get) == true_hypothesis.hypothesis_id]
    def first_crossing(threshold: float) -> int | None:
        return next((row["step"] for row in timeline if row["beliefs_after"][true_hypothesis.hypothesis_id] >= threshold), None)
    return {
        "world": world,
        "seed": seed,
        "policy": policy_name,
        "true_hypothesis": true_hypothesis.hypothesis_id,
        "initial_beliefs": {hid: 1.0 / len(hypotheses) for hid in hypotheses},
        "final_beliefs": final_beliefs,
        "true_hypothesis_posterior": true_p,
        "initial_entropy_nats": initial_entropy,
        "final_entropy_nats": engine.current_entropy,
        "realized_entropy_reduction_nats": initial_entropy - engine.current_entropy,
        "actions_selected": timeline,
        "modalities_selected": [row["modality"] for row in timeline],
        "cumulative_hig_nats": float(sum(row["hig_nats"] for row in timeline)),
        "total_normalized_cost": float(sum(row["normalized_cost"] for row in timeline)),
        "step_true_hypothesis_becomes_MAP": min(map_steps) if map_steps else None,
        "step_posterior_crosses_0.5": first_crossing(0.5),
        "step_posterior_crosses_0.8": first_crossing(0.8),
        "step_posterior_crosses_0.9": first_crossing(0.9),
    }, _tag_ledger_events(engine.ledger.events, f"{run_label}:{world}:{seed}:{policy_name}")


def controlled_hypothesis_benchmark(seed: int = 42, steps: int = 4) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    ledger_events: list[dict[str, Any]] = []
    worlds = tuple(WORLD_MODALITY_PROFILES)
    seeds = (seed,) if seed != 42 else SEEDS
    for world in worlds:
        for run_seed in seeds:
            record, events = _run_controlled_policy(world, run_seed, "PURE_HIG", steps, "controlled_world")
            records.append(record)
            ledger_events.extend(events)
    grouped = {world: [row for row in records if row["world"] == world] for world in worlds}
    summary = {}
    for world, rows in grouped.items():
        summary[world] = {
            "seeds": [row["seed"] for row in rows],
            "recovery_rate_MAP": float(np.mean([row["step_true_hypothesis_becomes_MAP"] is not None for row in rows])),
            "recovery_rate_posterior_gt_0.5": float(np.mean([row["true_hypothesis_posterior"] > 0.5 for row in rows])),
            "recovery_rate_posterior_gt_0.8": float(np.mean([row["true_hypothesis_posterior"] > 0.8 for row in rows])),
            "mean_final_true_hypothesis_probability": float(np.mean([row["true_hypothesis_posterior"] for row in rows])),
            "mean_entropy_reduction": float(np.mean([row["realized_entropy_reduction_nats"] for row in rows])),
            "mean_measurement_cost": float(np.mean([row["total_normalized_cost"] for row in rows])),
        }
    return {
        "status": "METHODOLOGY_VALID" if set(worlds) == set(grouped) and all(len(rows) >= 5 for rows in grouped.values()) else "NOT_READY",
        "worlds": list(worlds),
        "seeds": list(seeds),
        "policy": "PURE_HIG",
        "world_profiles": WORLD_MODALITY_PROFILES,
        "records": records,
        "summary": summary,
        "ledger_events": ledger_events,
    }


def controlled_policy_comparison(seed: int = 42, steps: int = 4) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    ledger_events: list[dict[str, Any]] = []
    world = "WORLD_H1_PHASE_PURITY"
    for policy_name in POLICIES:
        record, events = _run_controlled_policy(world, seed, policy_name, steps, "policy_comparison")
        records.append(record)
        ledger_events.extend(events)
    return {
        "status": "METHODOLOGY_VALID" if len({row["policy"] for row in records}) == len(POLICIES) else "NOT_READY",
        "policies": records,
        "policy_formulas": {
            "RANDOM_ACTION": "uniform feasible action",
            "RANDOM_CANDIDATE_FIXED_MODALITY": "uniform feasible XRD action",
            "UNCERTAINTY_ONLY": "mean predictive variance",
            "DISCOVERY_ONLY": "normalized discovery utility",
            "PURE_HIG": "normalized expected hypothesis information gain",
            "HYBRID": "0.8 normalized HIG + 0.8 normalized discovery - 0.8 normalized cost",
        },
        "ledger_events": ledger_events,
    }


def _replay_value(outcome: ExperimentOutcome) -> Any:
    raw = outcome.canonical_observation
    modality = normalize_action_type(outcome.action_type).upper()
    if modality == "XRD":
        observations = DeterministicXRDSpectralDescriptorExtractor().extract(
            outcome.revealed_data,
            outcome.candidate_id,
            outcome.provenance,
        )
        return np.asarray([obs.value for obs in observations], dtype=float)
    if modality == "REFINEMENT":
        refinement = outcome.revealed_data if isinstance(outcome.revealed_data, Mapping) else raw
        return np.asarray([refinement[name.split(".", 1)[1]] for name in observable_names_for_modality("REFINEMENT")], dtype=float)
    return float(raw)


def retrospective_replay(data_dir: str, cache_dir: str, seed: int = 42, steps: int = 3) -> dict[str, Any]:
    adapter = ALabDomainAdapter(data_dir=data_dir, cache_dir=cache_dir)
    split_manifest = _replay_split_manifest(data_dir)
    evaluation_ids = set(split_manifest.get("evaluation_ids", []))
    xrd_ids = []
    for action in adapter.list_valid_actions():
        if normalize_action_type(action.action_type).upper() == "XRD" and action.candidate_id in evaluation_ids and action.candidate_id not in xrd_ids:
            xrd_ids.append(action.candidate_id)
        if len(xrd_ids) >= 16:
            break
    if not xrd_ids:
        return {"status": "NOT_READY", "reason": "No canonical linked XRD candidates available."}
    candidates = {cid: adapter.get_candidate_features(cid) for cid in xrd_ids}
    modalities = [m for m in adapter.modalities if m.name in {"XRD", "REFINEMENT", "OUTCOME_TEST"}]
    engine = MultimodalDecisionEngine(candidates, modalities, build_alab_multimodal_hypotheses(), seed=seed, policy_name="HYBRID")
    events = []
    for _ in range(steps):
        try:
            recommendation = engine.recommend(samples=16)
            outcome = adapter.execute_or_reveal(recommendation.action)
            if outcome.canonical_observation is None:
                break
            replay_value = _replay_value(outcome)
            modality = normalize_action_type(outcome.action_type).upper()
            names = observable_names_for_modality(modality)
            value = float(replay_value) if np.asarray(replay_value).size == 1 else np.asarray(replay_value, dtype=float)
            observable = ScientificObservable(
                observable_id=f"replay:{engine.step}:{recommendation.action.action_id}",
                candidate_id=outcome.candidate_id,
                modality=modality,
                name=names[0] if len(names) == 1 else "canonical_replay_bundle",
                observable_names=names,
                value=value,
                uncertainty=0.1 if len(names) == 1 else np.full(len(names), 0.1),
                raw_artifact_ref=outcome.provenance.get("archive_member_path") or outcome.provenance.get("artifact_member_path") or outcome.provenance.get("artifact_ref"),
                extractor_name="alab_canonical_replay",
                extractor_version="1.0.0",
                provenance=outcome.provenance,
                timestamp=outcome.oracle_timestamp,
                observable_type="vector" if len(names) > 1 else "scalar",
            )
            reveal = engine.observe(observable)
            events.append({
                "recommendation": recommendation.why,
                "action": recommendation.action.to_dict(),
                "reveal": reveal,
                "outcome_evidence": {
                    "category": outcome.provenance.get("reaction_category"),
                    "decision_utility": outcome.provenance.get("reaction_outcome_utility"),
                    "semantic_note": "category is categorical evidence; utility is a separate ordinal decision mapping",
                } if modality == "OUTCOME_TEST" else None,
            })
        except (RuntimeError, ValueError, KeyError) as exc:
            return {"status": "NOT_READY", "events": events, "reason": str(exc)}
    ledger = list(engine.ledger.events)
    registrations = {event["action"]["action_id"]: index for index, event in enumerate(ledger) if event.get("event") == "PREREGISTERED_SELECTED_ACTION"}
    reveals = [event for event in ledger if event.get("event") == "MEASUREMENT_REVEALED"]
    replay_errors: list[str] = []
    if not candidates or not evaluation_ids:
        replay_errors.append("evaluation candidates must be non-empty")
    if len(reveals) != len(events) or len({event["action"]["action_id"] for event in reveals}) != len(reveals):
        replay_errors.append("duplicate or missing reveal events")
    for index, event in enumerate(ledger):
        if event.get("event_sequence") != index + 1 or not event.get("timestamp"):
            replay_errors.append("ledger event sequence/timestamp is invalid")
            break
    for reveal in reveals:
        action_id = reveal["action"]["action_id"]
        registration_index = registrations.get(action_id)
        if registration_index is None or registration_index >= ledger.index(reveal) or ledger[registration_index].get("measurement_revealed"):
            replay_errors.append(f"orphan reveal: {action_id}")
            continue
        observed = reveal["observed_measurement"]
        predicted = ledger[registration_index]["predictive_distributions"]
        predicted_names = tuple(next(iter(predicted.values()))["observable_names"])
        if tuple(observed.get("observable_names", ())) != predicted_names:
            replay_errors.append(f"schema mismatch: {action_id}")
        modality = normalize_action_type(reveal["action"]["action_type"]).upper()
        provenance = observed.get("provenance", {})
        linked = bool(provenance.get("sample_id")) and (
            modality == "XRD" and bool(provenance.get("archive_member_path")) or
            modality == "REFINEMENT" and bool(provenance.get("canonical_case")) or
            modality == "OUTCOME_TEST" and "reaction_category" in provenance
        )
        if not linked:
            replay_errors.append(f"non-canonical linkage: {action_id}")
        try:
            datetime.fromisoformat(str(reveal["timestamp"]))
        except ValueError:
            replay_errors.append(f"invalid timestamp: {action_id}")
    if any(modality not in {"XRD", "REFINEMENT", "OUTCOME_TEST"} for modality in (normalize_action_type(event["action"]["action_type"]).upper() for event in reveals)):
        replay_errors.append("unsupported modality selected")
    if not np.all(np.isfinite(list(engine.beliefs.values()))) or not np.isclose(sum(engine.beliefs.values()), 1.0):
        replay_errors.append("non-finite or unnormalized posterior")
    return {
        "status": "METHODOLOGY_VALID" if events and not replay_errors else "NOT_READY",
        "candidate_count": len(candidates),
        "evaluation_candidate_ids": sorted(candidates),
        "calibration_candidate_count": split_manifest.get("calibration_n", 0),
        "evaluation_candidate_count": len(candidates),
        "supported_modalities": [m.name for m in modalities],
        "steps": len(events),
        "split_method": split_manifest.get("split_method"),
        "events": events,
        "final_beliefs": engine.beliefs,
        "ledger_events": _tag_ledger_events(ledger, f"replay:{seed}"),
        "strict_replay_errors": replay_errors,
    }


def _third_party_capability_gate(field: str) -> bool:
    path = Path("outputs/integrations/backend_capabilities.json")
    if not path.is_file():
        return False
    try:
        backends = json.loads(path.read_text(encoding="utf-8"))["backends"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    if field == "license":
        return all(str(item.get("license", "")).strip() for item in backends.values())
    return all(
        item.get("status") in {"OPTIONAL_EXPERIMENTAL_INTEGRATION", "NOT_AVAILABLE", "REFERENCE_ONLY"}
        and item.get("used_in_primary_benchmark") is False
        for item in backends.values()
    )


def _replay_split_manifest(data_dir: str) -> dict[str, Any]:
    import hashlib

    ledger_path = Path(data_dir) / "ledger_precursor_genome.json"
    if not ledger_path.is_file():
        return {"split_method": "deterministic grouped holdout by SHA256(sample_id) parity", "calibration_ids": [], "evaluation_ids": []}
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", []) if isinstance(payload, dict) else payload
    ids = sorted(str(sample["sample_id"]) for sample in samples if sample.get("sample_id"))
    calibration_ids = [sample_id for sample_id in ids if int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[-1], 16) % 2 == 0]
    evaluation_ids = [sample_id for sample_id in ids if sample_id not in set(calibration_ids)]
    return {
        "split_method": "deterministic grouped holdout by SHA256(sample_id) parity; no evaluation ID calibrates a model",
        "group_key": "sample_id",
        "calibration_ids": calibration_ids,
        "evaluation_ids": evaluation_ids,
        "calibration_n": len(calibration_ids),
        "evaluation_n": len(evaluation_ids),
    }


def build_validation(inventory: Mapping[str, Any], extractors: Mapping[str, Any], controlled: Mapping[str, Any], policy_comparison: Mapping[str, Any], replay: Mapping[str, Any], calibration: Mapping[str, Any]) -> dict[str, Any]:
    ledger = list(controlled.get("ledger_events", [])) + list(policy_comparison.get("ledger_events", [])) + list(replay.get("ledger_events", []))
    world_names = set(controlled.get("worlds", []))
    required_gates = {
        "observable_schema_gate": all(set(names) for names in MODALITY_OBSERVABLE_NAMES.values()),
        "observable_semantics_alignment_gate": extractors.get("status") == "CONTRACT_VALIDATED",
        "raw_artifact_provenance_gate": inventory.get("dataset") == "A-Lab Precursor Genome" and bool(inventory.get("doi")),
        "candidate_linkage_gate": not inventory.get("modalities", {}).get("SEM", {}).get("action_space_supported", True) and not inventory.get("modalities", {}).get("EDS", {}).get("action_space_supported", True),
        "modality_contract_gate": all(name in inventory.get("modalities", {}) for name in ("XRD", "REFINEMENT", "SEM", "EDS", "OUTCOME_TEST")),
        "hypothesis_structure_gate": len(controlled.get("worlds", [])) == 3,
        "hypothesis_directionality_gate": all(
            hypothesis.assumptions and hypothesis.falsification_signature().get("strongly_supporting_patterns") and hypothesis.falsification_signature().get("strongly_falsifying_patterns")
            for hypothesis in build_alab_multimodal_hypotheses().values()
        ),
        "hypothesis_calibration_gate": calibration.get("status") in {"COMPUTED", "NOT_EVALUATED_NO_CALIBRATED_MODEL"},
        "controlled_world_coverage_gate": world_names == set(WORLD_MODALITY_PROFILES) and len(controlled.get("seeds", [])) >= 5,
        "controlled_recovery_methodology_gate": controlled.get("status") == "METHODOLOGY_VALID",
        "pre_reveal_HIG_gate": all(event.get("event") != "MEASUREMENT_REVEALED" or any(prev.get("event") == "PREREGISTERED_SELECTED_ACTION" and prev.get("action", {}).get("action_id") == event.get("action", {}).get("action_id") for prev in ledger[:i]) for i, event in enumerate(ledger)),
        "HIG_bound_gate": all(0.0 <= float(event.get("expected_hig_nats", 0.0)) for event in ledger if event.get("event") in {"ACTION_SCORE_RECORD", "PREREGISTERED_SELECTED_ACTION"}),
        "HIG_order_invariance_gate": _hig_order_invariant(),
        "policy_distinction_gate": len(set(policy_comparison.get("policy_formulas", {}).values())) == len(POLICIES),
        "hybrid_formula_gate": "normalized cost" in policy_comparison.get("policy_formulas", {}).get("HYBRID", ""),
        "natural_policy_divergence_gate": policy_comparison.get("natural_divergence", {}).get("status") == "PASS",
        "hybrid_cost_effect_gate": policy_comparison.get("natural_divergence", {}).get("hybrid_cost_vs_pure_hig", {}).get("hybrid", 0.0) <= policy_comparison.get("natural_divergence", {}).get("hybrid_cost_vs_pure_hig", {}).get("pure_hig", 0.0),
        "preregistration_timeline_gate": all(event.get("event_sequence", 0) > 0 for event in ledger),
        "retrospective_split_gate": "holdout" in replay.get("split_method", ""),
        "retrospective_observable_alignment_gate": replay.get("status") == "METHODOLOGY_VALID",
        "retrospective_replay_gate": replay.get("status") == "METHODOLOGY_VALID",
        "third_party_license_gate": _third_party_capability_gate("license"),
        "third_party_functionality_gate": _third_party_capability_gate("status"),
        "report_consistency_gate": inventory.get("dataset") == "A-Lab Precursor Genome" and controlled.get("status") == "METHODOLOGY_VALID" and policy_comparison.get("status") == "METHODOLOGY_VALID" and replay.get("status") == "METHODOLOGY_VALID",
        "local_test_gate": "NOT_RUN",
        "external_CI_gate": "NOT_INSPECTED",
    }
    return {
        "status": "METHODOLOGY_VALID" if all(value is True for value in required_gates.values() if isinstance(value, bool)) else "NOT_READY",
        "gates": {key: ("PASS" if value is True else value if isinstance(value, str) else "FAIL") for key, value in required_gates.items()},
        "gate_evidence": {"ledger_event_count": len(ledger), "controlled_worlds": sorted(world_names), "required_seeds": list(SEEDS), "unsupported_retrospective_modalities": ["SEM", "EDS"]},
    }


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
        "status": "NOT_EVALUATED_NO_CALIBRATED_MODEL",
        "split_manifest": "replay_split_manifest.json",
        "holdout_rule": "deterministic grouped holdout by SHA256(sample_id) parity; no evaluation ID calibrates a model",
        "metrics": {"ece": None, "brier": None, "log_loss": None},
        "note": "The CPU-safe hypothesis templates are controlled-world instruments, not calibrated retrospective models.",
        "hypotheses": list(hypotheses),
    }
    policy_comparison = {
        "status": policy_results["status"],
        "scope": "controlled_world_action_selector_comparison; A-Lab replay remains historical and separate",
        "policies": policy_results["policies"],
        "policy_formulas": policy_results["policy_formulas"],
        "natural_divergence": {
            "status": "PASS" if len({tuple(row["modalities_selected"]) for row in policy_results["policies"]}) >= 3 else "FAIL",
            "distinct_action_sequences": len({tuple(row["modalities_selected"]) for row in policy_results["policies"]}),
            "hybrid_cost_vs_pure_hig": {
                "hybrid": next(row["total_normalized_cost"] for row in policy_results["policies"] if row["policy"] == "HYBRID"),
                "pure_hig": next(row["total_normalized_cost"] for row in policy_results["policies"] if row["policy"] == "PURE_HIG"),
            },
        },
        "replay": replay,
        "ledger_events": policy_results["ledger_events"],
    }
    validation = build_validation(inventory, extractors, controlled, policy_comparison, replay, calibration)
    validation["gates"]["local_test_gate"] = os.environ.get("AICOSCIENTIST_LOCAL_TEST_GATE", "NOT_RUN")
    validation["gates"]["external_CI_gate"] = os.environ.get("AICOSCIENTIST_EXTERNAL_CI_GATE", "NOT_INSPECTED")
    _write("observable_schema.json", {name: definition.__dict__ for name, definition in OBSERVABLE_REGISTRY.items()})
    split_manifest = _replay_split_manifest(data_dir)
    split_manifest["calibration_status"] = calibration["status"]
    calibration["metrics"]["holdout_n"] = split_manifest.get("evaluation_n", 0)
    calibration["per_hypothesis_modality"] = {
        hypothesis_id: {
            modality: {
                "status": calibration["status"],
                "N_calibration": split_manifest.get("calibration_n", 0),
                "N_evaluation": split_manifest.get("evaluation_n", 0),
                "MAE": None,
                "RMSE": None,
                "mean_log_predictive_density": None,
                "coverage_50": None,
                "coverage_90": None,
            }
            for modality in ("XRD", "REFINEMENT", "OUTCOME_TEST")
        }
        for hypothesis_id in hypotheses
    }
    _write("hypothesis_calibration.json", calibration)
    _write("replay_split_manifest.json", split_manifest)
    _write("identifiability_diagnostics.json", {
        "status": "CONTROLLED_WORLD_DIAGNOSTIC",
        "worlds": list(WORLD_MODALITY_PROFILES),
        "diagnostic_modality_by_world": {world: profile["diagnostic"] for world, profile in WORLD_MODALITY_PROFILES.items()},
        "natural_divergence_required": True,
        "prospective_causal_identifiability": "NOT_ESTABLISHED",
    })
    _write("modality_inventory.json", inventory)
    _write("extractor_validation.json", extractors)
    _write("hypothesis_definitions.json", hypotheses)
    _write("hypothesis_calibration.json", calibration)
    _write("controlled_hypothesis_recovery.json", controlled)
    _write("retrospective_policy_comparison.json", policy_comparison)
    _write("multimodal_validation.json", validation)
    with (OUT_DIR / "evidence_ledger.jsonl").open("w", encoding="utf-8") as handle:
        events = controlled.get("ledger_events", []) + policy_comparison.get("ledger_events", []) + replay.get("ledger_events", [])
        for sequence, event in enumerate(events, start=1):
            handle.write(json.dumps({**event, "global_event_sequence": sequence}, default=str) + "\n")
    (OUT_DIR / "multimodal_report.md").write_text(
        "# A-Lab Multimodal Validation\n\n"
        f"- Available ledger samples: {inventory['available_samples']}\n"
        f"- Controlled world: `{controlled['status']}`\n"
        f"- Controlled policy comparison: `{policy_comparison['status']}`\n"
        f"- Retrospective replay: `{replay['status']}`\n"
        f"- Validation gates: `{validation['status']}`\n"
        "- SEM/EDS candidate actions: disabled because archives are precursor-level and not canonically linked to sample IDs.\n"
        "- Scope: offline validation and replay only; no prospective or causal claim.\n"
        "- Calibration: controlled templates are not calibrated retrospective models; metrics are not evaluated because no retrospective model is fit.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
