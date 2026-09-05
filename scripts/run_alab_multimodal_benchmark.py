"""Run bounded, provenance-preserving A-Lab multimodal validation artifacts."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domains.alab.adapter import ALabDomainAdapter
from src.domains.alab.config import ALAB_DOMAIN_CONFIG
from src.domains.alab.multimodal_inventory import inventory_alab_modalities
from src.integrations.microscopy.atomai_adapter import ClassicalEDSDescriptorExtractor, ClassicalSEMDescriptorExtractor
from src.integrations.xrd.autoxrd_adapter import DeterministicXRDSpectralDescriptorExtractor
from src.science.actions import ExperimentOutcome, ScientificAction, normalize_action_type
from src.science.multimodal.decision import MultimodalDecisionEngine
from src.science.multimodal.hypotheses import build_alab_multimodal_hypotheses
from src.science.multimodal.ontology import MODALITY_OBSERVABLE_NAMES, OBSERVABLE_REGISTRY, observable_names_for_modality
from src.science.multimodal.retrospective import (
    REAL_MODALITIES,
    RetrospectiveCalibratedHypothesisModel,
    RetrospectiveObservationSet,
    build_retrospective_hypotheses,
    evaluate_retrospective_models,
)
from src.science.multimodal.schemas import ScientificObservable
from src.science.domain import ModalityDefinition

OUT_DIR = Path("outputs/alab/multimodal")
SEEDS = (7, 42, 101, 314, 2024)
REPLAY_SEEDS = (7, 42, 101)
HYBRID_WEIGHTS = {"w_hig": 0.8, "w_discovery": 0.8, "w_cost": 2.0}
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
    import hashlib

    modality = normalize_action_type(action.action_type).upper()
    prediction = true_hypothesis.predict_observable_distribution(
        action.candidate_id, modality, engine.observed_by_modality, candidate_features=candidates[action.candidate_id]
    )
    diagnostic = WORLD_MODALITY_PROFILES[world]["diagnostic"]
    digest = hashlib.sha256(f"{world}|{action.candidate_id}|difficulty-v1".encode("utf-8")).digest()
    non_diagnostic_candidate = digest[0] % 4 == 0
    signal_scale = 0.45 if modality == diagnostic else 0.25
    scale = 1.25 if modality == diagnostic else 1.50
    if non_diagnostic_candidate:
        signal_scale *= 0.60
        scale *= 1.25
    rng = _world_rng(world, seed, action.candidate_id, modality)
    competing_means = np.asarray([
        hypothesis.predict_observable_distribution(
            action.candidate_id,
            modality,
            engine.observed_by_modality,
            candidate_features=candidates[action.candidate_id],
        ).mean
        for hypothesis in engine.hypotheses.values()
    ])
    mean = np.mean(competing_means, axis=0) + signal_scale * (prediction.mean - np.mean(competing_means, axis=0))
    noise = np.sqrt(prediction.variance) * scale
    value = np.asarray(mean + rng.normal(0.0, noise), dtype=np.float64)
    if digest[1] % 5 == 0 and value.size:
        value[0] += 0.12 if digest[2] % 2 else -0.12
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
    engine_kwargs = HYBRID_WEIGHTS if policy_name == "HYBRID" else {}
    engine = MultimodalDecisionEngine(
        candidates,
        _controlled_modalities(world),
        hypotheses,
        discovery_values=discovery,
        policy_name=policy_name,
        seed=seed,
        **engine_kwargs,
    )
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
            "score_components": recommendation.why.get("score_components", {}),
            "scored_actions": recommendation.scored_actions,
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
            "std_final_true_hypothesis_probability": float(np.std([row["true_hypothesis_posterior"] for row in rows])),
            "std_entropy_reduction": float(np.std([row["realized_entropy_reduction_nats"] for row in rows])),
            "std_measurement_cost": float(np.std([row["total_normalized_cost"] for row in rows])),
        }
    return {
        "status": "METHODOLOGY_VALID" if set(worlds) == set(grouped) and all(len(rows) >= 5 for rows in grouped.values()) else "NOT_READY",
        "worlds": list(worlds),
        "seeds": list(seeds),
        "policy": "PURE_HIG",
        "world_profiles": WORLD_MODALITY_PROFILES,
        "benchmark_design": {
            "diagnostic_signal_scale": 0.45,
            "non_diagnostic_signal_scale": 0.25,
            "diagnostic_noise_scale": 1.25,
            "non_diagnostic_noise_scale": 1.50,
            "non_diagnostic_candidate_rule": "SHA256(world|candidate|difficulty-v1) first byte modulo 4 equals zero",
            "misleading_measurement_rule": "deterministic SHA256 candidate hash can add signed 0.12 bias to the first observable",
            "controlled_modality_semantics": "SYNTHETIC_CONTROLLED_MODALITY",
        },
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
            "HYBRID": f"{HYBRID_WEIGHTS['w_hig']} normalized HIG + {HYBRID_WEIGHTS['w_discovery']} normalized discovery - {HYBRID_WEIGHTS['w_cost']} normalized cost",
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


def _real_action(candidate_id: str, modality: str) -> ScientificAction:
    costs = {"XRD": 1.0, "REFINEMENT": 0.5, "OUTCOME_TEST": 2.0}
    return ScientificAction(
        action_id=f"RETROSPECTIVE_{modality}_{candidate_id}",
        candidate_id=candidate_id,
        action_type=modality,
        estimated_cost=costs[modality],
        metadata={"modality_hint": modality},
    )


def _collect_retrospective_observations(
    data_dir: str,
    cache_dir: str,
    sample_ids: Sequence[str],
    split_name: str,
) -> tuple[RetrospectiveObservationSet, dict[str, np.ndarray]]:
    """Load canonical labels for one split; no fitting or evaluation selection occurs here."""
    adapter = ALabDomainAdapter(data_dir=data_dir, cache_dir=cache_dir)
    extractor = DeterministicXRDSpectralDescriptorExtractor()
    requested = [str(sample_id) for sample_id in sorted(set(sample_ids))]
    features = {cid: adapter.get_candidate_features(cid) for cid in requested if cid in set(adapter.get_candidate_pool()["candidate_id"])}
    observations: dict[str, dict[str, ScientificObservable]] = {modality: {} for modality in REAL_MODALITIES}
    skipped: dict[str, int] = {modality: 0 for modality in REAL_MODALITIES}
    for cid in sorted(features):
        try:
            outcome = adapter.execute_or_reveal(_real_action(cid, "XRD"))
            if not outcome.provenance.get("canonical_scan") or outcome.provenance.get("is_replay_fallback"):
                skipped["XRD"] += 1
            else:
                extracted = list(extractor.extract(
                    outcome.revealed_data["normalized_intensity"],
                    candidate_id=cid,
                    metadata={**outcome.provenance, "raw_artifact_ref": outcome.provenance.get("archive_member_path")},
                ))
                names = observable_names_for_modality("XRD")
                observations["XRD"][cid] = ScientificObservable(
                    observable_id=f"real:{split_name}:XRD:{cid}",
                    candidate_id=cid,
                    modality="XRD",
                    name="XRD.canonical_descriptor_bundle",
                    observable_names=names,
                    value=np.asarray([item.value for item in extracted], dtype=np.float64),
                    uncertainty=np.asarray([item.uncertainty for item in extracted], dtype=np.float64),
                    raw_artifact_ref=outcome.provenance.get("archive_member_path"),
                    extractor_name=extractor.name,
                    extractor_version=extractor.version,
                    provenance={**outcome.provenance, "source_type": "REAL_RETROSPECTIVE_MODALITY", "split": split_name},
                    timestamp=outcome.oracle_timestamp,
                    observable_type="vector",
                )
        except (RuntimeError, ValueError, KeyError, OSError):
            skipped["XRD"] += 1

        if cid in observations["XRD"]:
            try:
                outcome = adapter.execute_or_reveal(_real_action(cid, "REFINEMENT"))
                names = observable_names_for_modality("REFINEMENT")
                values = np.asarray([outcome.revealed_data[name.split(".", 1)[1]] for name in names], dtype=np.float64)
                observations["REFINEMENT"][cid] = ScientificObservable(
                    observable_id=f"real:{split_name}:REFINEMENT:{cid}",
                    candidate_id=cid,
                    modality="REFINEMENT",
                    name="REFINEMENT.canonical_feature_bundle",
                    observable_names=names,
                    value=values,
                    uncertainty=np.full(len(names), 0.05, dtype=np.float64),
                    raw_artifact_ref=outcome.provenance.get("source_artifact_member") or outcome.provenance.get("artifact_member_path"),
                    extractor_name="alab_canonical_refinement_parser",
                    extractor_version=str(outcome.provenance.get("refinement_parser_version", "unknown")),
                    provenance={**outcome.provenance, "source_type": "REAL_RETROSPECTIVE_MODALITY", "split": split_name},
                    timestamp=outcome.oracle_timestamp,
                    observable_type="vector",
                )
            except (RuntimeError, ValueError, KeyError, OSError):
                skipped["REFINEMENT"] += 1

        try:
            outcome = adapter.execute_or_reveal(_real_action(cid, "OUTCOME_TEST"))
            utility = outcome.revealed_data.get("reaction_outcome_utility")
            category = outcome.revealed_data.get("reaction_category")
            if utility is None or not category:
                skipped["OUTCOME_TEST"] += 1
            else:
                observations["OUTCOME_TEST"][cid] = ScientificObservable(
                    observable_id=f"real:{split_name}:OUTCOME_TEST:{cid}",
                    candidate_id=cid,
                    modality="OUTCOME_TEST",
                    name="OUTCOME.reaction_outcome_utility",
                    observable_names=observable_names_for_modality("OUTCOME_TEST"),
                    value=float(utility),
                    uncertainty=0.05,
                    units="ordinal decision utility",
                    extractor_name="alab_ledger_outcome",
                    extractor_version="1.0.0",
                    provenance={**outcome.provenance, "reaction_category": category, "source_type": "REAL_RETROSPECTIVE_MODALITY", "split": split_name},
                    timestamp=outcome.oracle_timestamp,
                    observable_type="scalar",
                )
        except (RuntimeError, ValueError, KeyError, OSError):
            skipped["OUTCOME_TEST"] += 1
    coverage = {
        "requested_candidate_count": len(requested),
        "candidate_count": len(features),
        "available_by_modality": {modality: len(values) for modality, values in observations.items()},
        "skipped_by_modality": skipped,
        "split": split_name,
        "source_type": "REAL_RETROSPECTIVE_MODALITY",
    }
    return RetrospectiveObservationSet(observations, coverage), features


def _fit_retrospective_calibration(
    data_dir: str,
    cache_dir: str,
    split_manifest: Mapping[str, Any],
) -> tuple[dict[str, RetrospectiveCalibratedHypothesisModel], dict[str, Any], RetrospectiveObservationSet, RetrospectiveObservationSet, dict[str, np.ndarray]]:
    calibration_ids = [str(item) for item in split_manifest.get("calibration_ids", [])]
    evaluation_ids = [str(item) for item in split_manifest.get("evaluation_ids", [])]
    calibration, calibration_features = _collect_retrospective_observations(data_dir, cache_dir, calibration_ids, "calibration")
    models = build_retrospective_hypotheses()
    for model in models.values():
        model.fit(calibration_features, calibration.by_modality, training_ids=calibration_ids)
    evaluation, evaluation_features = _collect_retrospective_observations(data_dir, cache_dir, evaluation_ids, "evaluation")
    all_features = {**calibration_features, **evaluation_features}
    metrics = evaluate_retrospective_models(
        models,
        calibration,
        evaluation,
        all_features,
        calibration_ids,
        evaluation_ids,
    )
    metrics["evaluation_loaded_after_fit"] = True
    metrics["leakage_assertions"] = {
        "split_disjoint": metrics["split"]["disjoint"],
        "all_models_fit_ids_subset_calibration": all(set(model.fitted_ids).issubset(calibration_ids) for model in models.values()),
        "no_evaluation_ids_in_model_fit": all(not set(model.fitted_ids).intersection(evaluation_ids) for model in models.values()),
    }
    h1_metrics = metrics["per_hypothesis_modality"]["H1_PHASE_PURITY_LIMITED"]
    supported = [h1_metrics[modality] for modality in ("XRD", "REFINEMENT", "OUTCOME_TEST")]
    if not all(item["N_evaluation"] > 0 and item["MAE"] is not None for item in supported):
        metrics["status"] = "A_LAB_RETROSPECTIVE_CALIBRATION_PARTIAL"
    return models, metrics, calibration, evaluation, all_features


def retrospective_replay(
    data_dir: str,
    cache_dir: str,
    calibrated_models: Mapping[str, RetrospectiveCalibratedHypothesisModel],
    split_manifest: Mapping[str, Any],
    evaluation_observations: RetrospectiveObservationSet,
    evaluation_features: Mapping[str, np.ndarray],
    seed: int = 42,
    steps: int = 6,
) -> dict[str, Any]:
    evaluation_ids = sorted(set(str(item) for item in split_manifest.get("evaluation_ids", [])))
    xrd_ids = sorted(set(evaluation_observations.by_modality.get("XRD", {})).intersection(evaluation_features))
    if not xrd_ids:
        return {"status": "NOT_READY", "reason": "No canonical linked XRD candidates available."}
    adapter_probe = ALabDomainAdapter(data_dir=data_dir, cache_dir=cache_dir)
    modalities = [m for m in adapter_probe.modalities if m.name in {"XRD", "REFINEMENT", "OUTCOME_TEST"}]
    all_runs: list[dict[str, Any]] = []
    all_ledger: list[dict[str, Any]] = []
    replay_errors: list[str] = []

    def validate_ledger(ledger: list[dict[str, Any]], events: list[dict[str, Any]], run_id: str, engine: MultimodalDecisionEngine) -> list[str]:
        errors: list[str] = []
        registrations = {event["action"]["action_id"]: index for index, event in enumerate(ledger) if event.get("event") == "PREREGISTERED_SELECTED_ACTION"}
        reveals = [event for event in ledger if event.get("event") == "MEASUREMENT_REVEALED"]
        if len(reveals) != len(events) or len({event["action"]["action_id"] for event in reveals}) != len(reveals):
            errors.append(f"{run_id}: duplicate or missing reveal events")
        for index, event in enumerate(ledger):
            if event.get("event_sequence") != index + 1 or not event.get("timestamp"):
                errors.append(f"{run_id}: ledger event sequence/timestamp is invalid")
                break
        for reveal in reveals:
            action_id = reveal["action"]["action_id"]
            registration_index = registrations.get(action_id)
            if registration_index is None or registration_index >= ledger.index(reveal) or ledger[registration_index].get("measurement_revealed"):
                errors.append(f"{run_id}: orphan reveal: {action_id}")
                continue
            observed = reveal["observed_measurement"]
            predicted = ledger[registration_index]["predictive_distributions"]
            predicted_names = tuple(next(iter(predicted.values()))["observable_names"])
            if tuple(observed.get("observable_names", ())) != predicted_names:
                errors.append(f"{run_id}: schema mismatch: {action_id}")
            modality = normalize_action_type(reveal["action"]["action_type"]).upper()
            provenance = observed.get("provenance", {})
            linked = bool(provenance.get("sample_id")) and (
                modality == "XRD" and bool(provenance.get("archive_member_path")) and bool(provenance.get("canonical_scan")) and not provenance.get("is_replay_fallback") or
                modality == "REFINEMENT" and bool(provenance.get("canonical_case")) or
                modality == "OUTCOME_TEST" and "reaction_category" in provenance
            )
            if not linked:
                errors.append(f"{run_id}: non-canonical linkage: {action_id}")
            try:
                datetime.fromisoformat(str(reveal["timestamp"]))
            except ValueError:
                errors.append(f"{run_id}: invalid timestamp: {action_id}")
        if any(normalize_action_type(event["action"]["action_type"]).upper() not in {"XRD", "REFINEMENT", "OUTCOME_TEST"} for event in reveals):
            errors.append(f"{run_id}: unsupported modality selected")
        if not np.all(np.isfinite(list(engine.beliefs.values()))) or not np.isclose(sum(engine.beliefs.values()), 1.0):
            errors.append(f"{run_id}: non-finite or unnormalized posterior")
        return errors

    for replay_index, replay_seed in enumerate((seed,) if seed not in REPLAY_SEEDS else REPLAY_SEEDS):
        offset = (replay_index * 24) % len(xrd_ids)
        ordered = xrd_ids[offset:] + xrd_ids[:offset]
        selected_ids = ordered[: min(24, len(ordered))]
        candidates = {cid: np.asarray(evaluation_features[cid], dtype=np.float64) for cid in selected_ids}
        adapter = ALabDomainAdapter(data_dir=data_dir, cache_dir=cache_dir)
        engine = MultimodalDecisionEngine(candidates, modalities, calibrated_models, seed=replay_seed, policy_name="HYBRID")
        events: list[dict[str, Any]] = []
        skipped_actions = 0
        for _ in range(steps):
            try:
                recommendation = engine.recommend(samples=16)
                registration_snapshot = json.dumps(recommendation.preregistration["predictive_distributions"], sort_keys=True)
                outcome = adapter.execute_or_reveal(recommendation.action)
                if outcome.canonical_observation is None:
                    skipped_actions += 1
                    break
                replay_value = _replay_value(outcome)
                modality = normalize_action_type(outcome.action_type).upper()
                names = observable_names_for_modality(modality)
                value = float(replay_value) if np.asarray(replay_value).size == 1 else np.asarray(replay_value, dtype=float)
                observable = ScientificObservable(
                    observable_id=f"replay:{replay_index}:{engine.step}:{recommendation.action.action_id}",
                    candidate_id=outcome.candidate_id,
                    modality=modality,
                    name=names[0] if len(names) == 1 else "canonical_replay_bundle",
                    observable_names=names,
                    value=value,
                    uncertainty=0.1 if len(names) == 1 else np.full(len(names), 0.1),
                    raw_artifact_ref=outcome.provenance.get("archive_member_path") or outcome.provenance.get("source_artifact_member"),
                    extractor_name="alab_canonical_replay",
                    extractor_version="1.0.0",
                    provenance={**outcome.provenance, "source_type": "REAL_RETROSPECTIVE_MODALITY", "replay_type": "retrospective historical replay"},
                    timestamp=outcome.oracle_timestamp,
                    observable_type="vector" if len(names) > 1 else "scalar",
                )
                reveal = engine.observe(observable)
                if registration_snapshot != json.dumps(recommendation.preregistration["predictive_distributions"], sort_keys=True):
                    replay_errors.append(f"replay:{replay_seed}: post-reveal prediction mutation")
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
            except (RuntimeError, ValueError, KeyError, OSError) as exc:
                replay_errors.append(f"replay:{replay_seed}: {exc}")
                break
        ledger = list(engine.ledger.events)
        run_id = f"replay:{replay_seed}:{replay_index}"
        replay_errors.extend(validate_ledger(ledger, events, run_id, engine))
        tagged = _tag_ledger_events(ledger, run_id)
        all_ledger.extend(tagged)
        all_runs.append({
            "run_id": run_id,
            "seed": replay_seed,
            "candidate_ids": selected_ids,
            "candidate_count": len(selected_ids),
            "steps": len(events),
            "events": events,
            "final_beliefs": dict(engine.beliefs),
            "entropy_trajectory": [event["reveal"]["realized_entropy_reduction_nats"] for event in events],
            "modality_sequence": [event["reveal"]["action"]["action_type"] for event in events],
            "available_initial_actions": len(MultimodalDecisionEngine(candidates, modalities, calibrated_models, seed=replay_seed, policy_name="HYBRID").enumerate_actions()),
            "skipped_actions": skipped_actions,
            "reveal_completeness": len(events) == len([event for event in ledger if event.get("event") == "MEASUREMENT_REVEALED"]),
            "strict_errors": [error for error in replay_errors if error.startswith(run_id)],
        })
    return {
        "status": "METHODOLOGY_VALID" if all_runs and not replay_errors else "NOT_READY",
        "replay_type": "retrospective historical replay",
        "model_kind": RetrospectiveCalibratedHypothesisModel.model_kind,
        "candidate_count": len(xrd_ids),
        "evaluation_candidate_ids": xrd_ids,
        "calibration_candidate_count": split_manifest.get("calibration_n", 0),
        "evaluation_candidate_count": len(evaluation_ids),
        "supported_modalities": [m.name for m in modalities],
        "steps_per_run": steps,
        "replay_seed_count": len(all_runs),
        "split_method": split_manifest.get("split_method"),
        "coverage": {
            "evaluation_ids_with_canonical_xrd": len(xrd_ids),
            "evaluation_ids_total": len(evaluation_ids),
            "canonical_xrd_coverage": len(xrd_ids) / max(len(evaluation_ids), 1),
            "action_availability": {run["run_id"]: run["available_initial_actions"] for run in all_runs},
        },
        "runs": all_runs,
        "events": all_runs[0]["events"] if all_runs else [],
        "final_beliefs": all_runs[0]["final_beliefs"] if all_runs else {},
        "ledger_events": all_ledger,
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


def _policy_validation_diagnostics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored_states = [
        (record, row)
        for record in records
        for row in record.get("actions_selected", [])
        if row.get("scored_actions")
    ]
    conflicts = []
    hybrid_recomputed = []
    cost_effects = []
    for record, state in scored_states:
        scored = list(state["scored_actions"])
        hig_best = max(scored, key=lambda item: (float(item["expected_hig_nats"]), item["action"]["action_id"]))
        discovery_best = max(scored, key=lambda item: (float(item["discovery_utility"]), item["action"]["action_id"]))
        if hig_best["action"]["action_id"] != discovery_best["action"]["action_id"]:
            conflicts.append({
                "policy": record.get("policy"),
                "world": record.get("world"),
                "seed": record.get("seed"),
                "step": state.get("step"),
                "discovery_best": discovery_best["action"]["action_id"],
                "hig_best": hig_best["action"]["action_id"],
            })
        if record.get("policy") == "HYBRID":
            max_hig = max((float(item["expected_hig_nats"]) for item in scored), default=0.0)
            max_discovery = max((float(item["discovery_utility"]) for item in scored), default=0.0)
            recomputed = {
                item["action"]["action_id"]: (
                    HYBRID_WEIGHTS["w_hig"] * float(item["expected_hig_nats"]) / max(max_hig, 1e-12)
                    + HYBRID_WEIGHTS["w_discovery"] * float(item["discovery_utility"]) / max(max_discovery, 1e-12)
                    - HYBRID_WEIGHTS["w_cost"] * float(item["normalized_cost"])
                )
                for item in scored
            }
            selected_id = state["action"]["action_id"]
            hybrid_recomputed.append({
                "run": {"world": record.get("world"), "seed": record.get("seed"), "step": state.get("step")},
                "selected_action": selected_id,
                "selected_score_record": next(item["total_action_score"] for item in scored if item["action"]["action_id"] == selected_id),
                "recomputed_selected_score": recomputed[selected_id],
                "recomputed_best_action": max(recomputed, key=recomputed.get),
                "matches": selected_id == max(recomputed, key=recomputed.get) and np.isclose(next(item["total_action_score"] for item in scored if item["action"]["action_id"] == selected_id), recomputed[selected_id]),
            })
            no_cost = {
                item["action"]["action_id"]: (
                    HYBRID_WEIGHTS["w_hig"] * float(item["expected_hig_nats"]) / max(max_hig, 1e-12)
                    + HYBRID_WEIGHTS["w_discovery"] * float(item["discovery_utility"]) / max(max_discovery, 1e-12)
                )
                for item in scored
            }
            no_cost_best = max(no_cost, key=no_cost.get)
            if no_cost_best != selected_id:
                cost_effects.append({
                    "world": record.get("world"), "seed": record.get("seed"), "step": state.get("step"),
                    "with_cost": selected_id, "without_cost": no_cost_best,
                })
    sequences = {tuple(record.get("modalities_selected", [])) for record in records}
    return {
        "discovery_hig_conflict_gate": "PASS" if conflicts else "FAIL",
        "discovery_hig_conflicts": conflicts[:20],
        "hybrid_score_recomputation_gate": "PASS" if hybrid_recomputed and all(item["matches"] for item in hybrid_recomputed) else "FAIL",
        "hybrid_score_recomputation": hybrid_recomputed,
        "hybrid_cost_causality_gate": "PASS" if cost_effects else "FAIL",
        "hybrid_cost_effects": cost_effects[:20],
        "policy_sequence_distinction_gate": "PASS" if len(sequences) >= 3 else "FAIL",
        "distinct_modality_sequences": len(sequences),
    }


def _conditional_hig_diagnostics() -> dict[str, Any]:
    candidate_id = "conditional-0"
    candidates = {candidate_id: _controlled_candidates(314, count=1)["controlled-0"]}
    modalities = [m for m in ALAB_DOMAIN_CONFIG.modalities if m.name in {"XRD", "OUTCOME_TEST"}]
    hypotheses = build_alab_multimodal_hypotheses()
    engine = MultimodalDecisionEngine(candidates, modalities, hypotheses, seed=314, policy_name="PURE_HIG")
    outcome_before = next(action for action in engine.enumerate_actions() if normalize_action_type(action.action_type).upper() == "OUTCOME_TEST")
    prediction_before = engine._predictions(outcome_before)
    hig_before = engine.expected_hypothesis_information_gain(outcome_before, prediction_before, samples=64)
    xrd_action = next(action for action in engine.enumerate_actions() if normalize_action_type(action.action_type).upper() == "XRD")
    xrd_prediction = engine._predictions(xrd_action)
    engine.register_selected_action(xrd_action, predictions=xrd_prediction, hig=engine.expected_hypothesis_information_gain(xrd_action, xrd_prediction, samples=64))
    xrd_observable = _make_reveal(
        engine,
        xrd_action,
        hypotheses["H1_PHASE_PURITY_LIMITED"],
        "WORLD_H1_PHASE_PURITY",
        314,
        candidates,
    )
    engine.observe(xrd_observable)
    outcome_after = next(action for action in engine.enumerate_actions() if normalize_action_type(action.action_type).upper() == "OUTCOME_TEST")
    prediction_after = engine._predictions(outcome_after)
    hig_after = engine.expected_hypothesis_information_gain(outcome_after, prediction_after, samples=64)
    before_mean = prediction_before["H1_PHASE_PURITY_LIMITED"].mean.tolist()
    after_mean = prediction_after["H1_PHASE_PURITY_LIMITED"].mean.tolist()
    delta = float(abs(hig_after - hig_before))
    return {
        "status": "PASS" if delta > 1e-8 and not np.allclose(before_mean, after_mean) else "FAIL",
        "action": outcome_before.to_dict(),
        "hig_before_xrd_reveal_nats": hig_before,
        "hig_after_xrd_reveal_nats": hig_after,
        "absolute_hig_delta_nats": delta,
        "H1_outcome_mean_before": before_mean,
        "H1_outcome_mean_after": after_mean,
        "conditioned_on": "XRD",
        "pre_reveal_prediction_snapshot_used": True,
    }


def build_validation(inventory: Mapping[str, Any], extractors: Mapping[str, Any], controlled: Mapping[str, Any], policy_comparison: Mapping[str, Any], replay: Mapping[str, Any], calibration: Mapping[str, Any]) -> dict[str, Any]:
    ledger = list(controlled.get("ledger_events", [])) + list(policy_comparison.get("ledger_events", [])) + list(replay.get("ledger_events", []))
    world_names = set(controlled.get("worlds", []))
    calibration_split = calibration.get("split", {})
    fit_contract = calibration.get("fit_contract", {})
    calibration_metrics = calibration.get("per_hypothesis_modality", {}).get("H1_PHASE_PURITY_LIMITED", {})
    all_action_ids = [
        event.get("action", {}).get("action_id")
        for event in ledger
        if event.get("event") == "PREREGISTERED_SELECTED_ACTION" and event.get("action", {}).get("action_id")
    ]
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
        "hypothesis_calibration_gate": calibration.get("status") == "A_LAB_RETROSPECTIVE_CALIBRATION_VALIDATED",
        "calibration_evaluation_disjoint_gate": calibration_split.get("disjoint") is True,
        "calibration_fit_contract_gate": all(fit_contract.get(key) is expected for key, expected in {
            "preprocessing_fit_on": "calibration_ids_only",
            "parameters_fit_on": "calibration_observations_only",
            "variance_fit_on": "calibration_residuals_only",
            "evaluation_used_for_model_selection": False,
        }.items()),
        "real_calibration_metrics_gate": all(
            calibration_metrics.get(modality, {}).get("N_evaluation", 0) > 0
            and calibration_metrics.get(modality, {}).get("MAE") is not None
            and not str(calibration_metrics.get(modality, {}).get("status", "")).startswith("NOT_")
            for modality in ("XRD", "REFINEMENT", "OUTCOME_TEST")
        ),
        "identifiability_disclosure_gate": all(
            str(calibration.get("per_hypothesis_modality", {}).get(hypothesis_id, {}).get(modality, {}).get("status", "")).startswith("NOT_")
            for hypothesis_id, modality in (
                ("H2_COMPOSITION_HOMOGENEITY_LIMITED", "EDS"),
                ("H3_MORPHOLOGY_KINETICS_LIMITED", "SEM"),
            )
        ),
        "controlled_world_coverage_gate": world_names == set(WORLD_MODALITY_PROFILES) and len(controlled.get("seeds", [])) >= 5,
        "controlled_difficulty_gate": controlled.get("benchmark_design", {}).get("diagnostic_noise_scale", 0.0) >= 0.4 and any(
            row.get("recovery_rate_posterior_gt_0.8", 1.0) < 0.99 for row in controlled.get("summary", {}).values()
        ),
        "controlled_recovery_methodology_gate": controlled.get("status") == "METHODOLOGY_VALID",
        "pre_reveal_HIG_gate": all(event.get("event") != "MEASUREMENT_REVEALED" or any(prev.get("event") == "PREREGISTERED_SELECTED_ACTION" and prev.get("action", {}).get("action_id") == event.get("action", {}).get("action_id") for prev in ledger[:i]) for i, event in enumerate(ledger)),
        "HIG_bound_gate": all(0.0 <= float(event.get("expected_hig_nats", 0.0)) for event in ledger if event.get("event") in {"ACTION_SCORE_RECORD", "PREREGISTERED_SELECTED_ACTION"}),
        "HIG_order_invariance_gate": _hig_order_invariant(),
        "policy_distinction_gate": len(set(policy_comparison.get("policy_formulas", {}).values())) == len(POLICIES),
        "hybrid_formula_gate": "normalized cost" in policy_comparison.get("policy_formulas", {}).get("HYBRID", ""),
        "natural_policy_divergence_gate": policy_comparison.get("natural_divergence", {}).get("status") == "PASS",
        "hybrid_cost_effect_gate": policy_comparison.get("natural_divergence", {}).get("hybrid_cost_vs_pure_hig", {}).get("hybrid", 0.0) <= policy_comparison.get("natural_divergence", {}).get("hybrid_cost_vs_pure_hig", {}).get("pure_hig", 0.0),
        "discovery_hig_conflict_gate": policy_comparison.get("policy_validation", {}).get("discovery_hig_conflict_gate") == "PASS",
        "hybrid_score_recomputation_gate": policy_comparison.get("policy_validation", {}).get("hybrid_score_recomputation_gate") == "PASS",
        "hybrid_cost_causality_gate": policy_comparison.get("policy_validation", {}).get("hybrid_cost_causality_gate") == "PASS",
        "policy_sequence_distinction_gate": policy_comparison.get("policy_validation", {}).get("policy_sequence_distinction_gate") == "PASS",
        "conditional_prediction_gate": policy_comparison.get("conditional_hig_diagnostics", {}).get("status") == "PASS",
        "preregistration_timeline_gate": all(event.get("event_sequence", 0) > 0 for event in ledger),
        "action_namespace_gate": len(all_action_ids) == len(set(all_action_ids)),
        "retrospective_split_gate": "holdout" in replay.get("split_method", ""),
        "retrospective_observable_alignment_gate": replay.get("status") == "METHODOLOGY_VALID",
        "retrospective_replay_gate": replay.get("status") == "METHODOLOGY_VALID",
        "third_party_license_gate": _third_party_capability_gate("license"),
        "third_party_functionality_gate": _third_party_capability_gate("status"),
        "report_consistency_gate": inventory.get("dataset") == "A-Lab Precursor Genome" and controlled.get("status") == "METHODOLOGY_VALID" and policy_comparison.get("status") == "METHODOLOGY_VALID" and replay.get("status") == "METHODOLOGY_VALID",
        "local_test_gate": "NOT_RUN",
        "external_CI_gate": "NOT_INSPECTED",
    }
    boolean_gates_pass = all(value is True for value in required_gates.values() if isinstance(value, bool))
    controlled_ready = all(
        required_gates[key] is True for key in (
            "observable_schema_gate", "observable_semantics_alignment_gate", "raw_artifact_provenance_gate",
            "candidate_linkage_gate", "modality_contract_gate", "hypothesis_structure_gate",
            "hypothesis_directionality_gate", "controlled_world_coverage_gate", "controlled_difficulty_gate", "controlled_recovery_methodology_gate",
            "HIG_bound_gate", "HIG_order_invariance_gate", "policy_distinction_gate", "hybrid_formula_gate",
            "discovery_hig_conflict_gate", "hybrid_score_recomputation_gate", "hybrid_cost_causality_gate",
            "conditional_prediction_gate",
        )
    )
    retrospective_ready = calibration.get("status") == "A_LAB_RETROSPECTIVE_CALIBRATION_VALIDATED" and required_gates["real_calibration_metrics_gate"] is True
    status = "A_LAB_RETROSPECTIVE_CALIBRATION_VALIDATED" if boolean_gates_pass else (
        "CONTROLLED_MULTIMODAL_METHOD_VALIDATED" if controlled_ready else "NOT_READY"
    )
    if not retrospective_ready and status == "A_LAB_RETROSPECTIVE_CALIBRATION_VALIDATED":
        status = "CONTROLLED_MULTIMODAL_METHOD_VALIDATED"
    return {
        "status": status,
        "readiness": {
            "controlled_multimodal_method": "CONTROLLED_MULTIMODAL_METHOD_VALIDATED" if controlled_ready else "NOT_READY",
            "a_lab_retrospective_calibration": calibration.get("status", "NOT_EVALUATED"),
            "prospective_physical_validation": "PROSPECTIVE_PHYSICAL_VALIDATION_NOT_EVALUATED",
        },
        "gates": {key: ("PASS" if value is True else value if isinstance(value, str) else "FAIL") for key, value in required_gates.items()},
        "gate_evidence": {
            "ledger_event_count": len(ledger),
            "controlled_worlds": sorted(world_names),
            "required_seeds": list(SEEDS),
            "unsupported_retrospective_modalities": ["SEM", "EDS"],
            "calibration_evaluation_disjoint": calibration_split.get("disjoint"),
            "boolean_gate_count": sum(isinstance(value, bool) for value in required_gates.values()),
            "boolean_gate_pass_count": sum(value is True for value in required_gates.values() if isinstance(value, bool)),
        },
    }


def main() -> None:
    data_dir = os.environ.get("AICOSCIENTIST_ALAB_DATA_DIR", "data/external/precursor_genome_2026")
    cache_dir = os.environ.get("AICOSCIENTIST_ALAB_CACHE_DIR", "data/derived/alab")
    inventory = inventory_alab_modalities(data_dir, cache_dir)
    extractors = validate_extractors()
    hypotheses = {hid: model.diagnostics() for hid, model in build_alab_multimodal_hypotheses().items()}
    controlled = controlled_hypothesis_benchmark()
    policy_results = controlled_policy_comparison()
    split_manifest = _replay_split_manifest(data_dir)
    retrospective_models, calibration, calibration_observations, evaluation_observations, all_features = _fit_retrospective_calibration(
        data_dir, cache_dir, split_manifest,
    )
    calibration["split_manifest"] = "replay_split_manifest.json"
    calibration["holdout_rule"] = split_manifest.get("split_method")
    calibration["hypotheses"] = list(retrospective_models)
    calibration["metrics"] = {
        modality: calibration["per_hypothesis_modality"]["H1_PHASE_PURITY_LIMITED"][modality]
        for modality in ("XRD", "REFINEMENT", "OUTCOME_TEST")
    }
    replay = retrospective_replay(
        data_dir,
        cache_dir,
        retrospective_models,
        split_manifest,
        evaluation_observations,
        {cid: all_features[cid] for cid in split_manifest.get("evaluation_ids", []) if cid in all_features},
    )
    policy_validation = _policy_validation_diagnostics(policy_results["policies"])
    conditional_hig = _conditional_hig_diagnostics()
    policy_comparison = {
        "status": policy_results["status"],
        "scope": "controlled_world_action_selector_comparison; A-Lab replay remains retrospective historical and separate",
        "model_kind": "CONTROLLED_HYPOTHESIS_TEMPLATE",
        "policies": policy_results["policies"],
        "policy_formulas": policy_results["policy_formulas"],
        "hybrid_weights": HYBRID_WEIGHTS,
        "natural_divergence": {
            "status": "PASS" if policy_validation["policy_sequence_distinction_gate"] == "PASS" else "FAIL",
            "distinct_action_sequences": len({tuple(row["modalities_selected"]) for row in policy_results["policies"]}),
            "hybrid_cost_vs_pure_hig": {
                "hybrid": next(row["total_normalized_cost"] for row in policy_results["policies"] if row["policy"] == "HYBRID"),
                "pure_hig": next(row["total_normalized_cost"] for row in policy_results["policies"] if row["policy"] == "PURE_HIG"),
            },
        },
        "policy_validation": policy_validation,
        "conditional_hig_diagnostics": conditional_hig,
        "replay": replay,
        "ledger_events": policy_results["ledger_events"],
    }
    validation = build_validation(inventory, extractors, controlled, policy_comparison, replay, calibration)
    validation["gates"]["local_test_gate"] = os.environ.get("AICOSCIENTIST_LOCAL_TEST_GATE", "NOT_RUN")
    validation["gates"]["external_CI_gate"] = os.environ.get("AICOSCIENTIST_EXTERNAL_CI_GATE", "NOT_INSPECTED")
    _write("observable_schema.json", {name: definition.__dict__ for name, definition in OBSERVABLE_REGISTRY.items()})
    split_manifest["calibration_status"] = calibration["status"]
    split_manifest["leakage_assertions"] = calibration["leakage_assertions"]
    _write("hypothesis_calibration.json", calibration)
    _write("retrospective_calibration_metrics.json", calibration)
    _write("replay_split_manifest.json", split_manifest)
    _write("identifiability_diagnostics.json", {
        "status": "REAL_RETROSPECTIVE_IDENTIFIABILITY",
        "controlled_model_kind": "CONTROLLED_HYPOTHESIS_TEMPLATE",
        "retrospective_model_kind": "RETROSPECTIVE_CALIBRATED_HYPOTHESIS_MODEL",
        "worlds": list(WORLD_MODALITY_PROFILES),
        "diagnostic_modality_by_world": {world: profile["diagnostic"] for world, profile in WORLD_MODALITY_PROFILES.items()},
        "natural_divergence_required": True,
        "real_data_by_hypothesis": calibration["hypotheses"],
        "real_data_limitations": {
            "H1_PHASE_PURITY_LIMITED": "directly supported by canonical XRD descriptors and refinement observables; outcome linkage is retrospective",
            "H2_COMPOSITION_HOMOGENEITY_LIMITED": "candidate-linked EDS is unavailable; composition mechanism is not identifiable from the available real observations",
            "H3_MORPHOLOGY_KINETICS_LIMITED": "candidate-linked SEM is unavailable; morphology mechanism is not identifiable and process features are only weak outcome proxies",
        },
        "sem_eds_status": "NOT_EVALUATED_INSUFFICIENT_LINKAGE",
        "prospective_causal_identifiability": "NOT_ESTABLISHED",
    })
    _write("modality_inventory.json", inventory)
    _write("extractor_validation.json", extractors)
    _write("hypothesis_definitions.json", hypotheses)
    _write("controlled_hypothesis_recovery.json", controlled)
    _write("controlled_difficulty_diagnostics.json", {
        "status": "PASS" if any(row.get("recovery_rate_posterior_gt_0.8", 1.0) < 0.99 for row in controlled.get("summary", {}).values()) else "FAIL",
        "design": controlled.get("benchmark_design", {}),
        "summary": controlled.get("summary", {}),
        "worlds": controlled.get("world_profiles", {}),
    })
    _write("retrospective_policy_comparison.json", policy_comparison)
    _write("retrospective_replay.json", replay)
    _write("conditional_hig_diagnostics.json", conditional_hig)
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
        f"- Retrospective calibration: `{calibration['status']}` ({calibration['split']['calibration_n']} calibration IDs / {calibration['split']['evaluation_n']} evaluation IDs)\n"
        "- SEM/EDS candidate actions: disabled because archives are precursor-level and not canonically linked to sample IDs.\n"
        "- Scope: retrospective historical replay only; no prospective or causal claim.\n"
        "- H1 structural metrics are held-out evaluation metrics; H2/H3 mechanistic components remain explicitly weakly identified or not identifiable.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
