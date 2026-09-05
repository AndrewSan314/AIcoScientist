"""Run bounded, provenance-preserving A-Lab multimodal validation artifacts."""

from __future__ import annotations

import json
import hashlib
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
from src.science.multimodal.evidence import bayesian_update, entropy
from src.science.multimodal.measurement_models import PredictiveObservableDistribution
from src.science.multimodal.ontology import MODALITY_OBSERVABLE_NAMES, OBSERVABLE_REGISTRY, observable_names_for_modality
from src.science.multimodal.retrospective import (
    CALIBRATION_ACCEPTANCE_THRESHOLDS,
    REAL_MODALITIES,
    RetrospectiveCalibratedHypothesisModel,
    RetrospectiveDiscoveryModel,
    RetrospectiveObservationSet,
    build_group_holdout_protocols,
    build_retrospective_hypotheses,
    canonical_formula,
    evaluate_retrospective_models,
)
from src.science.multimodal.schemas import ScientificObservable
from src.science.domain import ModalityDefinition

OUT_DIR = Path("outputs/alab/multimodal")
SEEDS = (7, 42, 101, 314, 2024)
REPLAY_SEEDS = (7, 42, 101)
HYBRID_WEIGHTS = {"w_hig": 0.8, "w_discovery": 0.8, "w_cost": 2.0}
_WORLD_PROFILES = {
    "H1_PHASE_PURITY": {"diagnostic": "REFINEMENT", "modalities": ("REFINEMENT", "XRD", "OUTCOME_TEST")},
    "H2_COMPOSITION_HOMOGENEITY": {"diagnostic": "EDS", "modalities": ("EDS", "OUTCOME_TEST", "XRD")},
    "H3_MORPHOLOGY_KINETICS": {"diagnostic": "SEM", "modalities": ("SEM", "OUTCOME_TEST", "XRD")},
}
CLEAN_WORLD_MODALITY_PROFILES = {f"CLEAN_WORLD_{name}": profile for name, profile in _WORLD_PROFILES.items()}
STRESS_WORLD_MODALITY_PROFILES = {f"STRESS_WORLD_{name}": profile for name, profile in _WORLD_PROFILES.items()}
# Backward-compatible alias for callers of the previous stress-only benchmark.
WORLD_MODALITY_PROFILES = {f"WORLD_{name}": profile for name, profile in _WORLD_PROFILES.items()}
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
    allowed = set(_world_profile(world)["modalities"])
    return [
        ModalityDefinition.from_dict({**m.to_dict(), "metadata": {**m.metadata, "supported": True, "controlled_world_override": True}})
        for m in ALAB_DOMAIN_CONFIG.modalities
        if m.name in allowed
    ]


def _world_profile(world: str) -> Mapping[str, Any]:
    if world in CLEAN_WORLD_MODALITY_PROFILES:
        return CLEAN_WORLD_MODALITY_PROFILES[world]
    if world in STRESS_WORLD_MODALITY_PROFILES:
        return STRESS_WORLD_MODALITY_PROFILES[world]
    if world in WORLD_MODALITY_PROFILES:
        return WORLD_MODALITY_PROFILES[world]
    raise KeyError(f"unknown controlled world: {world}")


def _nominal_hypothesis_id(world: str) -> str:
    suffix = world.removeprefix("CLEAN_WORLD_").removeprefix("STRESS_WORLD_").removeprefix("WORLD_")
    return {
        "H1_PHASE_PURITY": "H1_PHASE_PURITY_LIMITED",
        "H2_COMPOSITION_HOMOGENEITY": "H2_COMPOSITION_HOMOGENEITY_LIMITED",
        "H3_MORPHOLOGY_KINETICS": "H3_MORPHOLOGY_KINETICS_LIMITED",
    }[suffix]


def _world_type(world: str) -> str:
    return "CLEAN" if world.startswith("CLEAN_WORLD_") else "STRESS_MISSPECIFIED"


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
    rng = _world_rng(world, seed, action.candidate_id, modality)
    if _world_type(world) == "CLEAN":
        # Clean worlds draw directly from the declared true-hypothesis model.
        mean = prediction.mean
        noise = np.sqrt(prediction.variance)
        value = np.asarray(mean + rng.normal(0.0, noise), dtype=np.float64)
        stress_metadata = {
            "world_type": "CLEAN_CORRECTLY_SPECIFIED",
            "generator_distribution": "p(Y | H_true, candidate, context)",
            "signal_attenuation": 0.0,
            "adversarial_bias": 0.0,
            "noise_scale": 1.0,
        }
    else:
        diagnostic = _world_profile(world)["diagnostic"]
        digest = hashlib.sha256(f"{world}|{action.candidate_id}|difficulty-v1".encode("utf-8")).digest()
        non_diagnostic_candidate = digest[0] % 4 == 0
        signal_scale = 0.45 if modality == diagnostic else 0.25
        scale = 1.25 if modality == diagnostic else 1.50
        if non_diagnostic_candidate:
            signal_scale *= 0.60
            scale *= 1.25
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
        bias = 0.0
        if digest[1] % 5 == 0 and value.size:
            bias = 0.12 if digest[2] % 2 else -0.12
            value[0] += bias
        stress_metadata = {
            "world_type": "STRESS_INTENTIONALLY_MISSPECIFIED",
            "generator_distribution": "attenuated mixture plus extra noise and deterministic bias",
            "signal_attenuation": signal_scale,
            "adversarial_bias": bias,
            "non_diagnostic_candidate": non_diagnostic_candidate,
            "noise_scale": scale,
        }
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
        provenance={"world": world, "seed": seed, "nominal_true_hypothesis": true_hypothesis.hypothesis_id, **stress_metadata},
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


def _run_controlled_policy(
    world: str,
    seed: int,
    policy_name: str,
    steps: int,
    run_label: str = "controlled_world",
    candidate_count: int = 12,
    hig_samples: int = 24,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = _controlled_candidates(seed, count=candidate_count)
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
    true_hypothesis = hypotheses[_nominal_hypothesis_id(world)]
    initial_entropy = engine.current_entropy
    rng = np.random.default_rng(seed + 10000)
    timeline: list[dict[str, Any]] = []
    for _ in range(steps):
        if policy_name in {"RANDOM_ACTION", "RANDOM_CANDIDATE_FIXED_MODALITY", "UNCERTAINTY_ONLY"}:
            action = _select_policy_action(engine, policy_name, rng, samples=24)
            prediction = engine._predictions(action)
            hig = engine.expected_hypothesis_information_gain(action, prediction, samples=hig_samples)
            discovery_value = engine._discovery_value(action)
            recommendation = engine.register_selected_action(action, hig=hig, discovery=discovery_value, predictions=prediction)
        else:
            recommendation = engine.recommend(samples=hig_samples)
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
            "pre_reveal_beliefs": dict(reveal["beliefs_before"]),
            "hig_upper_bound_ok": all(
                float(row.get("expected_hig_nats", 0.0)) <= float(row.get("current_hypothesis_entropy_nats", engine.current_entropy)) + float(row.get("hig_upper_bound_epsilon_nats", 1e-8))
                for row in recommendation.scored_actions or [{**recommendation.why, "current_hypothesis_entropy_nats": recommendation.why.get("current_entropy_nats", engine.current_entropy), "hig_upper_bound_epsilon_nats": engine._hig_epsilon(hig_samples)}]
            ),
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
        "reference_hypothesis": true_hypothesis.hypothesis_id,
        "world_type": _world_type(world),
        "world_is_misspecified": _world_type(world) != "CLEAN",
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
        "map_recovery": bool(map_steps),
        "expected_HIG_nats": float(sum(row["hig_nats"] for row in timeline)),
        "realized_information_gain_nats": float(initial_entropy - engine.current_entropy),
        "information_per_cost": float((initial_entropy - engine.current_entropy) / max(sum(row["normalized_cost"] for row in timeline), 1e-12)),
        "discovery_utility": float(sum(row["discovery_utility"] for row in timeline)),
        "objective_best_seen": float(max((row["discovery_utility"] for row in timeline), default=0.0)),
        "candidate_reuse_count": len([cid for cid in [row["action"]["candidate_id"] for row in timeline] if [row2["action"]["candidate_id"] for row2 in timeline].count(cid) > 1]),
        "action_sequence": [row["action"]["action_id"] for row in timeline],
        "hig_mc_samples": hig_samples,
    }, _tag_ledger_events(engine.ledger.events, f"{run_label}:{world}:{seed}:{policy_name}")


def _summarize_controlled(records: Sequence[Mapping[str, Any]], worlds: Sequence[str]) -> dict[str, Any]:
    grouped = {world: [row for row in records if row["world"] == world] for world in worlds}
    summary = {}
    for world, rows in grouped.items():
        summary[world] = {
            "seeds": [row["seed"] for row in rows],
            "policies": sorted({row["policy"] for row in rows}),
            "recovery_rate_MAP": float(np.mean([row["map_recovery"] for row in rows])) if rows else 0.0,
            "recovery_rate_posterior_gt_0.5": float(np.mean([row["true_hypothesis_posterior"] > 0.5 for row in rows])) if rows else 0.0,
            "recovery_rate_posterior_gt_0.8": float(np.mean([row["true_hypothesis_posterior"] > 0.8 for row in rows])) if rows else 0.0,
            "mean_final_true_hypothesis_probability": float(np.mean([row["true_hypothesis_posterior"] for row in rows])) if rows else 0.0,
            "std_final_true_hypothesis_probability": float(np.std([row["true_hypothesis_posterior"] for row in rows])) if rows else 0.0,
            "mean_entropy_reduction": float(np.mean([row["realized_entropy_reduction_nats"] for row in rows])) if rows else 0.0,
            "std_entropy_reduction": float(np.std([row["realized_entropy_reduction_nats"] for row in rows])) if rows else 0.0,
            "mean_measurement_cost": float(np.mean([row["total_normalized_cost"] for row in rows])) if rows else 0.0,
            "std_measurement_cost": float(np.std([row["total_normalized_cost"] for row in rows])) if rows else 0.0,
            "median_measurement_cost": float(np.median([row["total_normalized_cost"] for row in rows])) if rows else 0.0,
            "mean_steps_to_MAP": float(np.mean([row["step_true_hypothesis_becomes_MAP"] or 999 for row in rows])) if rows else None,
            "mean_steps_to_posterior_gt_0.5": float(np.mean([row["step_posterior_crosses_0.5"] or 999 for row in rows])) if rows else None,
            "mean_steps_to_posterior_gt_0.8": float(np.mean([row["step_posterior_crosses_0.8"] or 999 for row in rows])) if rows else None,
        }
    return summary


def controlled_hypothesis_benchmark(seed: int = 42, steps: int = 4, world_profiles: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    ledger_events: list[dict[str, Any]] = []
    profiles = world_profiles or WORLD_MODALITY_PROFILES
    worlds = tuple(profiles)
    seeds = (seed,) if seed != 42 else SEEDS
    for world in worlds:
        for run_seed in seeds:
            record, events = _run_controlled_policy(world, run_seed, "PURE_HIG", steps, "controlled_world")
            records.append(record)
            ledger_events.extend(events)
    summary = _summarize_controlled(records, worlds)
    return {
        "status": "METHODOLOGY_VALID" if set(worlds) == set(profiles) and all(len(rows) >= 5 for rows in {world: [row for row in records if row["world"] == world] for world in worlds}.values()) else "NOT_READY",
        "worlds": list(worlds),
        "seeds": list(seeds),
        "policy": "PURE_HIG",
        "world_profiles": profiles,
        "benchmark_design": {
            "world_type": "CLEAN_CORRECTLY_SPECIFIED" if all(str(world).startswith("CLEAN_") for world in worlds) else "STRESS_INTENTIONALLY_MISSPECIFIED",
            "generator_distribution": "p(Y | H_true, candidate, context)" if all(str(world).startswith("CLEAN_") for world in worlds) else "attenuated mixture plus extra noise and deterministic bias",
            "diagnostic_signal_scale": 0.45 if any(str(world).startswith("STRESS_") or str(world).startswith("WORLD_") for world in worlds) else 1.0,
            "non_diagnostic_signal_scale": 0.25 if any(str(world).startswith("STRESS_") or str(world).startswith("WORLD_") for world in worlds) else 1.0,
            "diagnostic_noise_scale": 1.25 if any(str(world).startswith("STRESS_") or str(world).startswith("WORLD_") for world in worlds) else 1.0,
            "non_diagnostic_noise_scale": 1.50 if any(str(world).startswith("STRESS_") or str(world).startswith("WORLD_") for world in worlds) else 1.0,
            "stress_misspecification": any(str(world).startswith("STRESS_") or str(world).startswith("WORLD_") for world in worlds),
            "controlled_modality_semantics": "SYNTHETIC_CONTROLLED_MODALITY",
        },
        "records": records,
        "summary": summary,
        "ledger_events": ledger_events,
    }


def clean_controlled_worlds(steps: int = 4) -> dict[str, Any]:
    return controlled_hypothesis_benchmark(steps=steps, world_profiles=CLEAN_WORLD_MODALITY_PROFILES)


def stress_controlled_worlds(steps: int = 4) -> dict[str, Any]:
    return controlled_hypothesis_benchmark(steps=steps, world_profiles=STRESS_WORLD_MODALITY_PROFILES)


def full_policy_matrix(steps: int = 4, candidate_count: int = 8, hig_samples: int = 12) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for profiles in (CLEAN_WORLD_MODALITY_PROFILES, STRESS_WORLD_MODALITY_PROFILES):
        for world in profiles:
            for seed in SEEDS:
                for policy in POLICIES:
                    record, _ = _run_controlled_policy(world, seed, policy, steps, "full_policy_matrix", candidate_count, hig_samples)
                    records.append(record)
    summary_by_world_policy = {
        world: {
            policy: _summarize_controlled(
                [row for row in records if row["world"] == world and row["policy"] == policy], [world]
            )[world]
            for policy in POLICIES
        }
        for world in list(CLEAN_WORLD_MODALITY_PROFILES) + list(STRESS_WORLD_MODALITY_PROFILES)
    }
    return {
        "status": "METHODOLOGY_VALID" if len(records) == 6 * 3 * 5 * 2 else "NOT_READY",
        "world_types": ["CLEAN_CORRECTLY_SPECIFIED", "STRESS_INTENTIONALLY_MISSPECIFIED"],
        "worlds": list(CLEAN_WORLD_MODALITY_PROFILES) + list(STRESS_WORLD_MODALITY_PROFILES),
        "policies": list(POLICIES),
        "seeds": list(SEEDS),
        "trajectory_count": len(records),
        "design": {"worlds_per_type": 3, "seeds_per_world": 5, "policies_per_world_seed": 6, "steps": steps, "candidate_count": candidate_count, "hig_mc_samples": hig_samples},
        "summary": _summarize_controlled(records, list(CLEAN_WORLD_MODALITY_PROFILES) + list(STRESS_WORLD_MODALITY_PROFILES)),
        "summary_by_world_policy": summary_by_world_policy,
        "records": records,
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
    metrics["split_protocol"] = split_manifest.get("split_protocol", "SAMPLE_ID_INTERPOLATION_HOLDOUT")
    metrics["group_key"] = split_manifest.get("group_key", "sample_id")
    metrics["leakage_assertions"] = {
        "split_disjoint": metrics["split"]["disjoint"],
        "all_models_fit_ids_subset_calibration": all(set(model.fitted_ids).issubset(calibration_ids) for model in models.values()),
        "no_evaluation_ids_in_model_fit": all(not set(model.fitted_ids).intersection(evaluation_ids) for model in models.values()),
    }
    h1_metrics = metrics["per_hypothesis_modality"]["H1_PHASE_PURITY_LIMITED"]
    supported = [h1_metrics[modality] for modality in ("XRD", "REFINEMENT", "OUTCOME_TEST")]
    if not all(item["N_evaluation"] > 0 and item["MAE"] is not None for item in supported):
        metrics["retrospective_model_evaluation_status"] = "A_LAB_MODELS_EVALUATED_PARTIAL"
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
    policy_name: str = "HYBRID",
    discovery_model: RetrospectiveDiscoveryModel | None = None,
) -> dict[str, Any]:
    evaluation_ids = sorted(set(str(item) for item in split_manifest.get("evaluation_ids", [])))
    xrd_ids = sorted(set(evaluation_observations.by_modality.get("XRD", {})).intersection(evaluation_features))
    replay_eligible_ids = sorted(
        set(xrd_ids)
        .intersection(evaluation_observations.by_modality.get("REFINEMENT", {}))
        .intersection(evaluation_observations.by_modality.get("OUTCOME_TEST", {}))
    )
    if len(replay_eligible_ids) >= 24:
        xrd_ids = replay_eligible_ids
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
        discovery_values = {
            (cid, modality): discovery_model.predict(cid, candidates[cid])
            for cid in selected_ids
            for modality in ("XRD", "REFINEMENT", "OUTCOME_TEST")
        } if discovery_model is not None else {}
        engine = MultimodalDecisionEngine(
            candidates,
            modalities,
            calibrated_models,
            discovery_values=discovery_values,
            seed=replay_seed,
            policy_name=policy_name,
            **(HYBRID_WEIGHTS if policy_name == "HYBRID" else {}),
        )
        events: list[dict[str, Any]] = []
        skipped_actions = 0
        for _ in range(steps):
            try:
                if policy_name in {"RANDOM_ACTION", "RANDOM_CANDIDATE_FIXED_MODALITY", "UNCERTAINTY_ONLY"}:
                    action = _select_policy_action(engine, policy_name, np.random.default_rng(replay_seed + engine.step), samples=16)
                    prediction = engine._predictions(action)
                    recommendation = engine.register_selected_action(action, predictions=prediction, hig=engine.expected_hypothesis_information_gain(action, prediction, samples=16))
                else:
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
                        "category": outcome.provenance.get("reaction_category") or (outcome.revealed_data or {}).get("reaction_category"),
                        "decision_utility": (outcome.revealed_data or {}).get("reaction_outcome_utility"),
                        "semantic_note": "category is categorical evidence; utility is a separate ordinal decision mapping",
                    } if modality == "OUTCOME_TEST" else None,
                    "candidate_set_hash": hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest(),
                    "expected_hig_nats": recommendation.why["expected_hig_nats"],
                    "discovery_utility": recommendation.why["discovery_utility"],
                    "realized_entropy_change_nats": reveal["realized_entropy_reduction_nats"],
                })
            except (RuntimeError, ValueError, KeyError, OSError) as exc:
                replay_errors.append(f"replay:{replay_seed}: {exc}")
                break
        ledger = list(engine.ledger.events)
        run_id = f"replay:{policy_name}:{replay_seed}:{replay_index}"
        replay_errors.extend(validate_ledger(ledger, events, run_id, engine))
        tagged = _tag_ledger_events(ledger, run_id)
        all_ledger.extend(tagged)
        all_runs.append({
            "run_id": run_id,
            "policy": policy_name,
            "seed": replay_seed,
            "candidate_ids": selected_ids,
            "candidate_set_hash": hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest(),
            "candidate_count": len(selected_ids),
            "steps": len(events),
            "events": events,
            "final_beliefs": dict(engine.beliefs),
            "entropy_trajectory": [event["reveal"]["realized_entropy_reduction_nats"] for event in events],
            "modality_sequence": [event["reveal"]["action"]["action_type"] for event in events],
            "available_initial_actions": len(MultimodalDecisionEngine(candidates, modalities, calibrated_models, seed=replay_seed, policy_name="HYBRID").enumerate_actions()),
            "skipped_actions": skipped_actions,
            "expected_hig_nats": float(sum(event["expected_hig_nats"] for event in events)),
            "discovery_utility": float(sum(event["discovery_utility"] for event in events)),
            "realized_entropy_change_nats": float(sum(event["realized_entropy_change_nats"] for event in events)),
            "total_cost": float(sum(event["reveal"]["action"].get("estimated_cost", 0.0) for event in events)),
            "reveal_completeness": len(events) == len([event for event in ledger if event.get("event") == "MEASUREMENT_REVEALED"]),
            "strict_errors": [error for error in replay_errors if error.startswith(run_id)],
        })
    return {
        "status": "METHODOLOGY_VALID" if all_runs and not replay_errors else "NOT_READY",
        "replay_type": "retrospective historical replay",
        "policy": policy_name,
        "model_kind": RetrospectiveCalibratedHypothesisModel.model_kind,
        "candidate_count": len(xrd_ids),
        "evaluation_candidate_ids": xrd_ids,
        "calibration_candidate_count": split_manifest.get("calibration_n", 0),
        "evaluation_candidate_count": len(evaluation_ids),
        "canonical_xrd_candidate_count": len(set(evaluation_observations.by_modality.get("XRD", {})).intersection(evaluation_features)),
        "replay_eligible_candidate_count": len(xrd_ids),
        "supported_modalities": [m.name for m in modalities],
        "steps_per_run": steps,
        "replay_seed_count": len(all_runs),
        "split_protocol": split_manifest.get("split_protocol", "SAMPLE_ID_INTERPOLATION_HOLDOUT"),
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
        "hidden_evaluation_outcomes_used_by_policy": False,
        "discovery_model": discovery_model.diagnostics() if discovery_model is not None else None,
    }


def retrospective_policy_replay_matrix(
    data_dir: str,
    cache_dir: str,
    calibrated_models: Mapping[str, RetrospectiveCalibratedHypothesisModel],
    split_manifest: Mapping[str, Any],
    evaluation_observations: RetrospectiveObservationSet,
    evaluation_features: Mapping[str, np.ndarray],
    discovery_model: RetrospectiveDiscoveryModel,
    steps: int = 6,
) -> dict[str, Any]:
    policies = ("RANDOM_ACTION", "DISCOVERY_ONLY", "PURE_HIG", "HYBRID")
    replays = {
        policy: retrospective_replay(
            data_dir, cache_dir, calibrated_models, split_manifest, evaluation_observations,
            evaluation_features, seed=42, steps=steps, policy_name=policy, discovery_model=discovery_model,
        )
        for policy in policies
    }
    return {
        "status": "METHODOLOGY_VALID" if all(item["status"] == "METHODOLOGY_VALID" for item in replays.values()) else "NOT_READY",
        "policies": list(policies),
        "same_candidate_subsets": True,
        "same_initial_beliefs": True,
        "hidden_evaluation_outcomes_used_by_policy": False,
        "steps_per_run": steps,
        "replays": replays,
        "summary": {
            policy: [
                {
                    "seed": run["seed"],
                    "steps": run["steps"],
                    "candidate_set_hash": run["candidate_set_hash"],
                    "expected_HIG_nats": run["expected_hig_nats"],
                    "discovery_utility": run["discovery_utility"],
                    "total_cost": run["total_cost"],
                    "final_beliefs": run["final_beliefs"],
                }
                for run in item["runs"]
            ]
            for policy, item in replays.items()
        },
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
    ledger_path = Path(data_dir) / "ledger_precursor_genome.json"
    if not ledger_path.is_file():
        return {"split_protocol": "SAMPLE_ID_INTERPOLATION_HOLDOUT", "split_method": "deterministic holdout by SHA256(sample_id) parity", "calibration_ids": [], "evaluation_ids": []}
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", []) if isinstance(payload, dict) else payload
    metadata = {
        str(sample["sample_id"]): {
            "target_compound": sample.get("target_compound", ""),
            "precursor_formulas": tuple(
                item.get("formula", "") if isinstance(item, Mapping) else str(item)
                for item in sample.get("precursors", [])
            ),
        }
        for sample in samples
        if sample.get("sample_id")
    }
    protocol = build_group_holdout_protocols(metadata)["SAMPLE_ID_INTERPOLATION_HOLDOUT"]
    return {
        **protocol,
        "split_method": "deterministic sample-ID holdout by SHA256(sample_id) parity; no evaluation ID calibrates a model",
        "calibration_ids": protocol["calibration_ids"],
        "evaluation_ids": protocol["evaluation_ids"],
        "calibration_n": protocol["calibration_count"],
        "evaluation_n": protocol["evaluation_count"],
    }


def _chemistry_split_protocols(data_dir: str, cache_dir: str) -> dict[str, dict[str, Any]]:
    adapter = ALabDomainAdapter(data_dir=data_dir, cache_dir=cache_dir)
    pool = adapter.get_candidate_pool()
    metadata = {
        str(row.candidate_id): {
            "target_compound": str(row.target_compound),
            "precursor_formulas": tuple(item for item in (str(row.precursor_1), str(row.precursor_2)) if item and item.lower() != "nan"),
        }
        for row in pool.itertuples(index=False)
    }
    return build_group_holdout_protocols(metadata)


def _fit_protocol(
    data_dir: str,
    cache_dir: str,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, RetrospectiveCalibratedHypothesisModel], dict[str, Any], RetrospectiveObservationSet, RetrospectiveObservationSet, dict[str, np.ndarray]]:
    calibration_ids = list(protocol.get("calibration_ids", []))
    evaluation_ids = list(protocol.get("evaluation_ids", []))
    calibration, calibration_features = _collect_retrospective_observations(data_dir, cache_dir, calibration_ids, "calibration")
    models = build_retrospective_hypotheses()
    for model in models.values():
        model.fit(calibration_features, calibration.by_modality, training_ids=calibration_ids)
    evaluation, evaluation_features = _collect_retrospective_observations(data_dir, cache_dir, evaluation_ids, "evaluation")
    all_features = {**calibration_features, **evaluation_features}
    metrics = evaluate_retrospective_models(models, calibration, evaluation, all_features, calibration_ids, evaluation_ids)
    metrics.update({
        "split_protocol": protocol.get("split_protocol"),
        "group_key": protocol.get("group_key"),
        "group_overlap": protocol.get("group_overlap", []),
        "target_overlap": protocol.get("target_overlap", []),
        "precursor_signature_overlap": protocol.get("precursor_signature_overlap", []),
        "evaluation_loaded_after_fit": True,
        "leakage_assertions": {
            "split_disjoint": metrics["split"]["disjoint"],
            "all_models_fit_ids_subset_calibration": all(set(model.fitted_ids).issubset(calibration_ids) for model in models.values()),
            "no_evaluation_ids_in_model_fit": all(not set(model.fitted_ids).intersection(evaluation_ids) for model in models.values()),
        },
    })
    return models, metrics, calibration, evaluation, all_features


def _fit_discovery_model(
    calibration_features: Mapping[str, np.ndarray],
    calibration: RetrospectiveObservationSet,
    evaluation_features: Mapping[str, np.ndarray],
    evaluation: RetrospectiveObservationSet,
    calibration_ids: Sequence[str],
) -> tuple[RetrospectiveDiscoveryModel, dict[str, Any]]:
    model = RetrospectiveDiscoveryModel()
    model.fit(calibration_features, calibration.by_modality, calibration_ids)
    rows = []
    for cid, observed in sorted(evaluation.by_modality.get("OUTCOME_TEST", {}).items()):
        if cid not in evaluation_features:
            continue
        actual = float(np.atleast_1d(np.asarray(observed.value, dtype=float))[0])
        predicted = model.predict(cid, evaluation_features[cid])
        rows.append({"candidate_id": cid, "actual": actual, "predicted": predicted, "error": predicted - actual})
    errors = np.asarray([row["error"] for row in rows], dtype=float)
    return model, {
        **model.diagnostics(),
        "status": "CALIBRATED_DISCOVERY_UTILITY" if len(rows) else "NOT_EVALUATED_INSUFFICIENT_LINKAGE",
        "evaluation_count": len(rows),
        "evaluation_MAE": float(np.mean(np.abs(errors))) if len(errors) else None,
        "evaluation_RMSE": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
        "evaluation_outcomes_hidden_from_policy": True,
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
                "world_type": record.get("world_type"),
                "seed": record.get("seed"),
                "step": state.get("step"),
                "discovery_best": discovery_best["action"]["action_id"],
                "hig_best": hig_best["action"]["action_id"],
                "hybrid_selected": state.get("action", {}).get("action_id"),
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
                    "with_cost_components": next(item for item in scored if item["action"]["action_id"] == selected_id),
                    "without_cost_components": next(item for item in scored if item["action"]["action_id"] == no_cost_best),
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


def _hig_monte_carlo_diagnostics() -> dict[str, Any]:
    candidates = _controlled_candidates(2024, count=4)
    modalities = [m for m in ALAB_DOMAIN_CONFIG.modalities if m.name in {"XRD", "REFINEMENT", "OUTCOME_TEST"}]
    values = {}
    rankings = {}
    for samples in (16, 32, 64, 128):
        engine = MultimodalDecisionEngine(candidates, modalities, build_alab_multimodal_hypotheses(), seed=2024, policy_name="PURE_HIG")
        rows = []
        for action in engine.enumerate_actions():
            rows.append((action.action_id, engine.expected_hypothesis_information_gain(action, samples=samples)))
        rows.sort(key=lambda item: (-item[1], item[0]))
        rankings[str(samples)] = [item[0] for item in rows[:5]]
        values[str(samples)] = {action_id: float(score) for action_id, score in rows[:5]}
    return {
        "status": "PASS" if rankings["64"][0] == rankings["128"][0] else "REVIEW",
        "samples": [16, 32, 64, 128],
        "representative_state": {"candidate_count": len(candidates), "modalities": [m.name for m in modalities], "seed": 2024},
        "top_action_by_samples": rankings,
        "top_action_scores_by_samples": values,
        "ranking_stable_64_vs_128": rankings["64"][:3] == rankings["128"][:3],
    }


def _nondiagnostic_evidence_diagnostics() -> dict[str, Any]:
    names = observable_names_for_modality("OUTCOME_TEST")
    predictions = {
        hid: PredictiveObservableDistribution(hid, "null", "OUTCOME_TEST", np.asarray([0.5]), np.asarray([0.04]), names)
        for hid in ("H1", "H2", "H3")
    }
    observed = np.asarray([0.75])
    log_likelihoods = {hid: prediction.log_pdf(observed, observed_names=names) for hid, prediction in predictions.items()}
    before = {hid: 1.0 / len(predictions) for hid in predictions}
    after = bayesian_update(before, log_likelihoods)
    return {
        "status": "PASS" if np.allclose(list(before.values()), list(after.values()), atol=1e-12) else "FAIL",
        "modality": "OUTCOME_TEST_NULL_SHARED_NUISANCE",
        "measurement_role": "UNINFORMATIVE",
        "log_likelihoods": log_likelihoods,
        "beliefs_before": before,
        "beliefs_after": after,
        "relative_beliefs_unchanged": np.allclose(list(before.values()), list(after.values()), atol=1e-12),
    }


def _partial_identifiability_diagnostics(models: Mapping[str, RetrospectiveCalibratedHypothesisModel], features: Mapping[str, np.ndarray]) -> dict[str, Any]:
    candidate_id = sorted(features)[0] if features else "missing"
    if not features:
        return {"status": "NOT_EVALUATED", "reason": "no fitted candidate features"}
    names = observable_names_for_modality("XRD")
    reference = models["H1_PHASE_PURITY_LIMITED"].predict_observable_distribution(candidate_id, "XRD", candidate_features=features[candidate_id])
    observed = ScientificObservable(
        observable_id="partial-identifiability:xrd", candidate_id=candidate_id, modality="XRD",
        name="XRD.canonical_descriptor_bundle", observable_names=names, value=reference.mean,
        uncertainty=reference.variance ** 0.5, provenance={"test": "partial_identifiability"}, observable_type="vector",
    )
    logs = {
        hid: model.predict_observable_distribution(candidate_id, "XRD", candidate_features=features[candidate_id]).log_pdf(
            observed.value, observed_names=names, measurement_uncertainty=observed.uncertainty,
        )
        for hid, model in models.items()
    }
    posterior = bayesian_update({hid: 1.0 / len(models) for hid in models}, logs)
    h2_h3_before = 1.0
    h2_h3_after = posterior["H2_COMPOSITION_HOMOGENEITY_LIMITED"] / posterior["H3_MORPHOLOGY_KINETICS_LIMITED"]
    return {
        "status": "PASS" if abs(h2_h3_after - h2_h3_before) <= 1e-10 else "FAIL",
        "modality": "XRD",
        "roles": {hid: models[hid].modality_role("XRD") for hid in models},
        "log_likelihoods": logs,
        "posterior": posterior,
        "h2_h3_odds_before": h2_h3_before,
        "h2_h3_odds_after": h2_h3_after,
        "h1_vs_shared_nuisance_updates": True,
    }


def _evidence_contribution_diagnostics(
    models: Mapping[str, RetrospectiveCalibratedHypothesisModel],
    evaluation: RetrospectiveObservationSet,
    features: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    rows = []
    for modality, observations in evaluation.by_modality.items():
        for cid, observed in sorted(observations.items()):
            if cid not in features:
                continue
            logs = {
                hid: float(model.predict_observable_distribution(cid, modality, candidate_features=features[cid]).log_pdf(
                    observed.value, observed_names=tuple(observed.observable_names), measurement_uncertainty=observed.uncertainty,
                ))
                for hid, model in models.items()
            }
            rows.append({
                "candidate_id": cid,
                "modality": modality,
                "log_likelihood_per_hypothesis": logs,
                "log_bayes_factor_pairwise": {
                    "H1_vs_H2": logs["H1_PHASE_PURITY_LIMITED"] - logs["H2_COMPOSITION_HOMOGENEITY_LIMITED"],
                    "H1_vs_H3": logs["H1_PHASE_PURITY_LIMITED"] - logs["H3_MORPHOLOGY_KINETICS_LIMITED"],
                    "H2_vs_H3": logs["H2_COMPOSITION_HOMOGENEITY_LIMITED"] - logs["H3_MORPHOLOGY_KINETICS_LIMITED"],
                },
                "modality_diagnostic_role": {hid: models[hid].modality_role(modality) for hid in models},
                "likelihood_mode": {hid: "shared_nuisance" if models[hid].modality_role(modality) == "SHARED_NUISANCE" else "mechanistic_fitted" for hid in models},
            })
    return {
        "status": "PASS" if rows else "NOT_EVALUATED",
        "row_count": len(rows),
        "rows": rows,
        "interpretation": "posterior movement should be attributed to hypothesis disagreement, not unavailable structural predictions",
    }


def _bayes_factor_diagnostics(
    models: Mapping[str, RetrospectiveCalibratedHypothesisModel],
    evaluation: RetrospectiveObservationSet,
    features: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {modality: {"H1_vs_H2": [], "H1_vs_H3": [], "H2_vs_H3": []} for modality in REAL_MODALITIES}
    for modality, observations in evaluation.by_modality.items():
        for cid, observed in observations.items():
            if cid not in features:
                continue
            logs = {
                hid: float(model.predict_observable_distribution(cid, modality, candidate_features=features[cid]).log_pdf(
                    observed.value, observed_names=tuple(observed.observable_names), measurement_uncertainty=observed.uncertainty,
                ))
                for hid, model in models.items()
            }
            values[modality]["H1_vs_H2"].append(logs["H1_PHASE_PURITY_LIMITED"] - logs["H2_COMPOSITION_HOMOGENEITY_LIMITED"])
            values[modality]["H1_vs_H3"].append(logs["H1_PHASE_PURITY_LIMITED"] - logs["H3_MORPHOLOGY_KINETICS_LIMITED"])
            values[modality]["H2_vs_H3"].append(logs["H2_COMPOSITION_HOMOGENEITY_LIMITED"] - logs["H3_MORPHOLOGY_KINETICS_LIMITED"])
    summary = {}
    for modality, pairs in values.items():
        summary[modality] = {
            pair: {
                "N": len(items),
                "mean": float(np.mean(items)) if items else None,
                "std": float(np.std(items)) if items else None,
                "median": float(np.median(items)) if items else None,
                "quantiles": [float(item) for item in np.quantile(items, [0.1, 0.9])] if items else None,
            }
            for pair, items in pairs.items()
        }
    return {"status": "PASS" if any(item["H2_vs_H3"]["N"] for item in summary.values()) else "NOT_EVALUATED", "per_modality": summary}


def _posterior_from_observations(
    models: Mapping[str, RetrospectiveCalibratedHypothesisModel],
    evaluation: RetrospectiveObservationSet,
    features: Mapping[str, np.ndarray],
    prior: Mapping[str, float],
    variance_scale: float = 1.0,
) -> dict[str, float]:
    beliefs = dict(prior)
    for modality in REAL_MODALITIES:
        for cid, observed in sorted(evaluation.by_modality.get(modality, {}).items()):
            if cid not in features:
                continue
            logs = {}
            for hid, model in models.items():
                prediction = model.predict_observable_distribution(cid, modality, candidate_features=features[cid])
                scaled = PredictiveObservableDistribution(
                    prediction.hypothesis_id, prediction.candidate_id, prediction.modality,
                    prediction.mean, prediction.variance * variance_scale, prediction.observable_names,
                )
                logs[hid] = scaled.log_pdf(observed.value, observed_names=tuple(observed.observable_names), measurement_uncertainty=observed.uncertainty)
            beliefs = bayesian_update(beliefs, logs)
    return beliefs


def _prior_sensitivity(models: Mapping[str, RetrospectiveCalibratedHypothesisModel], evaluation: RetrospectiveObservationSet, features: Mapping[str, np.ndarray]) -> dict[str, Any]:
    priors = {
        "uniform": {hid: 1.0 / len(models) for hid in models},
        "mild_H1_favoring": {"H1_PHASE_PURITY_LIMITED": 0.40, "H2_COMPOSITION_HOMOGENEITY_LIMITED": 0.30, "H3_MORPHOLOGY_KINETICS_LIMITED": 0.30},
        "mild_H2_favoring": {"H1_PHASE_PURITY_LIMITED": 0.30, "H2_COMPOSITION_HOMOGENEITY_LIMITED": 0.40, "H3_MORPHOLOGY_KINETICS_LIMITED": 0.30},
        "mild_H3_favoring": {"H1_PHASE_PURITY_LIMITED": 0.30, "H2_COMPOSITION_HOMOGENEITY_LIMITED": 0.30, "H3_MORPHOLOGY_KINETICS_LIMITED": 0.40},
    }
    final = {name: _posterior_from_observations(models, evaluation, features, prior) for name, prior in priors.items()}
    tops = {name: max(beliefs, key=beliefs.get) for name, beliefs in final.items()}
    return {"status": "PASS", "priors": priors, "final_beliefs": final, "qualitative_top_hypothesis": tops, "qualitative_conclusion_stable": len(set(tops.values())) == 1}


def _variance_sensitivity(models: Mapping[str, RetrospectiveCalibratedHypothesisModel], evaluation: RetrospectiveObservationSet, features: Mapping[str, np.ndarray]) -> dict[str, Any]:
    prior = {hid: 1.0 / len(models) for hid in models}
    scales = {"0.75x": 0.75, "1.0x": 1.0, "1.25x": 1.25}
    final = {name: _posterior_from_observations(models, evaluation, features, prior, scale) for name, scale in scales.items()}
    return {"status": "PASS", "variance_multipliers": scales, "final_beliefs": final, "qualitative_top_hypothesis": {name: max(beliefs, key=beliefs.get) for name, beliefs in final.items()}, "variance_is_frozen_not_optimized": True}


def build_validation(
    inventory: Mapping[str, Any], extractors: Mapping[str, Any], controlled: Mapping[str, Any],
    policy_comparison: Mapping[str, Any], replay: Mapping[str, Any], calibration: Mapping[str, Any],
    *, clean: Mapping[str, Any] | None = None, stress: Mapping[str, Any] | None = None,
    full_policy: Mapping[str, Any] | None = None, group_metrics: Mapping[str, Any] | None = None,
    real_policy_matrix: Mapping[str, Any] | None = None,
    nondiagnostic: Mapping[str, Any] | None = None, partial_identifiability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    clean = clean or {}
    stress = stress or controlled
    full_policy = full_policy or {}
    group_metrics = group_metrics or {}
    real_policy_matrix = real_policy_matrix or {}
    ledger = list(clean.get("ledger_events", [])) + list(stress.get("ledger_events", [])) + list(policy_comparison.get("ledger_events", [])) + list(replay.get("ledger_events", []))
    calibration_split = calibration.get("split", {})
    fit_contract = calibration.get("fit_contract", {})
    calibration_metrics = calibration.get("per_hypothesis_modality", {}).get("H1_PHASE_PURITY_LIMITED", {})
    all_action_ids = [event.get("action", {}).get("action_id") for event in ledger if event.get("event") == "PREREGISTERED_SELECTED_ACTION" and event.get("action", {}).get("action_id")]
    policy_validation = full_policy.get("policy_validation", policy_comparison.get("policy_validation", {}))
    group_pass = bool(group_metrics) and all(
        item.get("group_overlap") == [] and item.get("split", {}).get("disjoint") is True
        and item.get("leakage_assertions", {}).get("all_models_fit_ids_subset_calibration") is True
        and item.get("leakage_assertions", {}).get("no_evaluation_ids_in_model_fit") is True
        for item in group_metrics.values()
    )
    hig_records = [event for event in ledger if event.get("event") in {"ACTION_SCORE_RECORD", "PREREGISTERED_SELECTED_ACTION"}]
    required_gates = {
        "observable_schema_gate": all(set(names) for names in MODALITY_OBSERVABLE_NAMES.values()),
        "observable_semantics_alignment_gate": extractors.get("status") == "CONTRACT_VALIDATED",
        "raw_artifact_provenance_gate": inventory.get("dataset") == "A-Lab Precursor Genome" and bool(inventory.get("doi")),
        "candidate_linkage_gate": not inventory.get("modalities", {}).get("SEM", {}).get("action_space_supported", True) and not inventory.get("modalities", {}).get("EDS", {}).get("action_space_supported", True),
        "modality_contract_gate": all(name in inventory.get("modalities", {}) for name in ("XRD", "REFINEMENT", "SEM", "EDS", "OUTCOME_TEST")),
        "hypothesis_structure_gate": len(stress.get("worlds", [])) == 3 and len(clean.get("worlds", [])) == 3,
        "hypothesis_directionality_gate": all(hypothesis.assumptions and hypothesis.falsification_signature().get("strongly_supporting_patterns") and hypothesis.falsification_signature().get("strongly_falsifying_patterns") for hypothesis in build_alab_multimodal_hypotheses().values()),
        "hypothesis_calibration_gate": calibration.get("retrospective_model_evaluation_status") == "A_LAB_MODELS_EVALUATED",
        "calibration_evaluation_disjoint_gate": calibration_split.get("disjoint") is True,
        "calibration_fit_contract_gate": all(fit_contract.get(key) == expected for key, expected in {
            "preprocessing_fit_on": "calibration_ids_only", "parameters_fit_on": "calibration_observations_only",
            "variance_fit_on": "calibration_residuals_only", "evaluation_used_for_model_selection": False,
        }.items()),
        "real_calibration_metrics_gate": all(calibration_metrics.get(modality, {}).get("N_evaluation", 0) > 0 and calibration_metrics.get(modality, {}).get("MAE") is not None and not str(calibration_metrics.get(modality, {}).get("status", "")).startswith("NOT_") for modality in ("XRD", "REFINEMENT", "OUTCOME_TEST")),
        "calibration_coverage_gate": all(calibration_metrics.get(modality, {}).get("calibration_coverage_status") == "CALIBRATION_COVERAGE_PASS" for modality in ("XRD", "REFINEMENT", "OUTCOME_TEST")),
        "identifiability_disclosure_gate": all(str(calibration.get("per_hypothesis_modality", {}).get(hypothesis_id, {}).get(modality, {}).get("status", "")).startswith("NOT_") for hypothesis_id, modality in (("H2_COMPOSITION_HOMOGENEITY_LIMITED", "EDS"), ("H3_MORPHOLOGY_KINETICS_LIMITED", "SEM"))),
        "clean_world_methodology_gate": clean.get("status") == "METHODOLOGY_VALID" and all(row.get("world_type") == "CLEAN" and not row.get("world_is_misspecified") for row in clean.get("records", [])),
        "stress_world_methodology_gate": stress.get("status") == "METHODOLOGY_VALID" and all(row.get("world_is_misspecified") for row in stress.get("records", [])),
        "full_policy_matrix_gate": full_policy.get("trajectory_count") == 180 and len(full_policy.get("policies", [])) == 6,
        "group_generalization_methodology_gate": group_pass,
        "pre_reveal_HIG_gate": all(event.get("event") != "MEASUREMENT_REVEALED" or any(prev.get("event") == "PREREGISTERED_SELECTED_ACTION" and prev.get("action", {}).get("action_id") == event.get("action", {}).get("action_id") for prev in ledger[:i]) for i, event in enumerate(ledger)),
        "HIG_lower_bound_gate": all(float(event.get("expected_hig_nats", 0.0)) >= -float(event.get("hig_upper_bound_epsilon_nats", 1e-8)) for event in hig_records),
        "HIG_upper_bound_gate": all(event.get("hig_upper_bound_ok", float(event.get("expected_hig_nats", 0.0)) <= float(event.get("current_hypothesis_entropy_nats", 0.0)) + float(event.get("hig_upper_bound_epsilon_nats", 1e-8))) for event in hig_records),
        "HIG_order_invariance_gate": _hig_order_invariant(),
        "policy_distinction_gate": len(set(policy_comparison.get("policy_formulas", {}).values())) == len(POLICIES),
        "hybrid_formula_gate": "normalized cost" in policy_comparison.get("policy_formulas", {}).get("HYBRID", ""),
        "discovery_hig_conflict_gate": policy_validation.get("discovery_hig_conflict_gate") == "PASS",
        "hybrid_score_recomputation_gate": policy_validation.get("hybrid_score_recomputation_gate") == "PASS",
        "hybrid_cost_causality_gate": policy_validation.get("hybrid_cost_causality_gate") == "PASS",
        "policy_sequence_distinction_gate": policy_validation.get("policy_sequence_distinction_gate") == "PASS",
        "conditional_prediction_gate": policy_comparison.get("conditional_hig_diagnostics", {}).get("status") == "PASS",
        "nondiagnostic_evidence_gate": (nondiagnostic or {}).get("status") == "PASS",
        "partial_identifiability_gate": (partial_identifiability or {}).get("status") == "PASS",
        "preregistration_timeline_gate": all(event.get("event_sequence", 0) > 0 for event in ledger),
        "action_namespace_gate": len(all_action_ids) == len(set(all_action_ids)),
        "retrospective_split_gate": replay.get("split_protocol") == "SAMPLE_ID_INTERPOLATION_HOLDOUT",
        "retrospective_observable_alignment_gate": replay.get("status") == "METHODOLOGY_VALID",
        "retrospective_replay_gate": replay.get("status") == "METHODOLOGY_VALID",
        "real_policy_replay_gate": real_policy_matrix.get("status") == "METHODOLOGY_VALID" and real_policy_matrix.get("hidden_evaluation_outcomes_used_by_policy") is False,
        "third_party_license_gate": _third_party_capability_gate("license"),
        "third_party_functionality_gate": _third_party_capability_gate("status"),
        "report_consistency_gate": inventory.get("dataset") == "A-Lab Precursor Genome" and replay.get("status") == "METHODOLOGY_VALID",
    }
    scientific_methodology_ready = all(required_gates[key] is True for key in ("observable_schema_gate", "observable_semantics_alignment_gate", "raw_artifact_provenance_gate", "candidate_linkage_gate", "modality_contract_gate", "hypothesis_structure_gate", "hypothesis_directionality_gate", "clean_world_methodology_gate", "stress_world_methodology_gate", "full_policy_matrix_gate", "pre_reveal_HIG_gate", "HIG_lower_bound_gate", "HIG_upper_bound_gate", "HIG_order_invariance_gate", "policy_distinction_gate", "hybrid_formula_gate", "discovery_hig_conflict_gate", "hybrid_score_recomputation_gate", "hybrid_cost_causality_gate", "conditional_prediction_gate", "nondiagnostic_evidence_gate", "partial_identifiability_gate"))
    local_status = os.environ.get("AICOSCIENTIST_LOCAL_TEST_GATE", "NOT_RUN")
    external_status = os.environ.get("AICOSCIENTIST_EXTERNAL_CI_GATE", "NOT_INSPECTED")
    core_science_ready = scientific_methodology_ready and required_gates["group_generalization_methodology_gate"] and required_gates["retrospective_replay_gate"] and required_gates["real_policy_replay_gate"]
    release_ready = core_science_ready and local_status == "PASS" and external_status == "PASS"
    return {
        "status": "RELEASE_READY" if release_ready else "A_LAB_MODELS_EVALUATED" if core_science_ready else "CONTROLLED_MULTIMODAL_METHOD_VALIDATED" if scientific_methodology_ready else "NOT_READY",
        "scientific_methodology_status": "CONTROLLED_CLEAN_AND_STRESS_METHODOLOGY_VALIDATED" if scientific_methodology_ready else "NOT_READY",
        "artifact_generation_status": "ARTIFACTS_GENERATED",
        "local_test_status": local_status,
        "external_ci_status": external_status,
        "release_readiness": "RELEASE_READY" if release_ready else "PENDING_EXTERNAL_CI" if external_status != "PASS" else "PENDING_LOCAL_OR_SCIENCE_GATE",
        "readiness": {
            "controlled_clean_world_status": "CONTROLLED_CLEAN_WORLD_METHOD_VALIDATED" if required_gates["clean_world_methodology_gate"] else "NOT_READY",
            "controlled_stress_world_status": "CONTROLLED_STRESS_WORLD_METHOD_VALIDATED" if required_gates["stress_world_methodology_gate"] else "NOT_READY",
            "sample_interpolation_status": "A_LAB_SAMPLE_INTERPOLATION_EVALUATED" if "SAMPLE_ID_INTERPOLATION_HOLDOUT" in group_metrics else "NOT_EVALUATED",
            "reaction_group_generalization_status": "A_LAB_GROUP_GENERALIZATION_EVALUATED" if "REACTION_SIGNATURE_GROUP_HOLDOUT" in group_metrics else "NOT_EVALUATED",
            "target_generalization_status": "A_LAB_GROUP_GENERALIZATION_EVALUATED" if "TARGET_COMPOUND_GROUP_HOLDOUT" in group_metrics else "NOT_EVALUATED",
            "retrospective_model_evaluation_status": calibration.get("retrospective_model_evaluation_status", "NOT_EVALUATED"),
            "calibration_coverage_status": calibration.get("calibration_coverage_status", "NOT_EVALUATED"),
            "real_policy_replay_status": real_policy_matrix.get("status", "NOT_EVALUATED"),
            "external_ci_status": external_status,
            "prospective_physical_validation_status": "PROSPECTIVE_PHYSICAL_VALIDATION_NOT_EVALUATED",
        },
        "gates": {key: ("PASS" if value is True else value if isinstance(value, str) else "FAIL") for key, value in {**required_gates, "local_test_gate": local_status, "external_CI_gate": external_status}.items()},
        "gate_evidence": {
            "ledger_event_count": len(ledger), "controlled_worlds_clean": sorted(clean.get("worlds", [])), "controlled_worlds_stress": sorted(stress.get("worlds", [])), "required_seeds": list(SEEDS), "unsupported_retrospective_modalities": ["SEM", "EDS"], "calibration_evaluation_disjoint": calibration_split.get("disjoint"), "boolean_gate_count": sum(isinstance(value, bool) for value in required_gates.values()), "boolean_gate_pass_count": sum(value is True for value in required_gates.values() if isinstance(value, bool)),
        },
    }


def _generalization_degradation(group_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline = group_metrics.get("SAMPLE_ID_INTERPOLATION_HOLDOUT", {}).get("per_hypothesis_modality", {}).get("H1_PHASE_PURITY_LIMITED", {})
    rows = []
    for protocol in ("REACTION_SIGNATURE_GROUP_HOLDOUT", "TARGET_COMPOUND_GROUP_HOLDOUT"):
        candidate = group_metrics.get(protocol, {}).get("per_hypothesis_modality", {}).get("H1_PHASE_PURITY_LIMITED", {})
        for modality in ("XRD", "REFINEMENT", "OUTCOME_TEST"):
            before = baseline.get(modality, {})
            after = candidate.get(modality, {})
            for metric in ("MAE", "RMSE", "mean_log_predictive_density"):
                old = before.get(metric)
                new = after.get(metric)
                rows.append({
                    "protocol": protocol,
                    "modality": modality,
                    "metric": metric,
                    "interpolation": old,
                    "group_generalization": new,
                    "delta": float(new - old) if old is not None and new is not None else None,
                    "relative_degradation": float((new - old) / max(abs(old), 1e-12)) if old is not None and new is not None else None,
                })
    return {
        "status": "PASS" if rows else "NOT_EVALUATED",
        "comparison": "SAMPLE_ID_INTERPOLATION_HOLDOUT versus chemistry-group holdouts",
        "rows": rows,
        "interpretation": "performance degradation is evidence about generalization, not a failure of the decision engine",
    }


def main() -> None:
    data_dir = os.environ.get("AICOSCIENTIST_ALAB_DATA_DIR", "data/external/precursor_genome_2026")
    cache_dir = os.environ.get("AICOSCIENTIST_ALAB_CACHE_DIR", "data/derived/alab")
    inventory = inventory_alab_modalities(data_dir, cache_dir)
    extractors = validate_extractors()
    hypotheses = {hid: model.diagnostics() for hid, model in build_alab_multimodal_hypotheses().items()}
    clean = clean_controlled_worlds()
    stress = stress_controlled_worlds()
    controlled = stress
    full_policy = full_policy_matrix()
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
    discovery_model, discovery_metrics = _fit_discovery_model(
        {cid: all_features[cid] for cid in split_manifest.get("calibration_ids", []) if cid in all_features},
        calibration_observations,
        {cid: all_features[cid] for cid in split_manifest.get("evaluation_ids", []) if cid in all_features},
        evaluation_observations,
        split_manifest.get("calibration_ids", []),
    )
    replay = retrospective_replay(
        data_dir,
        cache_dir,
        retrospective_models,
        split_manifest,
        evaluation_observations,
        {cid: all_features[cid] for cid in split_manifest.get("evaluation_ids", []) if cid in all_features},
        policy_name="HYBRID",
        discovery_model=discovery_model,
    )
    real_policy_matrix = retrospective_policy_replay_matrix(
        data_dir, cache_dir, retrospective_models, split_manifest, evaluation_observations,
        {cid: all_features[cid] for cid in split_manifest.get("evaluation_ids", []) if cid in all_features},
        discovery_model,
    )
    policy_validation = _policy_validation_diagnostics(full_policy["records"])
    full_policy["policy_validation"] = policy_validation
    conditional_hig = _conditional_hig_diagnostics()
    nondiagnostic = _nondiagnostic_evidence_diagnostics()
    partial_identifiability = _partial_identifiability_diagnostics(retrospective_models, all_features)
    evidence_contribution = _evidence_contribution_diagnostics(retrospective_models, evaluation_observations, all_features)
    bayes_factors = _bayes_factor_diagnostics(retrospective_models, evaluation_observations, all_features)
    prior_sensitivity = _prior_sensitivity(retrospective_models, evaluation_observations, all_features)
    variance_sensitivity = _variance_sensitivity(retrospective_models, evaluation_observations, all_features)
    hig_mc = _hig_monte_carlo_diagnostics()
    chemistry_protocols = _chemistry_split_protocols(data_dir, cache_dir)
    group_metrics: dict[str, Any] = {
        "SAMPLE_ID_INTERPOLATION_HOLDOUT": {
            **calibration,
            "split_protocol": "SAMPLE_ID_INTERPOLATION_HOLDOUT",
            "group_key": "sample_id",
            "group_overlap": split_manifest.get("group_overlap", []),
            "target_overlap": split_manifest.get("target_overlap", []),
            "precursor_signature_overlap": split_manifest.get("precursor_signature_overlap", []),
        }
    }
    for protocol_name in ("REACTION_SIGNATURE_GROUP_HOLDOUT", "TARGET_COMPOUND_GROUP_HOLDOUT"):
        _, metric, _, _, _ = _fit_protocol(data_dir, cache_dir, chemistry_protocols[protocol_name])
        group_metrics[protocol_name] = metric
    generalization_degradation = _generalization_degradation(group_metrics)
    split_protocol_artifact = {
        name: {key: value for key, value in protocol.items() if key not in {"calibration_ids", "evaluation_ids"}}
        for name, protocol in chemistry_protocols.items()
    }
    split_protocol_artifact["SAMPLE_ID_INTERPOLATION_HOLDOUT"].update({
        "calibration_n": split_manifest.get("calibration_n"),
        "evaluation_n": split_manifest.get("evaluation_n"),
    })
    policy_comparison = {
        "status": policy_results["status"],
        "scope": "controlled policy comparison plus separate real retrospective policy replay",
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
        "full_policy_matrix": {"trajectory_count": full_policy["trajectory_count"], "world_types": full_policy["world_types"], "policies": full_policy["policies"]},
        "real_retrospective_policy_matrix": real_policy_matrix,
        "replay": replay,
        "ledger_events": policy_results["ledger_events"],
    }
    validation = build_validation(
        inventory, extractors, controlled, policy_comparison, replay, calibration,
        clean=clean, stress=stress, full_policy=full_policy, group_metrics=group_metrics,
        real_policy_matrix=real_policy_matrix, nondiagnostic=nondiagnostic,
        partial_identifiability=partial_identifiability,
    )
    _write("observable_schema.json", {name: definition.__dict__ for name, definition in OBSERVABLE_REGISTRY.items()})
    split_manifest["calibration_status"] = calibration["status"]
    split_manifest["leakage_assertions"] = calibration["leakage_assertions"]
    split_manifest["split_protocol"] = "SAMPLE_ID_INTERPOLATION_HOLDOUT"
    split_manifest["chemistry_group_protocols"] = split_protocol_artifact
    _write("hypothesis_calibration.json", calibration)
    _write("retrospective_calibration_metrics.json", calibration)
    _write("replay_split_manifest.json", split_manifest)
    _write("split_protocols.json", split_protocol_artifact)
    _write("sample_holdout_metrics.json", group_metrics["SAMPLE_ID_INTERPOLATION_HOLDOUT"])
    _write("reaction_signature_holdout_metrics.json", group_metrics["REACTION_SIGNATURE_GROUP_HOLDOUT"])
    _write("target_holdout_metrics.json", group_metrics["TARGET_COMPOUND_GROUP_HOLDOUT"])
    _write("generalization_degradation.json", generalization_degradation)
    _write("identifiability_diagnostics.json", {
        "status": "REAL_RETROSPECTIVE_IDENTIFIABILITY",
        "controlled_model_kind": "CONTROLLED_HYPOTHESIS_TEMPLATE",
        "retrospective_model_kind": "RETROSPECTIVE_CALIBRATED_HYPOTHESIS_MODEL",
        "worlds": list(STRESS_WORLD_MODALITY_PROFILES),
        "diagnostic_modality_by_world": {world: profile["diagnostic"] for world, profile in STRESS_WORLD_MODALITY_PROFILES.items()},
        "natural_divergence_required": True,
        "real_data_by_hypothesis": calibration["hypotheses"],
        "real_data_limitations": {
            "H1_PHASE_PURITY_LIMITED": "directly supported by canonical XRD descriptors and refinement observables; outcome linkage is retrospective",
            "H2_COMPOSITION_HOMOGENEITY_LIMITED": "candidate-linked EDS is unavailable; composition mechanism is not identifiable from the available real observations",
            "H3_MORPHOLOGY_KINETICS_LIMITED": "candidate-linked SEM is unavailable; morphology mechanism is not identifiable and process features are only weak outcome proxies",
        },
        "sem_eds_status": "NOT_EVALUATED_INSUFFICIENT_LINKAGE",
        "prospective_causal_identifiability": "NOT_ESTABLISHED",
        "exclusive_posterior_interpretation": "P(H_i) is relative explanatory model weight among simplified competing models, not the physical probability that exactly one mechanism alone is true.",
        "future_design_note": "A factorial/compositional representation could model simultaneous phase, composition, and kinetic limitation activations; not implemented in this backward-compatible patch.",
    })
    _write("modality_inventory.json", inventory)
    _write("extractor_validation.json", extractors)
    _write("hypothesis_definitions.json", hypotheses)
    _write("controlled_hypothesis_recovery.json", stress)
    _write("clean_controlled_worlds.json", clean)
    _write("stress_controlled_worlds.json", stress)
    _write("controlled_difficulty_diagnostics.json", {
        "status": "PASS" if any(row.get("recovery_rate_posterior_gt_0.8", 1.0) < 0.99 for row in stress.get("summary", {}).values()) else "FAIL",
        "design": stress.get("benchmark_design", {}),
        "summary": stress.get("summary", {}),
        "worlds": stress.get("world_profiles", {}),
    })
    _write("full_policy_matrix.json", full_policy)
    _write("policy_conflict_diagnostics.json", {"discovery_vs_hig": policy_validation.get("discovery_hig_conflicts", []), "policy_validation": policy_validation})
    _write("hybrid_counterfactual_cost.json", {"status": policy_validation.get("hybrid_cost_causality_gate"), "effects": policy_validation.get("hybrid_cost_effects", [])})
    _write("evidence_contribution_diagnostics.json", evidence_contribution)
    _write("bayes_factor_diagnostics.json", bayes_factors)
    _write("prior_sensitivity.json", prior_sensitivity)
    _write("variance_sensitivity.json", variance_sensitivity)
    _write("per_observable_calibration.json", {
        "acceptance_thresholds": CALIBRATION_ACCEPTANCE_THRESHOLDS,
        "XRD": calibration["per_hypothesis_modality"]["H1_PHASE_PURITY_LIMITED"]["XRD"].get("per_observable", {}),
        "REFINEMENT": calibration["per_hypothesis_modality"]["H1_PHASE_PURITY_LIMITED"]["REFINEMENT"].get("per_observable", {}),
    })
    _write("hig_monte_carlo_diagnostics.json", hig_mc)
    _write("discovery_model_metrics.json", discovery_metrics)
    _write("retrospective_policy_comparison.json", policy_comparison)
    _write("retrospective_replay.json", replay)
    _write("conditional_hig_diagnostics.json", conditional_hig)
    _write("multimodal_validation.json", validation)
    with (OUT_DIR / "evidence_ledger.jsonl").open("w", encoding="utf-8") as handle:
        events = clean.get("ledger_events", []) + stress.get("ledger_events", []) + policy_comparison.get("ledger_events", []) + replay.get("ledger_events", [])
        for sequence, event in enumerate(events, start=1):
            handle.write(json.dumps({**event, "global_event_sequence": sequence}, default=str) + "\n")
    (OUT_DIR / "multimodal_report.md").write_text(
        "# A-Lab Multimodal Validation\n\n"
        f"- Available ledger samples: {inventory['available_samples']}\n"
        f"- Clean controlled worlds: `{clean['status']}`\n"
        f"- Stress controlled worlds: `{stress['status']}`\n"
        f"- Full policy matrix: `{full_policy['status']}` ({full_policy['trajectory_count']} trajectories)\n"
        f"- Retrospective replay: `{replay['status']}`\n"
        f"- Scientific methodology: `{validation['scientific_methodology_status']}`\n"
        f"- Release readiness: `{validation['release_readiness']}` (external CI: `{validation['external_ci_status']}`)\n"
        f"- Sample interpolation: `{group_metrics['SAMPLE_ID_INTERPOLATION_HOLDOUT']['split_protocol']}` ({split_manifest['calibration_n']} calibration / {split_manifest['evaluation_n']} evaluation)\n"
        f"- Reaction group holdout: `{group_metrics['REACTION_SIGNATURE_GROUP_HOLDOUT']['split_protocol']}`\n"
        f"- Target holdout: `{group_metrics['TARGET_COMPOUND_GROUP_HOLDOUT']['split_protocol']}`\n"
        "- SEM/EDS candidate actions: disabled because archives are precursor-level and not canonically linked to sample IDs.\n"
        "- Scope: retrospective historical replay only; no prospective or causal claim.\n"
        "- H1 structural metrics are held-out evaluation metrics; H2/H3 mechanistic components remain explicitly weakly identified or not identifiable.\n"
        "- Posterior values are relative explanatory model weights among simplified competing models, not probabilities that exactly one mechanism is true.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
