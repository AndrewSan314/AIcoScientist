#!/usr/bin/env python3
"""build_snapshot.py

Aggregates authentic project artifacts from outputs/ and data/ into a versioned,
deterministic presentation dataset for the AIcoScientist Mission Control.
Fail-closed validation, cryptographic provenance, real A-Lab sample extraction,
and source score preservation.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "presentation" / "data"
FRONTEND_DIR = ROOT / "presentation" / "frontend"
DEST_FILE = DATA_DIR / "snapshot.json"
MANIFEST_FILE = DATA_DIR / "snapshot_manifest.json"
REGISTRY_FILE = DATA_DIR / "dataset_registry.json"
SOURCE_PROVENANCE_FILE = DATA_DIR / "scientific_source_provenance.json"

SCIENTIFIC_SOURCE_COMMIT = None
SCIENTIFIC_SOURCE_COMMIT_STATUS = "UNVERIFIED"
SNAPSHOT_SCHEMA_VERSION = "1.3.0"


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_info() -> tuple[str, str]:
    """Retrieve current Git HEAD commit and branch."""
    values = []
    for args in (("git", "rev-parse", "HEAD"), ("git", "rev-parse", "--abbrev-ref", "HEAD")):
        result = subprocess.run(
            list(args),
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"Unable to resolve Git provenance: {' '.join(args)}")
        values.append(result.stdout.strip())
    return values[0], values[1]


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_source_provenance_manifest(
    source_artifacts: dict[str, Path],
    source_hashes: dict[str, str],
    generated_at_utc: str,
    snapshot_generator_commit: str | None = None,
    presentation_build_commit: str | None = None,
) -> dict[str, Any]:
    """Pin artifact identity without claiming an unverified scientific commit."""
    return {
        "schema_version": "1.0.0",
        "generated_at_utc": generated_at_utc,
        "snapshot_generator_commit": snapshot_generator_commit,
        "presentation_build_commit": presentation_build_commit,
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "scientific_source_commit_status": SCIENTIFIC_SOURCE_COMMIT_STATUS,
        "verification_method": (
            "SHA-256 hashes are verified against the local source artifacts. "
            "No required artifact embeds an independently verifiable scientific source SHA; "
            "the external-source audit records no commit SHA."
        ),
        "artifacts": [
            {
                "artifact_name": name,
                "artifact_path": path.relative_to(ROOT).as_posix(),
                "artifact_sha256": source_hashes[name],
                "generating_scientific_commit": None,
                "generating_snapshot_commit": snapshot_generator_commit,
                "presentation_build_commit": presentation_build_commit,
                "provenance_verification_method": "local_artifact_sha256_only",
            }
            for name, path in source_artifacts.items()
        ],
    }


def _parse_run_id(run_id: str) -> dict[str, Any]:
    parts = run_id.split(":")
    if parts[0] == "replay" and len(parts) == 4:
        return {"policy": parts[1], "seed": int(parts[2]), "mode": "HISTORICAL_REPLAY"}
    if parts[0] in {"controlled_world", "policy_comparison"} and len(parts) == 4:
        return {"world": parts[1], "seed": int(parts[2]), "policy": parts[3], "mode": "CONTROLLED_SYNTHETIC"}
    raise ValueError(f"Unsupported campaign run id: {run_id}")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _valid_beliefs(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(_finite(prob) and 0 <= float(prob) <= 1 for prob in value.values()) and abs(sum(float(prob) for prob in value.values()) - 1) <= 1e-4


def _canonical_refinement_case(scan: dict[str, Any]) -> tuple[dict[str, Any] | None, int | None, str | None]:
    cases = scan.get("refinement_cases") or []
    if not cases:
        return None, None, None
    active = scan.get("active_case_index")
    if isinstance(active, int) and 0 <= active < len(cases):
        return cases[active], active, "ledger_active_case_index"

    def rank(item: tuple[int, dict[str, Any]]) -> tuple[int, float, float, int]:
        index, case = item
        verification = case.get("verification") or {}
        is_manual = case.get("rank") == -1 or case.get("origin") == "manual"
        quality = verification.get("human_quality_score")
        quality_rank = float(quality) if _finite(quality) else float("inf")
        rwp = float(case["rwp"]) if _finite(case.get("rwp")) else float("inf")
        return (0 if is_manual else 1, quality_rank, rwp, index)

    index, selected = min(enumerate(cases), key=rank)
    method = "upstream_fallback_manual" if selected.get("rank") == -1 or selected.get("origin") == "manual" else "upstream_fallback_quality_or_rwp"
    return selected, index, method


def _canonical_scan(sample: dict[str, Any]) -> tuple[dict[str, Any] | None, int | None, str | None, bool]:
    scans = sample.get("characterization", {}).get("xrd", {}).get("scans", [])
    if not scans:
        return None, None, None, False
    active = sample.get("active_scan_index")
    if isinstance(active, int) and 0 <= active < len(scans):
        return scans[active], active, "ledger_active_scan_index", True

    candidates = []
    for index, scan in enumerate(scans):
        case, case_index, _ = _canonical_refinement_case(scan)
        if case is None:
            continue
        verification = case.get("verification") or {}
        quality = verification.get("human_quality_score")
        rwp = float(case["rwp"]) if _finite(case.get("rwp")) else float("inf")
        candidates.append((0 if case.get("rank") == -1 or case.get("origin") == "manual" else 1, float(quality) if _finite(quality) else float("inf"), rwp, index, scan, case_index))
    if candidates:
        _, _, _, index, scan, _ = min(candidates, key=lambda item: item[:4])
        return scan, index, "upstream_recomputed_active_scan", True
    valid = next((index for index, scan in enumerate(scans) if scan.get("is_active") or scan.get("status") == "valid"), 0)
    return scans[valid], valid, "noncanonical_replay_fallback", False


def _refinement_case_record(sample_id: str, scan: dict[str, Any], scan_index: int, case_index: int, case: dict[str, Any]) -> dict[str, Any]:
    scan_id = str(scan.get("filename") or f"{sample_id}:scan:{scan_index}")
    return {
        "scan_id": scan_id,
        "case_id": f"{scan_id}:case:{case_index}",
        "rwp": float(case["rwp"]) if _finite(case.get("rwp")) else None,
        "phases": case.get("phase_weights") if isinstance(case.get("phase_weights"), dict) else None,
        "source_index": case_index,
        "scan_index": scan_index,
    }


def _build_recorded_campaign_run(run_id: str, ledger_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve one complete ledger run without inventing score components or features."""
    events = [e for e in ledger_events if e.get("run_id") == run_id]
    if not events:
        raise ValueError(f"Campaign run {run_id} not found in evidence ledger")
    metadata = _parse_run_id(run_id)
    by_step: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"scores": [], "preregistration": None, "observation": None, "belief_update": None}
    )
    seen_event_keys: set[tuple[int, str]] = set()
    for event in events:
        step = event.get("step")
        if not isinstance(step, int):
            raise ValueError(f"Run {run_id} contains an event without an integer step")
        if event.get("run_id") != run_id or not isinstance(event.get("event_sequence"), int) or not isinstance(event.get("timestamp"), str) or not event.get("timestamp"):
            raise ValueError(f"Run {run_id} contains an event with invalid identity or timestamp")
        event_type = event.get("event")
        action = event.get("action")
        if event_type in {"ACTION_SCORE_RECORD", "PREREGISTERED_SELECTED_ACTION", "MEASUREMENT_REVEALED", "BELIEF_UPDATE"}:
            if not isinstance(action, dict) or not action.get("action_id") or not action.get("candidate_id") or not action.get("action_type"):
                raise ValueError(f"Run {run_id} step {step} has an invalid action record")
            if action.get("requested_at_step") not in {None, step}:
                raise ValueError(f"Run {run_id} step {step} action requested_at_step is inconsistent")
        if event_type == "ACTION_SCORE_RECORD":
            by_step[step]["scores"].append(event)
        elif event_type == "PREREGISTERED_SELECTED_ACTION":
            if (step, event_type) in seen_event_keys:
                raise ValueError(f"Run {run_id} step {step} contains duplicate preregistration")
            by_step[step]["preregistration"] = event
        elif event_type == "MEASUREMENT_REVEALED":
            if (step, event_type) in seen_event_keys:
                raise ValueError(f"Run {run_id} step {step} contains duplicate observation")
            by_step[step]["observation"] = event
        elif event_type == "BELIEF_UPDATE":
            if (step, event_type) in seen_event_keys:
                raise ValueError(f"Run {run_id} step {step} contains duplicate belief update")
            by_step[step]["belief_update"] = event
        seen_event_keys.add((step, event_type))

    if sorted(by_step) != list(range(1, len(by_step) + 1)):
        raise ValueError(f"Run {run_id} has non-contiguous step numbers")

    steps: list[dict[str, Any]] = []
    tested_candidates: list[str] = []
    candidate_modalities: dict[str, set[str]] = defaultdict(set)
    previous_beliefs: dict[str, float] | None = None
    for step_number in sorted(by_step):
        step_data = by_step[step_number]
        prereg = step_data["preregistration"]
        observation = step_data["observation"]
        belief_update = step_data["belief_update"]
        scores = step_data["scores"]
        if not prereg or not observation or not belief_update or not scores:
            raise ValueError(f"Run {run_id} step {step_number} is incomplete")
        sequences = [prereg.get("event_sequence"), observation.get("event_sequence"), belief_update.get("event_sequence")]
        if not all(isinstance(value, int) for value in sequences) or not (sequences[0] < sequences[1] < sequences[2]):
            raise ValueError(f"Run {run_id} step {step_number} violates preregistration/reveal/update ordering")
        p_action = prereg.get("action", {})
        o_action = observation.get("action", {})
        u_action = belief_update.get("action", {})
        action_key = (p_action.get("candidate_id"), p_action.get("action_type"))
        if action_key != (o_action.get("candidate_id"), o_action.get("action_type")) or action_key != (u_action.get("candidate_id"), u_action.get("action_type")):
            raise ValueError(f"Run {run_id} step {step_number} selected action does not match observation")
        prior = prereg.get("beliefs_before")
        after = belief_update.get("beliefs_after")
        if not _valid_beliefs(prior) or not _valid_beliefs(after) or prior.keys() != after.keys():
            raise ValueError(f"Run {run_id} step {step_number} has invalid beliefs")
        if previous_beliefs is not None and any(abs(float(prior[key]) - previous_beliefs.get(key, 0)) > 1e-4 for key in prior):
            raise ValueError(f"Run {run_id} step {step_number} breaks belief continuity")
        if belief_update.get("beliefs_before") != prior or observation.get("beliefs_before") not in (None, prior) or observation.get("beliefs_after") not in (None, after):
            raise ValueError(f"Run {run_id} step {step_number} has inconsistent belief records")
        previous_beliefs = {key: float(value) for key, value in after.items()}
        score_actions = {score.get("action", {}).get("action_id") for score in scores}
        if p_action.get("action_id") not in score_actions:
            raise ValueError(f"Run {run_id} step {step_number} selected action is absent from scored actions")
        distributions = prereg.get("predictive_distributions") or {}
        if not isinstance(distributions, dict) or not distributions:
            raise ValueError(f"Run {run_id} step {step_number} has no predictive distributions")
        for dist in distributions.values():
            if (dist.get("candidate_id"), dist.get("modality")) != action_key:
                raise ValueError(f"Run {run_id} step {step_number} predictive distribution is bound to another action")
            kind = dist.get("distribution_kind", "gaussian")
            if kind == "gaussian":
                means, variances = dist.get("mean"), dist.get("variance")
                if not isinstance(means, list) or not isinstance(variances, list) or len(means) != len(variances) or not means or not all(_finite(value) for value in means) or not all(_finite(value) and float(value) > 0 for value in variances):
                    raise ValueError(f"Run {run_id} step {step_number} has invalid predictive moments")
            elif kind == "categorical":
                probs = dist.get("probabilities")
                if not isinstance(probs, list) or not probs or not all(_finite(value) and 0 <= float(value) <= 1 for value in probs) or abs(sum(float(value) for value in probs) - 1) > 1e-4:
                    raise ValueError(f"Run {run_id} step {step_number} has invalid predictive probabilities")
        for score in scores:
            action = score.get("action", {})
            candidate_id = action.get("candidate_id")
            modality = action.get("action_type")
            if candidate_id and modality:
                candidate_modalities[candidate_id].add(modality)
            if any(not _finite(score.get(field)) for field in ["expected_hig_nats", "discovery_utility", "normalized_cost", "total_action_score"]):
                raise ValueError(f"Run {run_id} step {step_number} has a non-finite source score")
            if not _finite(action.get("estimated_cost")):
                raise ValueError(f"Run {run_id} step {step_number} has a non-finite action cost")
            if score.get("predictive_distributions") or score.get("predictive_distribution"):
                raise ValueError(f"Run {run_id} step {step_number} stores predictive distributions on an alternative score")
        selected_candidate = p_action.get("candidate_id")
        tested_before = list(tested_candidates)
        if selected_candidate and selected_candidate not in tested_candidates:
            tested_candidates.append(selected_candidate)
        sorted_scores = sorted(scores, key=lambda item: float(item["total_action_score"]), reverse=True)
        steps.append({
            "step": step_number,
            "preregistration": prereg,
            "observation": observation,
            "belief_update": belief_update,
            "all_scored_actions": scores,
            "top_actions": sorted_scores[:12],
            "total_actions_evaluated": len(scores),
            "tested_candidates_before": tested_before,
        })

    result = {
        "run_id": run_id,
        **metadata,
        "initial_beliefs": steps[0]["preregistration"].get("beliefs_before", {}),
        "steps": steps,
    }
    if metadata["mode"] == "CONTROLLED_SYNTHETIC":
        result["candidates"] = [
            {
                "candidate_id": candidate_id,
                "composition_label": candidate_id,
                "target_system": "Controlled-Synthetic Benchmark",
                "available_modalities": sorted(candidate_modalities[candidate_id]),
            }
            for candidate_id in sorted(candidate_modalities)
        ]
    else:
        result["replay_candidate_ids"] = sorted(tested_candidates)
    return result


def build_recorded_campaign_runs(ledger_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run_ids = sorted({str(event.get("run_id")) for event in ledger_events if event.get("run_id")})
    complete_run_ids = []
    for run_id in run_ids:
        steps = {event.get("step") for event in ledger_events if event.get("run_id") == run_id}
        if all(any(event.get("run_id") == run_id and event.get("step") == step and event.get("event") == "ACTION_SCORE_RECORD" for event in ledger_events) for step in steps):
            complete_run_ids.append(run_id)
    return [_build_recorded_campaign_run(run_id, ledger_events) for run_id in complete_run_ids]


def build_flagship_campaign(ledger_events: list[dict[str, Any]]) -> dict[str, Any]:
    run_id = "policy_comparison:WORLD_H1_PHASE_PURITY:42:HYBRID"
    return _build_recorded_campaign_run(run_id, ledger_events)


def build_alab_replay_campaign(ledger_events: list[dict[str, Any]]) -> dict[str, Any]:
    return _build_recorded_campaign_run("replay:HYBRID:42:1", ledger_events)


def extract_real_alab_sample_catalog() -> list[dict[str, Any]]:
    """Extract authentic A-Lab sample records from data/external/precursor_genome_2026/ledger_precursor_genome.json.
    
    Zero fabricated records, zero fabricated XRD or refinement observables.
    Every record traces directly to the peer-reviewed dataset (DOI: 10.5281/zenodo.21285546, CC BY 4.0).
    """
    ledger_path = ROOT / "data" / "external" / "precursor_genome_2026" / "ledger_precursor_genome.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"A-Lab source ledger missing at {ledger_path}")

    with ledger_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    samples_raw = raw.get("samples", [])
    if not samples_raw:
        raise ValueError("No samples found in ledger_precursor_genome.json!")

    catalog = []
    for s in samples_raw:
        sid = str(s["sample_id"])
        outcome = s.get("outcome") or {}
        scans = s.get("characterization", {}).get("xrd", {}).get("scans", [])

        selected_scan, selected_scan_index, scan_rule, is_canonical_scan = _canonical_scan(s)
        refinement_cases = []
        for scan_index, scan in enumerate(scans):
            for case_index, case in enumerate(scan.get("refinement_cases") or []):
                refinement_cases.append(_refinement_case_record(sid, scan, scan_index, case_index, case))
        selected_case = None
        selected_case_index = None
        case_rule = None
        if selected_scan is not None and selected_scan_index is not None:
            selected_case, selected_case_index, case_rule = _canonical_refinement_case(selected_scan)
        selected_case_record = (
            _refinement_case_record(sid, selected_scan, selected_scan_index, selected_case_index, selected_case)
            if selected_scan is not None and selected_scan_index is not None and selected_case is not None and selected_case_index is not None
            else None
        )
        rwp = selected_case_record.get("rwp") if selected_case_record else None

        cat = outcome.get("reaction_category")
        target_formula = s.get("target_compound")
        if not target_formula:
            raise ValueError(f"A-Lab source record {sid} has no target_compound")

        # Extract real precursors
        precs = []
        for p in s.get("precursors", []):
            if isinstance(p, dict):
                f_name = p.get("formula") or p.get("name")
                if f_name:
                    precs.append(str(f_name))
            elif p:
                precs.append(str(p))

        # Extract real refinement phase distribution
        phases = []
        for p in outcome.get("phases") or []:
            if isinstance(p, dict):
                phases.append({
                    "name": p.get("name"),
                    "weight_percent": p.get("weight_percent"),
                })

        synth = s.get("synthesis") if isinstance(s.get("synthesis"), dict) else {}
        heating_temp = synth.get("heating_temperature")
        heating_time = synth.get("heating_time")

        item = {
            "sample_id": sid,
            "target_formula": target_formula,
            "target_stoichiometry": s.get("target_stoichiometry"),
            "precursors": precs,
            "heating_temperature_c": float(heating_temp) if heating_temp is not None else None,
            "heating_time_minutes": float(heating_time) if heating_time is not None else None,
            "reaction_energy_ev_per_atom": float(s["reaction_energy_ev_per_atom"]) if s.get("reaction_energy_ev_per_atom") is not None else None,
            "reaction_category": cat,
            "xrd_available": bool(scans),
            "refinement_available": bool(refinement_cases),
            "outcome_available": bool(cat),
            "sem_available": False,
            "eds_available": False,
            "sem_availability_reason": "Archive present in sem.zip (408 MB) but lacks sample-level linkage (precursor-level only).",
            "eds_availability_reason": "Archive present in eds.zip (358 KB) but lacks sample-level linkage (precursor-level only).",
            "refinement_rwp": rwp,
            "refinement_phases": phases,
            "refinement_phases_preview": phases[:4],
            "refinement_cases": refinement_cases,
            "selected_refinement_case_id": selected_case_record.get("case_id") if selected_case_record else None,
            "refinement_selection_rule": (
                "source_canonical_active_scan_and_case" if scan_rule == "ledger_active_scan_index" and case_rule == "ledger_active_case_index"
                else scan_rule or case_rule or "no_refinement_case"
            ),
            "source_archive": "data/external/precursor_genome_2026/ledger_precursor_genome.json",
            "source_record_identifier": sid,
            "extractor_name": "ALabSourceLedgerExtractor.direct_v1",
            "extractor_version": "1.2.0",
        }
        catalog.append(item)

    return catalog
 

def build_dataset_registry(
    campaign_runs: list[dict[str, Any]],
    hypotheses: dict[str, Any],
    samples: list[dict[str, Any]],
    modality_inventory: dict[str, Any],
    electrolyte_screening: dict[str, Any],
    electrolyte_simulation: dict[str, Any],
    electrolyte_target_audit: dict[str, Any],
) -> dict[str, Any]:
    """Build registry fields from the source artifacts used by the snapshot."""
    controlled_runs = [run for run in campaign_runs if run.get("mode") == "CONTROLLED_SYNTHETIC"]
    replay_runs = [run for run in campaign_runs if run.get("mode") == "HISTORICAL_REPLAY"]
    controlled_ids = sorted({
        candidate["candidate_id"]
        for run in controlled_runs
        for candidate in run.get("candidates", [])
        if candidate.get("candidate_id")
    })
    controlled_modalities: dict[str, set[float]] = defaultdict(set)
    for run in controlled_runs:
        for step in run.get("steps", []):
            for record in step.get("all_scored_actions", []):
                action = record.get("action", {})
                modality = action.get("action_type")
                cost = action.get("estimated_cost")
                if modality and isinstance(cost, (int, float)):
                    controlled_modalities[modality].add(float(cost))

    def controlled_modality(name: str, diagnostic: bool, units: str) -> dict[str, Any]:
        costs = controlled_modalities.get(name, set())
        return {
            "cost": next(iter(costs)) if len(costs) == 1 else None,
            "diagnostic": diagnostic,
            "units": units,
            "available": bool(costs),
        }

    alab_modality_entries = {}
    for name, source in modality_inventory.get("modalities", {}).items():
        alab_modality_entries[name] = {
            "available": bool(source.get("action_space_supported")),
            "linkedCandidateCount": source.get("linked_candidate_samples"),
            "coverage": source.get("derived_observable_coverage"),
            "linkageQuality": source.get("candidate_sample_linkage_quality"),
            "missingness": source.get("missingness"),
            "source": source.get("source"),
        }
    target = electrolyte_target_audit.get("target_semantics", {})
    screening_trial = electrolyte_screening.get("working_set_trials", {}).get("200", {})
    simulation_policies = electrolyte_simulation.get("simulation_policies", {})
    simulation_seeds = electrolyte_simulation.get("evaluated_seeds", [])
    controlled_default = next(
        (run for run in controlled_runs if run.get("run_id") == "policy_comparison:WORLD_H1_PHASE_PURITY:42:HYBRID"),
        None,
    )
    replay_default = next((run for run in replay_runs if run.get("run_id") == "replay:HYBRID:42:1"), None)
    electrolyte_candidate_count = electrolyte_simulation.get("actual_search_space_size")
    if controlled_default is None or replay_default is None:
        raise ValueError("Required default controlled or replay run is missing from source campaign runs")
    if not isinstance(electrolyte_candidate_count, int) or electrolyte_candidate_count <= 0:
        raise ValueError("Electrolyte source artifact has no valid actual_search_space_size")
    if not isinstance(simulation_seeds, list) or not simulation_seeds:
        raise ValueError("Electrolyte source artifact has no evaluated seeds")
    surrogate_default_policy = "HYBRID_DEFAULT"
    if surrogate_default_policy not in simulation_policies:
        raise ValueError("Electrolyte source artifact has no HYBRID_DEFAULT policy summary")
    hypothesis_ids = sorted(hypotheses)
    replay_ids = sorted({candidate_id for run in replay_runs for candidate_id in run.get("replay_candidate_ids", [])})
    def run_configuration(run: dict[str, Any]) -> dict[str, Any]:
        candidate_ids = run.get("replay_candidate_ids") or [candidate.get("candidate_id") for candidate in run.get("candidates", []) if candidate.get("candidate_id")]
        modalities = sorted({
            action.get("action", {}).get("action_type")
            for step in run.get("steps", [])
            for action in step.get("all_scored_actions", [])
            if action.get("action", {}).get("action_type")
        })
        config = {
            "configurationId": run["run_id"],
            "runId": run["run_id"],
            "seed": run.get("seed"),
            "policy": run["policy"],
            "mode": run["mode"],
            "stepCount": len(run.get("steps", [])),
            "candidateIds": sorted(set(candidate_ids)),
            "modalities": modalities,
        }
        if run.get("world") is not None:
            config["world"] = run["world"]
        return config

    controlled_configurations = [run_configuration(run) for run in controlled_runs]
    replay_configurations = [run_configuration(run) for run in replay_runs]
    detailed_runs = electrolyte_simulation.get("detailed_policy_seed_runs", {})
    surrogate_configurations = [
        {
            "configurationId": f"{policy}::{run['seed']}",
            "policy": policy,
            "seed": run["seed"],
            "mode": "SIMULATED_SURROGATE",
            "stepCount": len(run.get("queried_candidate_ids", [])),
            "candidateIds": list(run.get("queried_candidate_ids", [])),
            "modalities": ["SURROGATE_ORACLE"],
        }
        for policy, policy_runs in sorted(detailed_runs.items())
        for run in sorted(policy_runs, key=lambda item: item.get("seed", 0))
    ]

    return {
        "registry_schema_version": "1.1.0",
        "datasets": [
            {
                "id": "controlled_multimodal_alloy",
                "legacy_id": "controlled_synthesis",
                "displayName": "Controlled Multimodal Alloy Benchmark",
                "domain": "In-Silico Controlled Solid-State Worlds",
                "provenance": {
                    "sourceType": "IN_SILICO_BENCHMARK",
                    "sourcePaths": [
                        "src/science/multimodal/",
                        "outputs/alab/multimodal/clean_controlled_worlds.json",
                        "outputs/alab/multimodal/full_policy_matrix.json",
                    ],
                    "citation": "AIcoScientist In-Silico Multimodal Verification Benchmark (180 trajectories across 6 worlds × 5 seeds × 6 policies)",
                    "doi": None,
                    "license": "MIT",
                },
                "candidateCount": len(controlled_ids),
                "candidateIds": controlled_ids,
                "availableConfigurations": controlled_configurations,
                "modalities": {
                    "XRD": controlled_modality("XRD", True, "intensity (a.u.)"),
                    "REFINEMENT": controlled_modality("REFINEMENT", True, "phase fraction [0, 1]"),
                    "OUTCOME_TEST": controlled_modality("OUTCOME_TEST", False, "synthesis success binary"),
                },
                "hypotheses": hypothesis_ids,
                "defaultConfiguration": {
                    "runId": controlled_default["run_id"],
                    "world": controlled_default["world"],
                    "seed": controlled_default["seed"],
                    "policy": controlled_default["policy"],
                },
                "capabilities": {
                    "modelHypothesesAvailable": True,
                    "posteriorModelWeightsAvailable": True,
                    "mutuallyExclusivePhysicalMechanismsClaimed": True,
                    "prospectiveMechanismIdentification": False,
                    "candidateScreening": False,
                    "preregistrationReplay": True,
                    "closedLoopExecution": True,
                    "surrogateSimulation": False,
                    "evidenceKind": "CONTROLLED_SYNTHETIC",
                },
                "summary": f"Recorded in-silico policy trajectories over {len(controlled_ids)} source candidate IDs and {len(hypothesis_ids)} source hypotheses; no physical synthesis is implied.",
                "statusBadge": "Controlled Benchmark",
                "disclosures": [
                    "Synthetic benchmark designed for formal Bayesian inference and policy comparison; source action scores are displayed as recorded."
                ],
            },
            {
                "id": "alab_precursor_genome",
                "legacy_id": "alab_replay",
                "displayName": "A-Lab Precursor Genome Retrospective Replay",
                "domain": "Autonomous Solid-State Inorganic Synthesis",
                "provenance": {
                    "sourceType": "PEER_REVIEWED_BENCHMARK",
                    "sourcePaths": [
                        "data/external/precursor_genome_2026/ledger_precursor_genome.json",
                        "outputs/alab/multimodal/evidence_ledger.jsonl",
                    ],
                    "citation": "A-Lab Precursor Genome Dataset (Zenodo DOI: 10.5281/zenodo.21285546, CC BY 4.0)",
                    "doi": "10.5281/zenodo.21285546",
                    "license": "CC BY 4.0",
                },
                "candidateCount": len(samples),
                "candidateIds": [sample["sample_id"] for sample in samples],
                "featuredCandidateIds": replay_ids,
                "availableConfigurations": replay_configurations,
                "modalities": alab_modality_entries,
                "hypotheses": hypothesis_ids,
                "defaultConfiguration": {
                    "runId": replay_default["run_id"],
                    "seed": replay_default["seed"],
                    "policy": replay_default["policy"],
                },
                "capabilities": {
                    "modelHypothesesAvailable": True,
                    "posteriorModelWeightsAvailable": True,
                    "mutuallyExclusivePhysicalMechanismsClaimed": False,
                    "prospectiveMechanismIdentification": False,
                    "candidateScreening": False,
                    "preregistrationReplay": True,
                    "closedLoopExecution": False,
                    "surrogateSimulation": False,
                    "evidenceKind": "HISTORICAL_REPLAY",
                },
                "summary": f"Retrospective replay over {len(samples)} source samples; the selected replay contains {len(replay_ids)} featured sample IDs and is distinct from the original laboratory execution log.",
                "statusBadge": "Historical Replay",
                "disclosures": [
                    "Replay policy and seed are recorded computational settings, not claims about the original laboratory policy.",
                    "SEM and EDS archives are present but not linked to candidate sample IDs, so they are unavailable to replay.",
                ],
            },
            {
                "id": "anode_free_electrolyte_screening",
                "legacy_id": "electrolyte_search",
                "displayName": "Anode-Free Electrolyte Screening & Surrogate Optimization",
                "domain": "High-Entropy LiFSI Liquid Battery Electrolytes",
                "provenance": {
                    "sourceType": "PEER_REVIEWED_EXPERIMENTAL_&_SURROGATE",
                    "sourcePaths": [
                        "data/external/al_anode_free_2025/",
                        "outputs/electrolyte/benchmark/screening_quality_diagnostics.json",
                        "outputs/electrolyte/benchmark/surrogate_simulation.json",
                    ],
                    "citation": "AmanchukwuLab, Nature Communications 2025 (DOI: 10.1038/s41467-025-63303-7)",
                    "doi": "10.1038/s41467-025-63303-7",
                    "license": "CC BY 4.0",
                },
                "candidateCount": electrolyte_candidate_count,
                "screenedWorkingSetCount": electrolyte_simulation.get("screened_working_set_size"),
                "targetObservable": target.get("raw_target_column"),
                "scientificTargetName": target.get("scientific_target_name"),
                "targetObservableDescription": target.get("scientific_meaning"),
                "modalities": {
                    "SURROGATE_ORACLE": {"diagnostic": False, "units": f"{target.get('scientific_target_name', 'source target')} [0, 1]", "available": True},
                    "SCREENING_FILTER": {"diagnostic": True, "units": "rank ensemble score", "available": True},
                },
                "hypotheses": ["SURROGATE_CAPACITY_OPTIMIZATION"],
                "defaultConfiguration": {
                    "policy": surrogate_default_policy,
                    "seed": min(simulation_seeds),
                    "queries": simulation_policies[surrogate_default_policy].get("queried_count"),
                },
                "availableConfigurations": surrogate_configurations,
                "capabilities": {
                    "modelHypothesesAvailable": False,
                    "posteriorModelWeightsAvailable": False,
                    "mutuallyExclusivePhysicalMechanismsClaimed": False,
                    "prospectiveMechanismIdentification": False,
                    "candidateScreening": True,
                    "preregistrationReplay": False,
                    "closedLoopExecution": False,
                    "surrogateSimulation": True,
                    "evidenceKind": "SIMULATED_SURROGATE",
                },
                "summary": f"Source diagnostic reports a {electrolyte_simulation.get('requested_search_space_size')} candidate virtual pool, a {electrolyte_simulation.get('screened_working_set_size')} candidate working set, and ensemble surrogate trajectories.",
                "statusBadge": "Screening & Surrogate",
                "disclosures": [
                    "Surrogate oracle is an in-silico computational approximation only; no live physical battery cycling is performed.",
                    electrolyte_simulation.get("MODEL_COUPLING_LIMITATION", "Model coupling limitation is unavailable in the source artifact."),
                    f"Screening timing is stage-specific: the 200-candidate diagnostic reports {screening_trial.get('screening_time_sec')} seconds; the end-to-end simulation artifact reports {electrolyte_simulation.get('screening_time_sec')} seconds.",
                    f"Target audit: {target.get('scientific_meaning', 'target semantics unavailable')}.",
                ],
            },
        ],
    }


def main() -> None:
    print("==========================================================")
    print("  AIcoScientist Discovery Mission Control - Snapshot Build")
    print("==========================================================")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    head_commit, current_branch = get_git_info()
    now_utc = datetime.now(timezone.utc).isoformat()

    # Define all required source artifacts to track and hash
    source_artifacts = {
        "evidence_ledger": OUTPUTS_DIR / "alab" / "multimodal" / "evidence_ledger.jsonl",
        "precursor_genome_ledger": ROOT / "data" / "external" / "precursor_genome_2026" / "ledger_precursor_genome.json",
        "multimodal_validation": OUTPUTS_DIR / "alab" / "multimodal" / "multimodal_validation.json",
        "hypothesis_calibration": OUTPUTS_DIR / "alab" / "multimodal" / "per_observable_calibration.json",
        "full_policy_matrix": OUTPUTS_DIR / "alab" / "multimodal" / "full_policy_matrix.json",
        "hig_sensitivity": OUTPUTS_DIR / "alab" / "multimodal" / "hig_trajectory_sensitivity.json",
        "hypothesis_definitions": OUTPUTS_DIR / "alab" / "multimodal" / "hypothesis_definitions.json",
        "modality_inventory": OUTPUTS_DIR / "alab" / "multimodal" / "modality_inventory.json",
        "alab_dataset_audit": OUTPUTS_DIR / "alab" / "alab_dataset_audit.json",
        "electrolyte_screening": OUTPUTS_DIR / "electrolyte" / "benchmark" / "screening_quality_diagnostics.json",
        "electrolyte_simulation": OUTPUTS_DIR / "electrolyte" / "benchmark" / "surrogate_simulation.json",
        "electrolyte_target_audit": OUTPUTS_DIR / "electrolyte" / "audit" / "experimental_identity_audit.json",
    }

    # Verify all source artifacts exist and calculate hashes
    source_hashes = {}
    for name, path in source_artifacts.items():
        if not path.exists():
            raise FileNotFoundError(f"CRITICAL: Missing source artifact '{name}' at {path}")
        source_hashes[name] = compute_sha256(path)
        print(f"Verified artifact: {name} (SHA-256: {source_hashes[name][:12]}...)")
    source_provenance = build_source_provenance_manifest(source_artifacts, source_hashes, now_utc, head_commit, head_commit)

    # 1. Validation & Readiness
    validation = load_json(source_artifacts["multimodal_validation"])
    gate_evidence = validation.get("gate_evidence", {})
    pass_count = gate_evidence.get("boolean_gate_pass_count")
    total_count = gate_evidence.get("boolean_gate_count")
    if not isinstance(pass_count, int) or not isinstance(total_count, int):
        raise ValueError("Validation artifact is missing boolean gate counts")
    print(f"Validation gates: {pass_count}/{total_count} boolean gates passed")

    # 2. Hypotheses
    hypotheses = load_json(source_artifacts["hypothesis_definitions"])

    # 3. Calibration
    calibration = load_json(source_artifacts["hypothesis_calibration"])

    # 4. Evidence Ledger
    ledger_events = load_jsonl(source_artifacts["evidence_ledger"])
    print(f"Evidence ledger: {len(ledger_events)} immutable audit events")

    # 5. Flagship & Replay Campaigns (with strict fail-closed validation)
    campaign_runs = build_recorded_campaign_runs(ledger_events)
    flagship = next(run for run in campaign_runs if run["run_id"] == "policy_comparison:WORLD_H1_PHASE_PURITY:42:HYBRID")
    replay = next(run for run in campaign_runs if run["run_id"] == "replay:HYBRID:42:1")
    print(f"Flagship campaign verified: {len(flagship['steps'])} steps, all event sequences valid")
    print(f"Replay campaign verified: {len(replay['steps'])} steps, {len(replay['replay_candidate_ids'])} candidates")

    # 6. Policy Benchmark Matrix
    full_matrix = load_json(source_artifacts["full_policy_matrix"])
    benchmark_data = {
        "status": full_matrix.get("status"),
        "trajectory_count": full_matrix.get("trajectory_count", 180),
        "world_types": full_matrix.get("world_types", []),
        "worlds": full_matrix.get("worlds", []),
        "policies": full_matrix.get("policies", []),
        "seeds": full_matrix.get("seeds", []),
        "design": full_matrix.get("design", {}),
        "summary": full_matrix.get("summary", {}),
        "summary_by_world_policy": full_matrix.get("summary_by_world_policy", {}),
        "policy_validation": full_matrix.get("policy_validation", {}),
    }

    # 7. HIG Sensitivity
    hig_sens = load_json(source_artifacts["hig_sensitivity"])
    sensitivity_data = {
        "status": hig_sens.get("status"),
        "design": hig_sens.get("design", {}),
        "trajectory_count": hig_sens.get("trajectory_count", 60),
        "aggregate_by_world_policy": hig_sens.get("aggregate_by_world_policy", {}),
    }

    # 8. Electrolyte Screening & Simulation
    elec_screen = load_json(source_artifacts["electrolyte_screening"])
    elec_sim = load_json(source_artifacts["electrolyte_simulation"])
    elec_target_audit = load_json(source_artifacts["electrolyte_target_audit"])

    # 9. A-Lab Dataset Audit & Modalities
    alab_audit = load_json(source_artifacts["alab_dataset_audit"])
    modality_inv = load_json(source_artifacts["modality_inventory"])

    # 10. Real A-Lab Sample Catalog (1035 genuine records)
    samples = extract_real_alab_sample_catalog()
    print(f"Real A-Lab sample catalog: {len(samples)} authentic samples extracted")

    # 11. Core Scientific Abstractions & Domain Contracts
    architecture = {
        "core_abstractions": [
            {
                "name": "MaterialDomainAdapter",
                "role": "Decouples core falsification & hypothesis selection from candidate schema & lab equipment.",
                "file": "src/science/domain.py",
            },
            {
                "name": "ModalityDefinition",
                "role": "Formalizes characterization vs outcome actions, duration, cost, and prerequisite dependencies.",
                "file": "src/science/domain.py",
            },
            {
                "name": "ScientificAction",
                "role": "Candidate ID × Modality action tuple requested at specific step.",
                "file": "src/science/actions.py",
            },
            {
                "name": "MultimodalScientificHypothesis",
                "role": "Predicts observable distributions under competing physical assumptions.",
                "file": "src/science/multimodal/hypotheses.py",
            },
            {
                "name": "PredictiveObservableDistribution",
                "role": "Strict variance contract containing total observation variance without double-counting measurement noise.",
                "file": "src/science/multimodal/measurement_models.py",
            },
            {
                "name": "MultimodalDecisionEngine",
                "role": "Joint Candidate × Modality recommendation while preserving source-recorded action scores.",
                "file": "src/science/multimodal/decision.py",
            },
            {
                "name": "MultimodalEvidenceLedger",
                "role": "Immutable, chronological event stream guaranteeing preregistration occurs strictly prior to observation reveal.",
                "file": "src/science/multimodal/evidence.py",
            },
        ],
        "domains": [
            {
                "domain_id": "controlled_synthetic",
                "name": "Controlled Multi-Hypothesis Synthetic Worlds",
                "status": "VALIDATED_METHODOLOGY",
                "candidates_count": len({candidate["candidate_id"] for run in campaign_runs if run.get("mode") == "CONTROLLED_SYNTHETIC" for candidate in run.get("candidates", [])}),
                "modalities": ["XRD", "REFINEMENT", "SEM", "EDS", "OUTCOME_TEST"],
                "purpose": "Inference verification, lower/upper bound guarantees, and policy benchmark comparisons.",
            },
            {
                "domain_id": "alab_synthesis",
                "name": "A-Lab Autonomous Solid-State Inorganic Synthesis",
                "status": "HISTORICAL_REPLAY",
                "candidates_count": len(samples),
                "modalities": [
                    "XRD (Canonical)",
                    "REFINEMENT (Rietveld)",
                    "OUTCOME_TEST (Ordinal)",
                    "SEM (Precursor unlinked)",
                    "EDS (Precursor unlinked)",
                ],
                "purpose": "Offline replay of source-linked characterization and synthesis outcomes; not the original lab policy log.",
            },
            {
                "domain_id": "battery_electrolyte",
                "name": "LiFSI High-Entropy Battery Electrolyte Formulations",
                "status": "SCREENING_&_SURROGATE",
                "candidates_count": elec_sim.get("actual_search_space_size"),
                "modalities": ["SCREENING_FILTER", "SURROGATE_ORACLE"],
                "purpose": "Source-backed virtual-pool screening and in-silico surrogate trajectory evaluation; no live cycling.",
            },
        ],
    }

    # Build formal snapshot manifest
    manifest = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at_utc": now_utc,
        "presentation_build_commit": head_commit,
        "snapshot_generator_commit": head_commit,
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "scientific_source_commit_status": SCIENTIFIC_SOURCE_COMMIT_STATUS,
        "scientific_source_provenance_manifest": "presentation/data/scientific_source_provenance.json",
        "source_branch": current_branch,
        "source_artifact_hashes": source_hashes,
        "source_dataset_manifest_hash": source_hashes["precursor_genome_ledger"],
        "campaign_run_id": flagship["run_id"],
        "campaign_world": flagship["world"],
        "campaign_seed": flagship["seed"],
        "campaign_policy": flagship["policy"],
        "campaign_step_count": len(flagship["steps"]),
        "campaign_run_count": len(campaign_runs),
        "total_real_samples": len(samples),
        "total_audit_events": len(ledger_events),
        "validation_gate_pass_count": pass_count,
        "validation_gate_total_count": total_count,
    }

    # 12. Canonical Dataset Registry
    dataset_registry = build_dataset_registry(
        campaign_runs,
        hypotheses,
        samples,
        modality_inv,
        elec_screen,
        elec_sim,
        elec_target_audit,
    )
    print(f"Dataset registry created: {len(dataset_registry['datasets'])} authentic scientific benchmarks")

    # Assemble canonical snapshot
    snapshot = {
        "version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at": now_utc,
        "manifest": manifest,
        "dataset_registry": dataset_registry,
        "provenance": {
            "head_commit": head_commit,
            "snapshot_generator_commit": head_commit,
            "presentation_build_commit": head_commit,
            "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
            "scientific_source_commit_status": SCIENTIFIC_SOURCE_COMMIT_STATUS,
            "scientific_source_provenance_manifest": "presentation/data/scientific_source_provenance.json",
            "branch": current_branch,
            "total_ledger_events": len(ledger_events),
            "generated_at_utc": now_utc,
        },
        "flagship_campaign": flagship,
        "alab_replay_campaign": replay,
        "campaign_runs": campaign_runs,
        "hypotheses": hypotheses,
        "validation": validation,
        "calibration": calibration,
        "benchmarks": benchmark_data,
        "sensitivity": sensitivity_data,
        "electrolyte_screening": elec_screen,
        "electrolyte_simulation": elec_sim,
        "electrolyte_target_audit": elec_target_audit,
        "alab_audit": alab_audit,
        "modality_inventory": modality_inv,
        "samples": samples,
        "architecture": architecture,
        "ledger_sample_events": [e for e in ledger_events if e.get("run_id") == flagship["run_id"]],
    }

    # Write snapshot.json, snapshot_manifest.json, and dataset_registry.json
    with DEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    with MANIFEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with REGISTRY_FILE.open("w", encoding="utf-8") as f:
        json.dump(dataset_registry, f, indent=2)

    with SOURCE_PROVENANCE_FILE.open("w", encoding="utf-8") as f:
        json.dump(source_provenance, f, indent=2)

    # Run fail-closed runtime validation on generated snapshot
    from presentation.scripts.validate_snapshot import validate_snapshot_file
    validate_snapshot_file(DEST_FILE)
    print("Runtime schema validation: PASSED (all invariants verified)")

    # Automatically synchronize canonical snapshot to frontend public and dist directories
    frontend_public = FRONTEND_DIR / "public"
    frontend_dist = FRONTEND_DIR / "dist"

    if frontend_public.exists():
        shutil.copyfile(DEST_FILE, frontend_public / "snapshot.json")
        shutil.copyfile(MANIFEST_FILE, frontend_public / "snapshot_manifest.json")
        shutil.copyfile(REGISTRY_FILE, frontend_public / "dataset_registry.json")
        shutil.copyfile(SOURCE_PROVENANCE_FILE, frontend_public / "scientific_source_provenance.json")
        print(f"Synchronized snapshot and registry to {frontend_public}")

    if frontend_dist.exists():
        try:
            shutil.copyfile(DEST_FILE, frontend_dist / "snapshot.json")
            shutil.copyfile(MANIFEST_FILE, frontend_dist / "snapshot_manifest.json")
            shutil.copyfile(REGISTRY_FILE, frontend_dist / "dataset_registry.json")
            shutil.copyfile(SOURCE_PROVENANCE_FILE, frontend_dist / "scientific_source_provenance.json")
            print(f"Synchronized snapshot and registry to {frontend_dist}")
        except OSError as exc:
            print(f"Warning: frontend dist sync deferred ({exc})")

    size_mb = DEST_FILE.stat().st_size / (1024 * 1024)
    print(f"\nSUCCESS: Snapshot generated at {DEST_FILE} ({size_mb:.2f} MB)")
    print(f"SUCCESS: Manifest generated at {MANIFEST_FILE}")
    print(f"SUCCESS: Registry generated at {REGISTRY_FILE}")
    print(f"SUCCESS: Source provenance generated at {SOURCE_PROVENANCE_FILE}")


if __name__ == "__main__":
    main()
