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

SCIENTIFIC_SOURCE_COMMIT = "dc1f5fda1eb4327a4fe709de24a302643e0ecc8e"
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


def _parse_run_id(run_id: str) -> dict[str, Any]:
    parts = run_id.split(":")
    if parts[0] == "replay" and len(parts) == 4:
        return {"policy": parts[1], "seed": int(parts[2]), "mode": "HISTORICAL_REPLAY"}
    if parts[0] in {"controlled_world", "policy_comparison"} and len(parts) == 4:
        return {"world": parts[1], "seed": int(parts[2]), "policy": parts[3], "mode": "CONTROLLED_SYNTHETIC"}
    raise ValueError(f"Unsupported campaign run id: {run_id}")


def _build_recorded_campaign_run(run_id: str, ledger_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve one complete ledger run without inventing score components or features."""
    events = [e for e in ledger_events if e.get("run_id") == run_id]
    if not events:
        raise ValueError(f"Campaign run {run_id} not found in evidence ledger")
    metadata = _parse_run_id(run_id)
    by_step: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"scores": [], "preregistration": None, "observation": None, "belief_update": None}
    )
    for event in events:
        step = event.get("step")
        if not isinstance(step, int):
            raise ValueError(f"Run {run_id} contains an event without an integer step")
        event_type = event.get("event")
        if event_type == "ACTION_SCORE_RECORD":
            by_step[step]["scores"].append(event)
        elif event_type == "PREREGISTERED_SELECTED_ACTION":
            by_step[step]["preregistration"] = event
        elif event_type == "MEASUREMENT_REVEALED":
            by_step[step]["observation"] = event
        elif event_type == "BELIEF_UPDATE":
            by_step[step]["belief_update"] = event

    steps: list[dict[str, Any]] = []
    tested_candidates: list[str] = []
    candidate_modalities: dict[str, set[str]] = defaultdict(set)
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
        if (p_action.get("candidate_id"), p_action.get("action_type")) != (o_action.get("candidate_id"), o_action.get("action_type")):
            raise ValueError(f"Run {run_id} step {step_number} selected action does not match observation")
        beliefs = belief_update.get("beliefs_after")
        if not isinstance(beliefs, dict) or abs(sum(float(value) for value in beliefs.values()) - 1.0) > 1e-4:
            raise ValueError(f"Run {run_id} step {step_number} has invalid posterior beliefs")
        for score in scores:
            action = score.get("action", {})
            candidate_id = action.get("candidate_id")
            modality = action.get("action_type")
            if candidate_id and modality:
                candidate_modalities[candidate_id].add(modality)
            if "total_action_score" not in score:
                raise ValueError(f"Run {run_id} step {step_number} has an unscored action record")
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

        # Find canonical refinement case Rwp if available
        rwp = None
        for sc in scans:
            cases = sc.get("refinement_cases", [])
            if cases and cases[0].get("rwp") is not None:
                try:
                    rwp = float(cases[0]["rwp"])
                    break
                except (ValueError, TypeError):
                    pass

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
            "refinement_available": rwp is not None or bool(any(sc.get("refinement_cases") for sc in scans)),
            "sem_available": False,
            "eds_available": False,
            "sem_availability_reason": "Archive present in sem.zip (408 MB) but lacks sample-level linkage (precursor-level only).",
            "eds_availability_reason": "Archive present in eds.zip (358 KB) but lacks sample-level linkage (precursor-level only).",
            "refinement_rwp": rwp,
            "refinement_phases": phases,
            "refinement_phases_preview": phases[:4],
            "source_archive": "data/external/precursor_genome_2026/ledger_precursor_genome.json",
            "source_record_identifier": sid,
            "extractor_name": "ALabSourceLedgerExtractor.direct_v1",
            "extractor_version": "1.1.0",
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
    source_run_ids = [run.get("run_id") for run in campaign_runs if run.get("run_id")]
    hypothesis_ids = sorted(hypotheses)
    replay_ids = sorted({candidate_id for run in replay_runs for candidate_id in run.get("replay_candidate_ids", [])})

    return {
        "registry_schema_version": "1.0.0",
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
                "availableRunIds": [run_id for run_id in source_run_ids if run_id.startswith(("controlled_world:", "policy_comparison:"))],
                "availablePolicies": sorted({run.get("policy") for run in controlled_runs if run.get("policy")}),
                "availableSeeds": sorted({run.get("seed") for run in controlled_runs if run.get("seed") is not None}),
                "availableWorlds": sorted({run.get("world") for run in controlled_runs if run.get("world")}),
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
                    "competingHypotheses": True,
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
                "availableRunIds": [run_id for run_id in source_run_ids if run_id.startswith("replay:")],
                "availablePolicies": sorted({run.get("policy") for run in replay_runs if run.get("policy")}),
                "availableSeeds": sorted({run.get("seed") for run in replay_runs if run.get("seed") is not None}),
                "modalities": alab_modality_entries,
                "hypotheses": hypothesis_ids,
                "defaultConfiguration": {
                    "runId": replay_default["run_id"],
                    "seed": replay_default["seed"],
                    "policy": replay_default["policy"],
                },
                "capabilities": {
                    "competingHypotheses": False,
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
                "availablePolicies": sorted(simulation_policies),
                "availableSeeds": simulation_seeds,
                "capabilities": {
                    "competingHypotheses": False,
                    "candidateScreening": True,
                    "preregistrationReplay": False,
                    "closedLoopExecution": True,
                    "surrogateSimulation": True,
                    "evidenceKind": "SIMULATED_SURROGATE",
                },
                "summary": f"Source diagnostic reports a {electrolyte_simulation.get('requested_search_space_size')} candidate virtual pool, a {electrolyte_simulation.get('screened_working_set_size')} candidate working set, and {electrolyte_simulation.get('surrogate_model_family')} surrogate trajectories.",
                "statusBadge": "Screening & Surrogate",
                "disclosures": [
                    "Surrogate oracle is an ExtraTrees in-silico computational approximation only; no live physical battery cycling is performed.",
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
                "candidates_count": 12,
                "modalities": ["XRD", "REFINEMENT", "SEM", "EDS", "OUTCOME_TEST"],
                "purpose": "Inference verification, lower/upper bound guarantees, and policy benchmark comparisons.",
            },
            {
                "domain_id": "alab_synthesis",
                "name": "A-Lab Autonomous Solid-State Inorganic Synthesis",
                "status": "HISTORICAL_REPLAY",
                "candidates_count": 1035,
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
                "candidates_count": 333333,
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
        print(f"Synchronized snapshot and registry to {frontend_public}")

    if frontend_dist.exists():
        shutil.copyfile(DEST_FILE, frontend_dist / "snapshot.json")
        shutil.copyfile(MANIFEST_FILE, frontend_dist / "snapshot_manifest.json")
        shutil.copyfile(REGISTRY_FILE, frontend_dist / "dataset_registry.json")
        print(f"Synchronized snapshot and registry to {frontend_dist}")

    size_mb = DEST_FILE.stat().st_size / (1024 * 1024)
    print(f"\nSUCCESS: Snapshot generated at {DEST_FILE} ({size_mb:.2f} MB)")
    print(f"SUCCESS: Manifest generated at {MANIFEST_FILE}")
    print(f"SUCCESS: Registry generated at {REGISTRY_FILE}")


if __name__ == "__main__":
    main()
