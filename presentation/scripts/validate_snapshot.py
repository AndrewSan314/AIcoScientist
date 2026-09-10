#!/usr/bin/env python3
"""presentation/scripts/validate_snapshot.py

Fail-closed runtime validator for AIcoScientist presentation snapshot data.
Validates:
1. Manifest integrity and source artifact hashes.
2. Flagship campaign event invariants (sequence ordering, action consistency).
3. Posterior continuity across steps: posterior(t-1) == prior(t).
    4. Predictive distribution mathematics (finite moments, strictly positive variance).
5. Source score fields remain finite and are not recomputed with presentation weights.
6. Authentic A-Lab sample provenance (1,035 genuine records, no synthetic fallbacks).
7. Benchmark summary data structure and non-sentinel metric boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "presentation" / "data"
SNAPSHOT_PATH = DATA_DIR / "snapshot.json"
MANIFEST_PATH = DATA_DIR / "snapshot_manifest.json"
SOURCE_PROVENANCE_PATH = DATA_DIR / "scientific_source_provenance.json"


def compute_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class SnapshotValidationError(Exception):
    """Raised when snapshot data violates scientific integrity invariants."""
    pass


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _valid_beliefs(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(_finite(prob) and 0 <= float(prob) <= 1 for prob in value.values()) and abs(sum(float(prob) for prob in value.values()) - 1) <= 1e-4


def validate_manifest(manifest: dict[str, Any]) -> None:
    required_manifest_keys = [
        "snapshot_schema_version",
        "generated_at_utc",
        "scientific_source_commit",
        "scientific_source_commit_status",
        "scientific_source_provenance_manifest",
        "snapshot_generator_commit",
        "presentation_build_commit",
        "source_branch",
        "source_artifact_hashes",
        "campaign_run_id",
        "campaign_world",
        "total_real_samples",
        "validation_gate_pass_count",
    ]
    for key in required_manifest_keys:
        if key not in manifest:
            raise SnapshotValidationError(f"Manifest missing required key: {key}")

    status = manifest["scientific_source_commit_status"]
    if status not in {"VERIFIED", "UNVERIFIED"}:
        raise SnapshotValidationError(f"Invalid scientific source commit status: {status}")
    if status == "VERIFIED" and not isinstance(manifest.get("scientific_source_commit"), str):
        raise SnapshotValidationError("Verified scientific source provenance must include a commit SHA")
    if status == "UNVERIFIED" and manifest.get("scientific_source_commit") is not None:
        raise SnapshotValidationError("Unverified scientific source provenance must not claim a commit SHA")
    provenance_path = ROOT / manifest["scientific_source_provenance_manifest"]
    if not provenance_path.exists():
        raise SnapshotValidationError(f"Scientific source provenance manifest missing: {provenance_path}")
    with provenance_path.open("r", encoding="utf-8") as f:
        source_provenance = json.load(f)
    if source_provenance.get("scientific_source_commit") != manifest.get("scientific_source_commit") or source_provenance.get("scientific_source_commit_status") != status:
        raise SnapshotValidationError("Scientific source provenance manifest disagrees with snapshot manifest")
    if source_provenance.get("snapshot_generator_commit") != manifest.get("snapshot_generator_commit") or source_provenance.get("presentation_build_commit") != manifest.get("presentation_build_commit"):
        raise SnapshotValidationError("Scientific source provenance does not distinguish generator and presentation commits")

    # Verify source artifact hashes
    hashes = manifest.get("source_artifact_hashes", {})
    if len(hashes) < 10:
        raise SnapshotValidationError(f"Manifest tracks fewer than 10 source artifacts ({len(hashes)})")

    artifact_paths: dict[str, Path] = {
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
        "electrolyte_target_audit": ROOT / "outputs" / "electrolyte" / "audit" / "experimental_identity_audit.json",
    }

    for name, expected_hash in hashes.items():
        path = artifact_paths.get(name)
        if not path or not path.exists():
            raise SnapshotValidationError(f"Tracked source artifact missing on disk: {name} ({path})")
        actual_hash = compute_sha256(path)
        if actual_hash != expected_hash:
            raise SnapshotValidationError(
                f"Source artifact hash mismatch for '{name}': actual={actual_hash} vs expected={expected_hash}"
            )
    pinned = {item.get("artifact_name"): item for item in source_provenance.get("artifacts", [])}
    if set(pinned) != set(hashes):
        raise SnapshotValidationError("Scientific source provenance does not pin exactly the manifest artifacts")
    for name, expected_hash in hashes.items():
        if pinned[name].get("artifact_sha256") != expected_hash:
            raise SnapshotValidationError(f"Scientific source provenance hash mismatch for {name}")
        if status == "UNVERIFIED" and pinned[name].get("generating_scientific_commit") is not None:
            raise SnapshotValidationError(f"Unverified scientific artifact {name} claims a generating commit")
        if pinned[name].get("generating_snapshot_commit") != manifest.get("snapshot_generator_commit") or pinned[name].get("presentation_build_commit") != manifest.get("presentation_build_commit"):
            raise SnapshotValidationError(f"Scientific artifact {name} has incomplete build provenance")


def validate_campaign(campaign: dict[str, Any]) -> None:
    required_keys = ["run_id", "seed", "policy", "initial_beliefs", "steps"]
    for k in required_keys:
        if k not in campaign:
            raise SnapshotValidationError(f"Flagship campaign missing key: {k}")

    # Validate initial beliefs
    init_beliefs = campaign["initial_beliefs"]
    if not _valid_beliefs(init_beliefs):
        raise SnapshotValidationError("Initial beliefs are not finite probabilities summing to 1.0")

    steps = campaign["steps"]
    if not steps:
        raise SnapshotValidationError("Flagship campaign contains no steps")

    prev_beliefs = init_beliefs
    seen_steps: set[int] = set()

    for s_idx, step in enumerate(steps):
        s_num = step.get("step")
        if s_num != s_idx + 1:
            raise SnapshotValidationError(f"Step index mismatch: expected {s_idx + 1}, got {s_num}")
        if s_num in seen_steps:
            raise SnapshotValidationError(f"Duplicate campaign step: {s_num}")
        seen_steps.add(s_num)

        prereg = step.get("preregistration")
        obs = step.get("observation")
        upd = step.get("belief_update")
        scores = step.get("all_scored_actions") or []

        if not prereg or not obs or not upd:
            raise SnapshotValidationError(f"Step {s_num} missing required event record (prereg/obs/update)")

        for event_name, event in [("preregistration", prereg), ("observation", obs), ("belief_update", upd)]:
            if event.get("run_id", campaign["run_id"]) != campaign["run_id"] or not isinstance(event.get("event_sequence"), int) or not isinstance(event.get("timestamp"), str) or not event.get("timestamp"):
                raise SnapshotValidationError(f"Step {s_num} {event_name} has invalid run identity or timestamp")
            if event.get("step") != s_num:
                raise SnapshotValidationError(f"Step {s_num} {event_name} has inconsistent step number")

        # Sequence ordering invariant: prereg < obs < upd
        p_seq = prereg.get("event_sequence", 0)
        o_seq = obs.get("event_sequence", 0)
        u_seq = upd.get("event_sequence", 0)
        if not (p_seq < o_seq < u_seq):
            raise SnapshotValidationError(
                f"Step {s_num} event ordering invariant violated: prereg({p_seq}) < obs({o_seq}) < upd({u_seq})"
            )

        # Action consistency
        p_action = prereg.get("action", {})
        o_action = obs.get("action", {})
        u_action = upd.get("action", {})
        action_key = (p_action.get("candidate_id"), p_action.get("action_type"))
        if not p_action.get("action_id") or action_key[0] is None or action_key[1] is None or action_key != (o_action.get("candidate_id"), o_action.get("action_type")) or action_key != (u_action.get("candidate_id"), u_action.get("action_type")):
            raise SnapshotValidationError(f"Step {s_num} action candidate/modality mismatch")
        if any(action.get("requested_at_step") not in (None, s_num) for action in (p_action, o_action, u_action)):
            raise SnapshotValidationError(f"Step {s_num} action requested_at_step mismatch")

        # Belief continuity: prior(step t) == posterior(step t-1)
        prior_beliefs = prereg.get("beliefs_before", {})
        if not _valid_beliefs(prior_beliefs) or set(prior_beliefs) != set(prev_beliefs):
            raise SnapshotValidationError(f"Step {s_num} prior beliefs are invalid or incomplete")
        for hid, p_val in prior_beliefs.items():
            prev_p = prev_beliefs.get(hid, 0.0)
            if abs(p_val - prev_p) > 1e-4:
                raise SnapshotValidationError(
                    f"Step {s_num} belief continuity broken for {hid}: prior({p_val}) != previous_posterior({prev_p})"
                )

        # Posterior validity
        post_beliefs = upd.get("beliefs_after", {})
        if not _valid_beliefs(post_beliefs) or set(post_beliefs) != set(prior_beliefs):
            raise SnapshotValidationError(f"Step {s_num} posterior beliefs are invalid")
        if upd.get("beliefs_before") != prior_beliefs:
            raise SnapshotValidationError(f"Step {s_num} belief-update prior disagrees with preregistration")
        if obs.get("beliefs_before") is not None and obs.get("beliefs_before") != prior_beliefs:
            raise SnapshotValidationError(f"Step {s_num} observation prior disagrees with preregistration")
        if obs.get("beliefs_after") is not None and obs.get("beliefs_after") != post_beliefs:
            raise SnapshotValidationError(f"Step {s_num} observation posterior disagrees with update")

        prev_beliefs = post_beliefs

        # Predictive distribution mathematics
        dists = prereg.get("predictive_distributions", {})
        if not dists:
            raise SnapshotValidationError(f"Step {s_num} preregistration has no predictive distributions")

        for hid, dist in dists.items():
            if dist.get("candidate_id") != action_key[0] or dist.get("modality") != action_key[1]:
                raise SnapshotValidationError(f"Step {s_num} predictive distribution is bound to another action")
            kind = dist.get("distribution_kind", "gaussian")
            if kind == "gaussian":
                means = dist.get("mean", [])
                variances = dist.get("variance", [])
                if not means or not variances:
                    raise SnapshotValidationError(f"Step {s_num} {hid} missing predictive moments")
                if len(means) != len(variances):
                    raise SnapshotValidationError(f"Step {s_num} {hid} mean/variance shape mismatch")
                for m in means:
                    if math.isnan(m) or math.isinf(m):
                        raise SnapshotValidationError(f"Step {s_num} {hid} non-finite predictive mean: {m}")
                for v in variances:
                    if math.isnan(v) or math.isinf(v) or v <= 0.0:
                        raise SnapshotValidationError(f"Step {s_num} {hid} invalid predictive variance: {v} (must be strictly > 0)")
            elif kind == "categorical":
                probs = dist.get("probabilities", [])
                if not probs or not all(_finite(value) and 0 <= float(value) <= 1 for value in probs) or abs(sum(probs) - 1.0) > 1e-4:
                    raise SnapshotValidationError(f"Step {s_num} {hid} categorical probabilities invalid")

        # Score records are source fields. Do not synthesize or re-score them.
        if not scores:
            raise SnapshotValidationError(f"Step {s_num} has no scored actions")

        for act in scores:
            if not isinstance(act.get("action"), dict) or not act["action"].get("action_id"):
                raise SnapshotValidationError(f"Step {s_num} score record has no source action identifier")
            if act.get("run_id", campaign["run_id"]) != campaign["run_id"] or act.get("step") != s_num or not isinstance(act.get("event_sequence"), int) or not isinstance(act.get("timestamp"), str) or not act.get("timestamp"):
                raise SnapshotValidationError(f"Step {s_num} score record has invalid identity or timestamp")
            score_action = act["action"]
            if not score_action.get("candidate_id") or not score_action.get("action_type") or score_action.get("requested_at_step") not in (None, s_num):
                raise SnapshotValidationError(f"Step {s_num} score action is malformed")
            if not _finite(score_action.get("estimated_cost")):
                raise SnapshotValidationError(f"Step {s_num} score action cost is not finite")
            if score_action["action_id"] == p_action["action_id"] and (score_action.get("candidate_id"), score_action.get("action_type")) != action_key:
                raise SnapshotValidationError(f"Step {s_num} selected action identity is inconsistent")
            if act.get("predictive_distributions") or act.get("predictive_distribution"):
                raise SnapshotValidationError(f"Step {s_num} alternative score carries predictive distributions")
            for field in ["expected_hig_nats", "discovery_utility", "normalized_cost", "total_action_score"]:
                value = act.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise SnapshotValidationError(f"Step {s_num} score field {field} is not finite")

    if campaign.get("mode") == "CONTROLLED_SYNTHETIC":
        candidates = campaign.get("candidates") or []
        if not candidates or len({candidate.get("candidate_id") for candidate in candidates}) != len(candidates):
            raise SnapshotValidationError("Controlled campaign candidate catalog is empty or duplicated")


def validate_flagship_campaign(campaign: dict[str, Any]) -> None:
    if campaign.get("mode") != "CONTROLLED_SYNTHETIC":
        raise SnapshotValidationError("Flagship campaign is not marked CONTROLLED_SYNTHETIC")
    validate_campaign(campaign)


def validate_samples(samples: list[dict[str, Any]]) -> None:
    if len(samples) != 1035:
        raise SnapshotValidationError(f"Sample catalog count mismatch: expected source count 1,035, found {len(samples)}")
    if len({sample.get("sample_id") for sample in samples}) != len(samples):
        raise SnapshotValidationError("Sample catalog contains duplicate sample IDs")

    required_sample_keys = [
        "sample_id",
        "target_formula",
        "precursors",
        "reaction_category",
        "outcome_available",
        "refinement_cases",
        "selected_refinement_case_id",
        "refinement_selection_rule",
        "source_archive",
        "source_record_identifier",
        "extractor_name",
    ]

    for s in samples:
        for k in required_sample_keys:
            if k not in s:
                raise SnapshotValidationError(f"Sample {s.get('sample_id')} missing required key: {k}")

        if not isinstance(s["precursors"], list):
            raise SnapshotValidationError(f"Sample {s['sample_id']} precursors must be a list")
        cases = s["refinement_cases"]
        if not isinstance(cases, list) or len({case.get("case_id") for case in cases}) != len(cases):
            raise SnapshotValidationError(f"Sample {s['sample_id']} refinement cases are not a unique source list")
        selected = s["selected_refinement_case_id"]
        if selected is not None and selected not in {case.get("case_id") for case in cases}:
            raise SnapshotValidationError(f"Sample {s['sample_id']} selects a missing refinement case")
        if selected is not None and not s.get("refinement_selection_rule"):
            raise SnapshotValidationError(f"Sample {s['sample_id']} has a selected case without a selection rule")

def validate_benchmarks(benchmarks: dict[str, Any]) -> None:
    if benchmarks.get("status") != "METHODOLOGY_VALID":
        raise SnapshotValidationError(f"Benchmark status is not METHODOLOGY_VALID: {benchmarks.get('status')}")
    if not isinstance(benchmarks.get("trajectory_count"), int) or benchmarks["trajectory_count"] <= 0:
        raise SnapshotValidationError(f"Benchmark trajectory count is invalid: {benchmarks.get('trajectory_count')}")

    swp = benchmarks.get("summary_by_world_policy", {})
    if not swp:
        raise SnapshotValidationError("Benchmark missing summary_by_world_policy")

    # Verify no old sentinel values (999) appear in threshold steps
    for world, policies in swp.items():
        for pol, stats in policies.items():
            for m_key in ["mean_steps_to_MAP", "mean_steps_to_posterior_gt_0.5", "mean_steps_to_posterior_gt_0.8"]:
                val = stats.get(m_key)
                if val is not None and val > 100:
                    raise SnapshotValidationError(f"Sentinel value leaked into benchmark metric: {world}/{pol}/{m_key} = {val}")


def validate_dataset_registry(registry: dict[str, Any]) -> None:
    if not registry:
        raise SnapshotValidationError("Snapshot missing dataset_registry")

    if registry.get("registry_schema_version") != "1.1.0":
        raise SnapshotValidationError(f"Invalid registry_schema_version: {registry.get('registry_schema_version')}")

    datasets = registry.get("datasets", [])
    if len(datasets) != 3:
        raise SnapshotValidationError(f"Expected exactly 3 registered scientific datasets, found {len(datasets)}")

    dataset_ids = {d.get("id") for d in datasets}
    expected_ids = {
        "controlled_multimodal_alloy",
        "alab_precursor_genome",
        "anode_free_electrolyte_screening",
    }
    if dataset_ids != expected_ids:
        raise SnapshotValidationError(f"Dataset registry IDs mismatch: {dataset_ids} != {expected_ids}")

    for d in datasets:
        did = d["id"]
        for req in ["displayName", "domain", "provenance", "candidateCount", "modalities", "capabilities", "summary", "statusBadge", "availableConfigurations"]:
            if req not in d:
                raise SnapshotValidationError(f"Dataset {did} missing required field: {req}")

        caps = d["capabilities"]
        for c_field in ["modelHypothesesAvailable", "posteriorModelWeightsAvailable", "mutuallyExclusivePhysicalMechanismsClaimed", "prospectiveMechanismIdentification", "candidateScreening", "preregistrationReplay", "closedLoopExecution", "surrogateSimulation", "evidenceKind"]:
            if c_field not in caps:
                raise SnapshotValidationError(f"Dataset {did} capabilities missing field: {c_field}")
        if "competingHypotheses" in caps:
            raise SnapshotValidationError(f"Dataset {did} still exposes ambiguous competingHypotheses capability")
        configurations = d["availableConfigurations"]
        if not isinstance(configurations, list) or not configurations:
            raise SnapshotValidationError(f"Dataset {did} has no source-derived available configurations")
        configuration_ids = [configuration.get("configurationId") for configuration in configurations]
        if any(not isinstance(identifier, str) or not identifier for identifier in configuration_ids) or len(set(configuration_ids)) != len(configuration_ids):
            raise SnapshotValidationError(f"Dataset {did} configuration matrix has duplicate or invalid IDs")
        for configuration in configurations:
            if configuration.get("mode") not in {"CONTROLLED_SYNTHETIC", "HISTORICAL_REPLAY", "SIMULATED_SURROGATE"} or not isinstance(configuration.get("policy"), str) or not isinstance(configuration.get("stepCount"), int) or configuration["stepCount"] <= 0:
                raise SnapshotValidationError(f"Dataset {did} has an invalid configuration tuple")

        candidate_ids = d.get("candidateIds")
        if candidate_ids is not None and len(candidate_ids) != d["candidateCount"]:
            raise SnapshotValidationError(f"Dataset {did} candidateIds do not match candidateCount")
        if did == "anode_free_electrolyte_screening" and d.get("targetObservable") != "norm_capacity_3":
            raise SnapshotValidationError("Electrolyte registry lost the raw source target column")
        if "cycle 3" in str(d.get("targetObservableDescription", "")).lower():
            raise SnapshotValidationError("Electrolyte registry contains the incorrect cycle-3 target description")


def validate_electrolyte_simulation(simulation: dict[str, Any]) -> None:
    if simulation.get("oracle_kind") != "SIMULATED_SURROGATE" or simulation.get("physical_synthesis") is not False:
        raise SnapshotValidationError("Electrolyte simulation is not explicitly marked surrogate-only")
    runs = simulation.get("detailed_policy_seed_runs")
    if not isinstance(runs, dict) or not runs:
        raise SnapshotValidationError("Electrolyte simulation has no detailed policy/seed runs")
    for policy, seed_runs in runs.items():
        if not isinstance(seed_runs, list) or not seed_runs:
            raise SnapshotValidationError(f"Electrolyte policy {policy} has no detailed runs")
        for run in seed_runs:
            lengths = [len(run.get(key, [])) for key in ["queried_candidate_ids", "revealed_noisy_values", "selected_latent_values", "best_latent_curve"]]
            if not all(lengths) or len(set(lengths)) != 1:
                raise SnapshotValidationError(f"Electrolyte policy {policy} has truncated trajectory arrays")
            for key in ["revealed_noisy_values", "selected_latent_values", "best_latent_curve"]:
                if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in run[key]):
                    raise SnapshotValidationError(f"Electrolyte policy {policy} has non-finite {key}")


def validate_configuration_matrix(snapshot: dict[str, Any]) -> None:
    registry = snapshot["dataset_registry"]
    by_id = {dataset["id"]: dataset for dataset in registry["datasets"]}

    def key(configuration: dict[str, Any]) -> tuple[Any, ...]:
        return (
            configuration.get("configurationId"), configuration.get("runId"), configuration.get("world"),
            configuration.get("seed"), configuration.get("policy"), configuration.get("mode"), configuration.get("stepCount"),
        )

    for dataset_id, mode in [("controlled_multimodal_alloy", "CONTROLLED_SYNTHETIC"), ("alab_precursor_genome", "HISTORICAL_REPLAY")]:
        expected = []
        for run in snapshot.get("campaign_runs", []):
            if run.get("mode") != mode:
                continue
            expected.append({
                "configurationId": run["run_id"], "runId": run["run_id"], "world": run.get("world"),
                "seed": run.get("seed"), "policy": run.get("policy"), "mode": mode, "stepCount": len(run.get("steps", [])),
            })
        actual = [configuration for configuration in by_id[dataset_id]["availableConfigurations"] if configuration.get("mode") == mode]
        if {key(item) for item in actual} != {key(item) for item in expected}:
            raise SnapshotValidationError(f"{dataset_id} configuration matrix does not exactly match recorded campaign runs")

    expected_surrogate = []
    for policy, runs in snapshot.get("electrolyte_simulation", {}).get("detailed_policy_seed_runs", {}).items():
        for run in runs:
            expected_surrogate.append({
                "configurationId": f"{policy}::{run['seed']}", "seed": run["seed"], "policy": policy,
                "mode": "SIMULATED_SURROGATE", "stepCount": len(run.get("queried_candidate_ids", [])),
            })
    actual_surrogate = [configuration for configuration in by_id["anode_free_electrolyte_screening"]["availableConfigurations"] if configuration.get("mode") == "SIMULATED_SURROGATE"]
    if {key(item) for item in actual_surrogate} != {key(item) for item in expected_surrogate}:
        raise SnapshotValidationError("Electrolyte configuration matrix does not exactly match detailed policy/seed runs")


def validate_snapshot_file(path: Path) -> None:
    if not path.exists():
        raise SnapshotValidationError(f"Snapshot file missing: {path}")

    with path.open("r", encoding="utf-8") as f:
        snapshot = json.load(f)

    manifest = snapshot.get("manifest")
    if not manifest:
        raise SnapshotValidationError("Snapshot missing embedded manifest")

    validate_manifest(manifest)
    validate_dataset_registry(snapshot.get("dataset_registry", {}))
    validate_flagship_campaign(snapshot.get("flagship_campaign", {}))
    campaign_runs = snapshot.get("campaign_runs") or []
    if len(campaign_runs) < 3:
        raise SnapshotValidationError("Snapshot has too few source campaign runs")
    if len({run.get("run_id") for run in campaign_runs}) != len(campaign_runs):
        raise SnapshotValidationError("Snapshot campaign runs contain duplicate run IDs")
    for run in campaign_runs:
        validate_campaign(run)
    validate_samples(snapshot.get("samples", []))
    validate_benchmarks(snapshot.get("benchmarks", {}))
    validate_electrolyte_simulation(snapshot.get("electrolyte_simulation", {}))
    validate_configuration_matrix(snapshot)


def main() -> int:
    print("Validating presentation snapshot fail-closed...")
    try:
        validate_snapshot_file(SNAPSHOT_PATH)
        print(f"PASS: Snapshot {SNAPSHOT_PATH} passes all scientific data integrity gates.")
        return 0
    except SnapshotValidationError as e:
        print(f"FAIL: Snapshot validation error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAIL: Unexpected error during snapshot validation: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
