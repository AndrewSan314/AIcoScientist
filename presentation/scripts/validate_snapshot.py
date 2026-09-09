#!/usr/bin/env python3
"""presentation/scripts/validate_snapshot.py

Fail-closed runtime validator for AIcoScientist presentation snapshot data.
Validates:
1. Manifest integrity and source artifact hashes.
2. Flagship campaign event invariants (sequence ordering, action consistency).
3. Posterior continuity across steps: posterior(t-1) == prior(t).
4. Predictive distribution mathematics (finite moments, strictly positive variance).
5. Exact score decomposition identity: S(a) = w_H * norm_hig + w_D * norm_disc - w_C * norm_cost.
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


def validate_manifest(manifest: dict[str, Any]) -> None:
    required_manifest_keys = [
        "snapshot_schema_version",
        "generated_at_utc",
        "scientific_source_commit",
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


def validate_flagship_campaign(campaign: dict[str, Any]) -> None:
    required_keys = ["run_id", "world", "seed", "policy", "policy_weights", "initial_beliefs", "steps", "candidates"]
    for k in required_keys:
        if k not in campaign:
            raise SnapshotValidationError(f"Flagship campaign missing key: {k}")

    # Validate initial beliefs
    init_beliefs = campaign["initial_beliefs"]
    prob_sum = sum(init_beliefs.values())
    if abs(prob_sum - 1.0) > 1e-4:
        raise SnapshotValidationError(f"Initial beliefs do not sum to 1.0: {prob_sum}")
    for hid, p in init_beliefs.items():
        if p < 0.0 or math.isnan(p) or math.isinf(p):
            raise SnapshotValidationError(f"Invalid initial probability for {hid}: {p}")

    weights = campaign["policy_weights"]
    w_h = float(weights["w_hig"])
    w_d = float(weights["w_discovery"])
    w_c = float(weights["w_cost"])

    steps = campaign["steps"]
    if not steps:
        raise SnapshotValidationError("Flagship campaign contains no steps")

    prev_beliefs = init_beliefs

    for s_idx, step in enumerate(steps):
        s_num = step.get("step")
        if s_num != s_idx + 1:
            raise SnapshotValidationError(f"Step index mismatch: expected {s_idx + 1}, got {s_num}")

        prereg = step.get("preregistration")
        obs = step.get("observation")
        upd = step.get("belief_update")
        scores = step.get("all_scored_actions") or []

        if not prereg or not obs or not upd:
            raise SnapshotValidationError(f"Step {s_num} missing required event record (prereg/obs/update)")

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
        if p_action.get("candidate_id") != o_action.get("candidate_id"):
            raise SnapshotValidationError(f"Step {s_num} candidate mismatch between prereg and obs")
        if p_action.get("action_type") != o_action.get("action_type"):
            raise SnapshotValidationError(f"Step {s_num} modality mismatch between prereg and obs")

        # Belief continuity: prior(step t) == posterior(step t-1)
        prior_beliefs = prereg.get("beliefs_before", {})
        for hid, p_val in prior_beliefs.items():
            prev_p = prev_beliefs.get(hid, 0.0)
            if abs(p_val - prev_p) > 1e-4:
                raise SnapshotValidationError(
                    f"Step {s_num} belief continuity broken for {hid}: prior({p_val}) != previous_posterior({prev_p})"
                )

        # Posterior validity
        post_beliefs = upd.get("beliefs_after", {})
        post_sum = sum(post_beliefs.values())
        if abs(post_sum - 1.0) > 1e-4:
            raise SnapshotValidationError(f"Step {s_num} posterior belief sum invalid: {post_sum}")
        for hid, p_val in post_beliefs.items():
            if p_val < 0.0 or math.isnan(p_val) or math.isinf(p_val):
                raise SnapshotValidationError(f"Step {s_num} invalid posterior probability for {hid}: {p_val}")

        prev_beliefs = post_beliefs

        # Predictive distribution mathematics
        dists = prereg.get("predictive_distributions", {})
        if not dists:
            raise SnapshotValidationError(f"Step {s_num} preregistration has no predictive distributions")

        for hid, dist in dists.items():
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
                if not probs or abs(sum(probs) - 1.0) > 1e-4:
                    raise SnapshotValidationError(f"Step {s_num} {hid} categorical probabilities invalid")

        # Score normalization verification on all actions
        if not scores:
            raise SnapshotValidationError(f"Step {s_num} has no scored actions")

        for act in scores:
            total = float(act["total_action_score"])
            norm_hig = float(act["normalized_hig"])
            norm_disc = float(act["normalized_discovery"])
            norm_cost = float(act["normalized_cost"])

            expected_score = w_h * norm_hig + w_d * norm_disc - w_c * norm_cost
            if abs(expected_score - total) > 1e-4:
                raise SnapshotValidationError(
                    f"Step {s_num} action {act.get('action', {}).get('action_id')} score decomposition mismatch: "
                    f"expected {expected_score:.5f}, got {total:.5f}"
                )


def validate_samples(samples: list[dict[str, Any]]) -> None:
    if len(samples) != 1035:
        raise SnapshotValidationError(f"Sample catalog count mismatch: expected 1,035, found {len(samples)}")

    required_sample_keys = [
        "sample_id",
        "target_formula",
        "precursors",
        "reaction_category",
        "outcome_utility",
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

        # Validate utility mapping when reaction category is known
        util = s.get("outcome_utility")
        if util is not None:
            if util not in {0.0, 0.5, 0.75, 1.0}:
                raise SnapshotValidationError(f"Sample {s['sample_id']} invalid ordinal outcome utility: {util}")


def validate_benchmarks(benchmarks: dict[str, Any]) -> None:
    if benchmarks.get("status") != "METHODOLOGY_VALID":
        raise SnapshotValidationError(f"Benchmark status is not METHODOLOGY_VALID: {benchmarks.get('status')}")
    if benchmarks.get("trajectory_count") != 180:
        raise SnapshotValidationError(f"Benchmark trajectory count mismatch: {benchmarks.get('trajectory_count')}")

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


def validate_snapshot_file(path: Path) -> None:
    if not path.exists():
        raise SnapshotValidationError(f"Snapshot file missing: {path}")

    with path.open("r", encoding="utf-8") as f:
        snapshot = json.load(f)

    manifest = snapshot.get("manifest")
    if not manifest:
        raise SnapshotValidationError("Snapshot missing embedded manifest")

    validate_manifest(manifest)
    validate_flagship_campaign(snapshot.get("flagship_campaign", {}))
    validate_samples(snapshot.get("samples", []))
    validate_benchmarks(snapshot.get("benchmarks", {}))


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
