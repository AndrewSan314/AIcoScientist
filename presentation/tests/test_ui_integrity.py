import hashlib
import json
import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
PRESENTATION_DIR = ROOT / "presentation"
DATA_DIR = PRESENTATION_DIR / "data"
SNAPSHOT_PATH = DATA_DIR / "snapshot.json"
MANIFEST_PATH = DATA_DIR / "snapshot_manifest.json"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def snapshot():
    assert SNAPSHOT_PATH.exists(), "snapshot.json missing!"
    with SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST_PATH.exists(), "snapshot_manifest.json missing!"
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_manifest_artifact_hashes(manifest):
    hashes = manifest.get("source_artifact_hashes", {})
    assert len(hashes) >= 10, f"Expected at least 10 tracked source artifacts, found {len(hashes)}"

    artifact_paths = {
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
        assert path is not None, f"Untracked artifact name: {name}"
        assert path.exists(), f"Source artifact missing on disk: {path}"
        actual_hash = compute_sha256(path)
        assert actual_hash == expected_hash, f"Hash mismatch for {name}: {actual_hash} != {expected_hash}"


def test_alab_sample_provenance_and_count(snapshot):
    samples = snapshot.get("samples", [])
    assert len(samples) == 1035, f"Expected exactly 1,035 samples, found {len(samples)}"

    pg_0309 = next((s for s in samples if s["sample_id"] == "PG_0309"), None)
    assert pg_0309 is not None, "PG_0309 missing from samples!"
    assert pg_0309["target_formula"] == "Co3B3H9O13", f"PG_0309 formula fabricated: {pg_0309['target_formula']}"
    assert "B(OH)3" in pg_0309["precursors"], "B(OH)3 precursor missing in PG_0309"
    assert "Co3O4" in pg_0309["precursors"], "Co3O4 precursor missing in PG_0309"
    assert pg_0309["heating_temperature_c"] == 200.0
    assert pg_0309["reaction_category"] == "transformed"
    assert pg_0309["outcome_utility"] == 0.75
    assert pg_0309["refinement_rwp"] == 1.17

    for s in samples:
        assert s.get("source_archive") == "data/external/precursor_genome_2026/ledger_precursor_genome.json"
        assert s.get("source_record_identifier") == s["sample_id"]
        assert s.get("extractor_name") == "ALabDomainAdapter.canonical_extractor"
        assert isinstance(s.get("precursors"), list)


def test_score_decomposition_exact_formula(snapshot):
    flagship = snapshot["flagship_campaign"]
    weights = flagship["policy_weights"]
    w_h = weights["w_hig"]
    w_d = weights["w_discovery"]
    w_c = weights["w_cost"]

    for step in flagship["steps"]:
        s_num = step["step"]
        actions = step.get("all_scored_actions") or step.get("top_actions") or []
        assert actions, f"Step {s_num} has no scored actions!"

        for act in actions:
            norm_hig = act["normalized_hig"]
            norm_disc = act["normalized_discovery"]
            norm_cost = act["normalized_cost"]

            expected_contrib_h = w_h * norm_hig
            expected_contrib_d = w_d * norm_disc
            expected_contrib_c = w_c * norm_cost

            recomputed_score = expected_contrib_h + expected_contrib_d - expected_contrib_c
            total_score = act["total_action_score"]

            assert abs(recomputed_score - total_score) < 1e-5, f"Step {s_num} action score mismatch: {recomputed_score} vs {total_score}"


def test_step_1_counterfactual_policy_rankings(snapshot):
    step1 = snapshot["flagship_campaign"]["steps"][0]
    actions = step1["all_scored_actions"]
    assert len(actions) == 24, f"Step 1 action pool must have 24 actions, found {len(actions)}"

    # 1. HYBRID winner
    hybrid_sorted = sorted(actions, key=lambda x: x["total_action_score"], reverse=True)
    assert hybrid_sorted[0]["action"]["candidate_id"] == "controlled-3"
    assert hybrid_sorted[0]["action"]["action_type"] == "XRD"
    assert abs(hybrid_sorted[0]["total_action_score"] - 0.37938) < 1e-4

    # 2. PURE_HIG winner
    hig_sorted = sorted(actions, key=lambda x: x["raw_expected_hig_nats"], reverse=True)
    assert hig_sorted[0]["action"]["candidate_id"] == "controlled-0"
    assert hig_sorted[0]["action"]["action_type"] == "XRD"
    assert hig_sorted[0]["normalized_hig"] == 1.0

    # 3. DISCOVERY_ONLY winner
    disc_sorted = sorted(actions, key=lambda x: x["raw_discovery_utility"], reverse=True)
    assert disc_sorted[0]["action"]["candidate_id"] == "controlled-9"
    assert disc_sorted[0]["action"]["action_type"] == "XRD"
    assert disc_sorted[0]["normalized_discovery"] == 1.0


def test_no_synthetic_fabrications_or_fake_constants():
    frontend_src = ROOT / "presentation" / "frontend" / "src"
    ts_files = list(frontend_src.rglob("*.tsx")) + list(frontend_src.rglob("*.ts"))

    forbidden_patterns = [
        (r"\*\s*0\.18", "Fake 0.18 cost multiplier"),
        (r"i\s*%\s*7\s*===\s*0", "Fake i % 7 Pareto selection"),
        (r"i\s*%\s*5\s*===\s*0", "Fake i % 5 tested selection"),
        (r"VoI:\s*\+0\.482", "Hardcoded fake VoI string"),
        (r"SYNTHESIZE NEXT ELEMENT", "Old misleading synthesis button"),
    ]

    for fpath in ts_files:
        text = fpath.read_text(encoding="utf-8")
        for pat, desc in forbidden_patterns:
            matches = re.findall(pat, text)
            assert not matches, f"Found forbidden pattern '{desc}' in {fpath.name}: {matches}"
