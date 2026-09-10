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
        "electrolyte_target_audit": ROOT / "outputs" / "electrolyte" / "audit" / "experimental_identity_audit.json",
    }

    for name, expected_hash in hashes.items():
        path = artifact_paths.get(name)
        assert path is not None, f"Untracked artifact name: {name}"
        assert path.exists(), f"Source artifact missing on disk: {path}"
        actual_hash = compute_sha256(path)
        assert actual_hash == expected_hash, f"Hash mismatch for {name}: {actual_hash} != {expected_hash}"

    assert manifest["scientific_source_commit"] is None
    assert manifest["scientific_source_commit_status"] == "UNVERIFIED"
    provenance_path = ROOT / manifest["scientific_source_provenance_manifest"]
    provenance = json.load(provenance_path.open("r", encoding="utf-8"))
    assert provenance["scientific_source_commit"] is None
    assert provenance["scientific_source_commit_status"] == "UNVERIFIED"
    assert {item["artifact_name"] for item in provenance["artifacts"]} == set(hashes)


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
    assert "outcome_utility" not in pg_0309
    assert pg_0309["refinement_rwp"] == 1.17
    assert len(pg_0309["refinement_cases"]) == 2
    assert pg_0309["selected_refinement_case_id"].endswith(":case:0")
    assert pg_0309["refinement_selection_rule"] == "source_canonical_active_scan_and_case"

    for s in samples:
        assert s.get("source_archive") == "data/external/precursor_genome_2026/ledger_precursor_genome.json"
        assert s.get("source_record_identifier") == s["sample_id"]
        assert s.get("extractor_name") == "ALabSourceLedgerExtractor.direct_v1"
        assert isinstance(s.get("precursors"), list)


def test_source_score_records_are_preserved_without_recomputation(snapshot):
    flagship = snapshot["flagship_campaign"]
    for step in flagship["steps"]:
        s_num = step["step"]
        actions = step.get("all_scored_actions") or []
        assert actions, f"Step {s_num} has no scored actions!"
        for act in actions:
            assert all(isinstance(act[field], (int, float)) for field in ["expected_hig_nats", "discovery_utility", "normalized_cost", "total_action_score"])
            assert "normalized_hig" not in act
            assert "normalized_discovery" not in act
            assert "w_hig" not in act


def test_step_1_source_ranking_and_run_identity(snapshot):
    step1 = snapshot["flagship_campaign"]["steps"][0]
    actions = step1["all_scored_actions"]
    assert len(actions) == 24, f"Step 1 action pool must have 24 actions, found {len(actions)}"

    ranked = sorted(actions, key=lambda x: x["total_action_score"], reverse=True)
    assert ranked[0]["action"]["candidate_id"] == snapshot["flagship_campaign"]["steps"][0]["preregistration"]["action"]["candidate_id"]
    assert ranked[0]["action"]["action_type"] == snapshot["flagship_campaign"]["steps"][0]["preregistration"]["action"]["action_type"]
    assert snapshot["flagship_campaign"]["run_id"] == "policy_comparison:WORLD_H1_PHASE_PURITY:42:HYBRID"


def test_predictive_distribution_action_binding(snapshot):
    """Verify predictive distributions are strictly bound to preregistered actions."""
    flagship = snapshot["flagship_campaign"]
    for step in flagship["steps"]:
        s_num = step["step"]
        prereg = step["preregistration"]
        selected_candidate = prereg["action"]["candidate_id"]
        selected_modality = prereg["action"]["action_type"]

        dists = prereg["predictive_distributions"]
        assert len(dists) == 3, f"Step {s_num} must have exactly 3 hypothesis distributions"

        for hid, dist in dists.items():
            assert dist["candidate_id"] == selected_candidate
            assert dist["modality"] == selected_modality
            # Strictly positive variances
            for v in dist.get("variance", []):
                assert v > 0, f"Step {s_num} {hid} non-positive variance: {v}"

        for act in step.get("all_scored_actions", []):
            assert "predictive_distribution_available" not in act
            assert "predictive_distribution_unavailability_reason" not in act


def test_belief_continuity_and_sequence_ordering(snapshot):
    flagship = snapshot["flagship_campaign"]
    init_beliefs = flagship["initial_beliefs"]
    prev_beliefs = init_beliefs

    for step in flagship["steps"]:
        s_num = step["step"]
        prereg = step["preregistration"]
        obs = step["observation"]
        upd = step["belief_update"]

        # Sequence ordering
        assert prereg["event_sequence"] < obs["event_sequence"] < upd["event_sequence"]

        # Action consistency
        assert prereg["action"]["candidate_id"] == obs["action"]["candidate_id"]
        assert prereg["action"]["action_type"] == obs["action"]["action_type"]

        # Belief continuity
        prior = prereg["beliefs_before"]
        for hid, p_val in prior.items():
            assert abs(p_val - prev_beliefs[hid]) < 1e-4

        post = upd["beliefs_after"]
        assert abs(sum(post.values()) - 1.0) < 1e-4
        prev_beliefs = post


def test_tested_candidates_before_tracking(snapshot):
    steps = snapshot["flagship_campaign"]["steps"]
    tested_so_far = []
    for step in steps:
        assert step.get("tested_candidates_before") == tested_so_far
        cand = step["preregistration"]["action"]["candidate_id"]
        if cand not in tested_so_far:
            tested_so_far.append(cand)


def test_benchmark_claims_grounded_in_artifacts(snapshot):
    benchmarks = snapshot["benchmarks"]
    assert benchmarks["status"] == "METHODOLOGY_VALID"
    assert benchmarks["trajectory_count"] == 180

    swp = benchmarks["summary_by_world_policy"]

    # Q1: Clean H1 MAP recovery is 1.0 (100%)
    clean_h1_hybrid = swp["CLEAN_WORLD_H1_PHASE_PURITY"]["HYBRID"]
    assert clean_h1_hybrid["recovery_rate_MAP"] == 1.0
    clean_h2_hybrid = swp["CLEAN_WORLD_H2_COMPOSITION_HOMOGENEITY"]["HYBRID"]
    assert clean_h2_hybrid["recovery_rate_MAP"] == 1.0

    # Q3: Hybrid vs Pure HIG cost reduction in Clean H1: 1.95 down to 1.55 (20.5% reduction)
    h1_cost_pure = swp["CLEAN_WORLD_H1_PHASE_PURITY"]["PURE_HIG"]["mean_measurement_cost"]
    h1_cost_hybrid = clean_h1_hybrid["mean_measurement_cost"]
    h1_reduction = (h1_cost_pure - h1_cost_hybrid) / h1_cost_pure
    assert 0.20 < h1_reduction < 0.21

    # Q4: Calibration coverage
    calib = snapshot["calibration"]
    xrd_calib = calib["XRD"]["XRD.normalized_intensity_std_proxy"]
    assert abs(xrd_calib["coverage50"] - 0.601) < 0.01
    assert abs(xrd_calib["coverage90"] - 0.914) < 0.01


def test_no_synthetic_fabrications_or_fake_constants():
    frontend_src = ROOT / "presentation" / "frontend" / "src"
    ts_files = list(frontend_src.rglob("*.tsx")) + list(frontend_src.rglob("*.ts"))

    forbidden_patterns = [
        (r"\*\s*0\.18", "Fake 0.18 cost multiplier"),
        (r"i\s*%\s*7\s*===\s*0", "Fake i % 7 Pareto selection"),
        (r"i\s*%\s*5\s*===\s*0", "Fake i % 5 tested selection"),
        (r"VoI:\s*\+0\.482", "Hardcoded fake VoI string"),
        (r"SYNTHESIZE NEXT ELEMENT", "Old misleading synthesis button"),
        (r"0\.746", "Synthetic HIG normalization fallback 0.746"),
        (r"0\.978", "Synthetic discovery normalization fallback 0.978"),
        (r"0\.5066", "Synthetic raw HIG fallback 0.5066"),
        (r"0\.2861", "Synthetic net score fallback 0.2861"),
        (r"0\.536", "Synthetic HIG fallback 0.536"),
        (r"0\.7886", "Synthetic latent max fallback 0.7886"),
        (r"\?\?\s*0\.88", "Synthetic utility fallback 0.88"),
        (r"\?\?\s*0\.3333", "Synthetic prior fallback 0.3333"),
        (r"\?\?\s*['\"]1\.0['\"]", "Synthetic string cost fallback '1.0'"),
    ]

    for fpath in ts_files:
        text = fpath.read_text(encoding="utf-8")
        for pat, desc in forbidden_patterns:
            matches = re.findall(pat, text)
            assert not matches, f"Found forbidden pattern '{desc}' in {fpath.name}: {matches}"


def test_dataset_registry_completeness(snapshot):
    """Verify that the dataset registry is present, complete, and contains only authentic datasets."""
    registry_file = DATA_DIR / "dataset_registry.json"
    assert registry_file.exists(), "presentation/data/dataset_registry.json must exist"

    with registry_file.open("r", encoding="utf-8") as f:
        file_registry = json.load(f)

    snapshot_registry = snapshot.get("dataset_registry")
    assert snapshot_registry is not None, "snapshot.json must contain dataset_registry"
    assert file_registry == snapshot_registry, "dataset_registry.json and snapshot.dataset_registry must match exactly"

    datasets_list = snapshot_registry.get("datasets", [])
    datasets = {d["id"]: d for d in datasets_list}
    expected_ids = {
        "controlled_multimodal_alloy",
        "alab_precursor_genome",
        "anode_free_electrolyte_screening",
    }
    assert set(datasets.keys()) == expected_ids, f"Dataset IDs must be exactly {expected_ids}, got {set(datasets.keys())}"

    # 1. Controlled Multimodal Alloy
    alloy = datasets["controlled_multimodal_alloy"]
    assert alloy["candidateCount"] == 12
    assert alloy["provenance"]["sourceType"] == "IN_SILICO_BENCHMARK"
    assert alloy["capabilities"]["modelHypothesesAvailable"] is True
    assert alloy["capabilities"]["posteriorModelWeightsAvailable"] is True
    assert alloy["capabilities"]["mutuallyExclusivePhysicalMechanismsClaimed"] is True
    assert alloy["capabilities"]["prospectiveMechanismIdentification"] is False
    assert alloy["capabilities"]["closedLoopExecution"] is True
    assert alloy["capabilities"]["preregistrationReplay"] is True
    alloy_modalities = set(alloy["modalities"].keys())
    assert {"XRD", "REFINEMENT", "OUTCOME_TEST"}.issubset(alloy_modalities)

    # 2. A-Lab Precursor Genome
    alab = datasets["alab_precursor_genome"]
    assert alab["candidateCount"] == 1035
    assert alab["provenance"]["sourceType"] == "PEER_REVIEWED_BENCHMARK"
    assert alab["capabilities"]["modelHypothesesAvailable"] is True
    assert alab["capabilities"]["posteriorModelWeightsAvailable"] is True
    assert alab["capabilities"]["mutuallyExclusivePhysicalMechanismsClaimed"] is False
    assert alab["capabilities"]["prospectiveMechanismIdentification"] is False
    assert alab["capabilities"]["closedLoopExecution"] is False
    assert alab["capabilities"]["preregistrationReplay"] is True
    assert alab["provenance"]["doi"] == "10.5281/zenodo.21285546"
    assert alab["provenance"]["license"] == "CC BY 4.0"
    alab_modalities = {k: v["available"] for k, v in alab["modalities"].items()}
    assert alab_modalities.get("XRD") is True
    assert alab_modalities.get("REFINEMENT") is True
    assert alab_modalities.get("SEM") is False
    assert alab_modalities.get("EDS") is False

    # 3. Anode-Free Electrolyte Screening
    electrolyte = datasets["anode_free_electrolyte_screening"]
    assert electrolyte["candidateCount"] == 333333
    assert electrolyte["screenedWorkingSetCount"] == 200
    assert electrolyte["capabilities"]["surrogateSimulation"] is True
    assert electrolyte["capabilities"]["modelHypothesesAvailable"] is False
    assert electrolyte["capabilities"]["posteriorModelWeightsAvailable"] is False
    assert electrolyte["capabilities"]["mutuallyExclusivePhysicalMechanismsClaimed"] is False
    assert electrolyte["capabilities"]["prospectiveMechanismIdentification"] is False
    assert electrolyte["provenance"]["doi"] == "10.1038/s41467-025-63303-7"
    assert electrolyte["targetObservable"] == "norm_capacity_3"
    assert electrolyte["scientificTargetName"] == "C_norm^20"
    assert "20th cycle" in electrolyte["targetObservableDescription"]


def test_dataset_registry_zero_fabrications(snapshot):
    """Verify registry counts and metadata strictly agree with raw source artifacts."""
    datasets_list = snapshot["dataset_registry"]["datasets"]
    datasets = {d["id"]: d for d in datasets_list}

    # Verify A-Lab sample count against raw external ledger
    raw_alab_path = ROOT / "data" / "external" / "precursor_genome_2026" / "ledger_precursor_genome.json"
    with raw_alab_path.open("r", encoding="utf-8") as f:
        raw_alab = json.load(f)
    assert len(raw_alab["samples"]) == datasets["alab_precursor_genome"]["candidateCount"] == 1035

    # Verify Electrolyte screening metrics against raw diagnostic artifact
    raw_screening_path = ROOT / "outputs" / "electrolyte" / "benchmark" / "screening_quality_diagnostics.json"
    with raw_screening_path.open("r", encoding="utf-8") as f:
        raw_screening = json.load(f)
    assert raw_screening["search_space_size"] == datasets["anode_free_electrolyte_screening"]["candidateCount"] == 333333
    assert raw_screening["working_set_trials"]["200"]["screening_latent_gap"] == 0.0
    target_audit = json.load((ROOT / "outputs" / "electrolyte" / "audit" / "experimental_identity_audit.json").open("r", encoding="utf-8"))
    assert target_audit["target_semantics"]["scientific_target_name"] == "C_norm^20"
    assert target_audit["target_semantics"]["numerical_alias_validation"]["exceptions_count"] == 0

    # Verify Flagship policy matrix runs against raw multimodal artifact
    raw_matrix_path = ROOT / "outputs" / "alab" / "multimodal" / "full_policy_matrix.json"
    with raw_matrix_path.open("r", encoding="utf-8") as f:
        raw_matrix = json.load(f)
    assert raw_matrix["trajectory_count"] == 180


def test_campaign_resolution_isolation(snapshot):
    """Verify that dataset sample pools and candidate identities remain completely isolated."""
    # Controlled alloy candidates
    controlled_cands = snapshot["flagship_campaign"].get("candidates", [])
    assert len(controlled_cands) == 12
    for c in controlled_cands:
        assert c["candidate_id"].startswith("controlled-")

    # A-Lab sample pool
    samples = snapshot.get("samples", [])
    assert len(samples) == 1035
    for s in samples:
        assert s["sample_id"].startswith("PG_")

    # Electrolyte virtual screening
    raw_simulation = ROOT / "outputs" / "electrolyte" / "benchmark" / "surrogate_simulation.json"
    with raw_simulation.open("r", encoding="utf-8") as f:
        sim_data = json.load(f)
    hybrid_runs = sim_data["detailed_policy_seed_runs"]["HYBRID_DEFAULT"]
    assert len(hybrid_runs) >= 1
    queried_cands = hybrid_runs[0]["queried_candidate_ids"]
    assert len(queried_cands) == 15
    for cid in queried_cands:
        assert cid.startswith("ELEC_")


def test_exact_source_configuration_matrices(snapshot):
    datasets = {dataset["id"]: dataset for dataset in snapshot["dataset_registry"]["datasets"]}

    def tuple_key(configuration):
        return tuple(configuration.get(key) for key in ("configurationId", "runId", "world", "seed", "policy", "mode", "stepCount"))

    for dataset_id, mode in (("controlled_multimodal_alloy", "CONTROLLED_SYNTHETIC"), ("alab_precursor_genome", "HISTORICAL_REPLAY")):
        expected = {
            (run["run_id"], run["run_id"], run.get("world"), run["seed"], run["policy"], mode, len(run["steps"]))
            for run in snapshot["campaign_runs"] if run.get("mode") == mode
        }
        actual = {tuple_key(configuration) for configuration in datasets[dataset_id]["availableConfigurations"]}
        assert actual == expected

    expected_electrolyte = {
        (f"{policy}::{run['seed']}", None, None, run["seed"], policy, "SIMULATED_SURROGATE", len(run["queried_candidate_ids"]))
        for policy, runs in snapshot["electrolyte_simulation"]["detailed_policy_seed_runs"].items()
        for run in runs
    }
    actual_electrolyte = {tuple_key(configuration) for configuration in datasets["anode_free_electrolyte_screening"]["availableConfigurations"]}
    assert actual_electrolyte == expected_electrolyte


def test_alab_featured_source_metadata_joins(snapshot):
    raw = json.load((ROOT / "data" / "external" / "precursor_genome_2026" / "ledger_precursor_genome.json").open("r", encoding="utf-8"))
    joined = {sample["sample_id"]: sample for sample in snapshot["samples"]}
    source = {sample["sample_id"]: sample for sample in raw["samples"]}
    for sample_id in ("PG_0309", "PG_0214", "PG_0209"):
        assert joined[sample_id]["target_formula"] == source[sample_id]["target_compound"]
        assert joined[sample_id]["target_stoichiometry"] == source[sample_id].get("target_stoichiometry")
        assert joined[sample_id]["source_record_identifier"] == sample_id
        assert joined[sample_id]["refinement_available"] is True
        assert joined[sample_id]["selected_refinement_case_id"] is not None
        assert joined[sample_id]["refinement_selection_rule"]


