from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import pytest

import src.process.simulators.artistic.config as config_module
import src.process.simulators.artistic.provenance as provenance_module
import src.process.simulators.artistic.runner as runner_module
from src.datasets.battery_process.artistic import ArtisticSimulationAdapter
from src.process.simulators.base import SimulationResult, SimulationStatus
from src.process.simulators.artistic import ArtisticRecipe, ArtisticRunConfig, ArtisticSimulator, CalenderingRecipe, Checkpoint, ConvergenceRunEvidence, ConvergenceStatus, DryingMode, FidelityMode, MetricTolerancePolicy, ParticleCountSafetyError, ReferenceAgreementStatus, SlurryRecipe, StabilityPolicy, build_convergence_report, build_convergence_study_plan, estimate_particles, stability_status
from src.process.simulators.artistic.config import ExecutionMode, MPIEnvironmentError, PINNED_COMMIT, PINNED_SOURCE_TREE_HASH, SourcePinError, validate_mpi_environment
from src.process.simulators.artistic.parser import ParsedArtisticOutput, parse_artistic_output, parse_thermo_checkpoints, parse_thermo_log
from src.process.simulators.artistic.validation import output_errors
from src.process.simulators.artistic.runner import _version
from src.process.simulators.artistic.schemas import _lammps_round
from src.process.information_horizon import InformationHorizon
from src.process.stages import ProcessStage


def _slurry() -> SlurryRecipe:
    return SlurryRecipe(1, (5.0,) * 10, (1.0,) + (0.0,) * 9, 1.0, 0.5, 0.1, 0.9, 0.1, 0, 0.5)


def _recipe() -> ArtisticRecipe:
    return ArtisticRecipe(_slurry(), DryingMode.HOMOGENEOUS, None, CalenderingRecipe(0.25, 0.05, True, False))


def _data(atoms: int = 10, z: float = 12.0) -> str:
    return f"LAMMPS data\n\n{atoms} atoms\n\n0 10 xlo xhi\n0 10 ylo yhi\n0 {z} zlo zhi\n"


def _dump(*z_values: float, zlo: float = 0.0, zhi: float = 12.0) -> str:
    atoms = "\n".join(f"{index} 1 1 1 {z} 0.5" for index, z in enumerate(z_values, 1))
    return f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{len(z_values)}\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n{zlo} {zhi}\nITEM: ATOMS id type x y z radius\n{atoms}\n"


def _successful_result(root: Path, run_id: str, recipe: ArtisticRecipe) -> SimulationResult:
    run_directory = root / run_id; run_directory.mkdir()
    provenance = {
        "checked_out_commit": PINNED_COMMIT, "source_tree_hash": PINNED_SOURCE_TREE_HASH,
        "fidelity_mode": "REFERENCE", "requested_slurry_steps": 20_000_000, "completed_slurry_steps": 20_000_000,
        "dump_interval_steps": 1_000_000, "fidelity_identity": "reference-id", "reference_equivalence_status": "NOT_EVALUATED",
        "physics_config_fingerprint": "physics", "simulation_manifest_hash": "manifest",
        "rendered_source_file_hashes": {"workspace/in_slurry.run": "input-digest"}, "patches": ["workspace-only"],
        "output_hashes": {"workspace/coord_out_cal.data": "output-digest"}, "stage_lineage": [{"boundary": "drying_to_calendering"}],
        "executable_versions": {"lammps": "LAMMPS test", "python": "Python test"}, "executable_identity": {"lammps_command": "lmp-test"},
    }
    (run_directory / "manifest.json").write_text(json.dumps(provenance), encoding="utf-8")
    return SimulationResult(SimulationStatus.SUCCESS, run_id, run_directory, {"slurry": {"slurry_density": 1.0}, "drying": {"am_loading": 2.0, "drying_porosity_bulk_percent": 10.0, "drying_porosity_all_percent": 20.0}}, {"calendered_electrode_thickness": 8.0, "calendered_cbd_nanoporosity": 0.4, "calendered_porosity_bulk_percent": 5.0, "calendered_porosity_all_percent": 15.0}, provenance)


def _evidence(checkpoints: list[Checkpoint], mode: FidelityMode, steps: int, *, recipe: str = "recipe", physics: str = "physics", successful: bool = True) -> ConvergenceRunEvidence:
    return ConvergenceRunEvidence(
        run_id=f"{mode.value}-{steps}", recipe_fingerprint=recipe, pinned_commit="commit", pinned_source_tree_hash="tree",
        physics_config_fingerprint=physics, fidelity_mode=mode, fidelity_identity=f"{mode.value}-{steps}",
        requested_dynamic_steps=steps, dump_interval_steps=1_000_000, checkpoints=tuple(checkpoints),
        simulation_manifest_hash=f"manifest-{mode.value}-{steps}", successful=successful,
    )


def _legacy_stability_policy() -> StabilityPolicy:
    return StabilityPolicy(minimum_checkpoints=3)


@pytest.fixture
def pinned_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source"; root = source / "NMC" / "Updated version"; slurry = root / "Slurry"; slurry.mkdir(parents=True)
    (slurry / "user_inputs.txt").write_text("variable nAM_part equal @nAM_part@\n", encoding="utf-8")
    (slurry / "init_structure.txt").write_text("\n".join(f"variable n_AM{i} equal round(v_n_AM*v_p_AM6)" for i in range(7, 11)), encoding="utf-8")
    (slurry / "in_slurry.run").write_text("dump 1 all custom 1000000 dump.atom id type x y z radius\nvariable run equal 20000000\nrun ${run}\n", encoding="utf-8")
    for name, files in {"Drying_homogeneous": ("in_evap_hom.run", "pores.py"), "Calendering": ("in_cal.run", "pores_cal.py", "Reformatting_cal_electrode.py")}.items():
        directory = root / name; directory.mkdir()
        for file in files: (directory / file).write_text("# source fixture\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=source, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=source, check=True); subprocess.run(["git", "config", "user.name", "test"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True); subprocess.run(["git", "commit", "-m", "fixture"], cwd=source, check=True, capture_output=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD:NMC/Updated version"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
    for module in (config_module, provenance_module):
        monkeypatch.setattr(module, "PINNED_COMMIT", commit); monkeypatch.setattr(module, "PINNED_SOURCE_TREE_HASH", tree)
    return source


def test_source_pin_fails_closed_on_commit_mismatch(pinned_source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ArtisticRunConfig(source_root=pinned_source); config.verify_source_pin()
    monkeypatch.setattr(config_module, "PINNED_COMMIT", "0" * 40)
    with pytest.raises(SourcePinError, match="commit mismatch"): config.verify_source_pin()
    monkeypatch.setattr(config_module, "PINNED_COMMIT", subprocess.run(["git", "rev-parse", "HEAD"], cwd=pinned_source, check=True, capture_output=True, text=True).stdout.strip())
    monkeypatch.setattr(config_module, "PINNED_SOURCE_TREE_HASH", "0" * 40)
    with pytest.raises(SourcePinError, match="source tree mismatch"): config.verify_source_pin()


def test_source_pin_rejects_a_modified_checkout(pinned_source: Path) -> None:
    (pinned_source / "NMC" / "Updated version" / "Slurry" / "in_slurry.run").write_text("modified", encoding="utf-8")
    with pytest.raises(SourcePinError, match="modified"):
        ArtisticRunConfig(source_root=pinned_source).verify_source_pin()


def test_renderer_patch_is_workspace_only_and_run_ids_are_isolated(pinned_source: Path, tmp_path: Path) -> None:
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=pinned_source, output_root=tmp_path / "runs"))
    run = simulator.prepare(ArtisticRecipe(slurry=_slurry()), run_id="render")
    rendered = (run / "workspace" / "init_structure.txt").read_text(encoding="utf-8")
    assert "v_p_AM7" in rendered and "v_p_AM10" in rendered
    assert "v_p_AM6" in (pinned_source / "NMC" / "Updated version" / "Slurry" / "init_structure.txt").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to append"): simulator.prepare(ArtisticRecipe(slurry=_slurry()), run_id="render")


def test_short_horizon_is_explicit_workspace_patch_and_has_distinct_identity(pinned_source: Path, tmp_path: Path) -> None:
    config = ArtisticRunConfig(
        source_root=pinned_source, output_root=tmp_path / "runs", fidelity_mode=FidelityMode.SHORT_HORIZON,
        slurry_steps=500_000, dump_interval_steps=250_000,
    )
    run = ArtisticSimulator(config).prepare(ArtisticRecipe(slurry=_slurry()), run_id="short")
    rendered = (run / "workspace" / "in_slurry.run").read_text(encoding="utf-8")
    source = (pinned_source / "NMC" / "Updated version" / "Slurry" / "in_slurry.run").read_text(encoding="utf-8")
    assert "variable run equal 500000" in rendered and "custom 250000" in rendered
    assert "variable run equal 20000000" in source and "custom 1000000" in source
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fidelity_mode"] == "SHORT_HORIZON"
    assert manifest["requested_slurry_steps"] == 500_000
    assert manifest["reference_equivalence_status"] == "REFERENCE_NOT_AVAILABLE"
    assert {patch["id"] for patch in manifest["patches"]} >= {"short_horizon_slurry_steps", "short_horizon_checkpoint_interval"}
    assert config.fidelity_identity != ArtisticRunConfig(source_root=pinned_source).fidelity_identity


def test_fidelity_validation_and_study_plan_are_fail_closed_and_dry_run_only() -> None:
    with pytest.raises(ValueError, match="exactly 20,000,000"):
        ArtisticRunConfig(slurry_steps=500_000)
    with pytest.raises(ValueError, match="explicit slurry_steps"):
        ArtisticRunConfig(fidelity_mode=FidelityMode.SHORT_HORIZON)
    config = ArtisticRunConfig(fidelity_mode=FidelityMode.REFERENCE, confirm_reference_execution=False)
    plan = build_convergence_study_plan(config, steps_per_second=1000, recipe=_recipe())
    assert plan.dry_run is True and all(entry["auto_launch"] is False for entry in plan.entries)
    assert plan.entries[0]["estimated_wall_seconds"] == 500.0
    assert plan.horizons == (500_000, 1_000_000, 2_000_000)
    assert all(entry["requires_explicit_reference_confirmation"] is False for entry in plan.entries)
    assert plan.entries[0]["particle_preflight"]["total_particles"] == 21517
    assert plan.entries[0]["requested_dynamic_steps"] == 500_000
    assert plan.entries[0]["execution_mode"] == "local" and plan.entries[0]["fidelity_identity"]
    assert plan.entries[0]["physics_config_fingerprint"]
    assert len({entry["physics_config_fingerprint"] for entry in plan.entries}) == 1
    assert {entry["fidelity_mode"] for entry in plan.entries} == {"SHORT_HORIZON"}
    assert ArtisticRunConfig().confirm_reference_execution is False
    with pytest.raises(ValueError, match="dump_interval_steps=1,000,000"):
        ArtisticRunConfig(dump_interval_steps=500_000)


def test_reference_execution_requires_explicit_confirmation(pinned_source: Path, tmp_path: Path) -> None:
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=pinned_source, output_root=tmp_path / "runs"))
    with pytest.raises(ValueError, match="explicit confirm_reference_execution"):
        simulator.execute(ArtisticRecipe(slurry=_slurry()), run_id="reference-confirmation-required")
    assert not (tmp_path / "runs" / "reference-confirmation-required").exists()


def test_convergence_requires_stability_and_explicit_reference_agreement() -> None:
    short = [Checkpoint(500_000, {"density": 1.0}), Checkpoint(1_000_000, {"density": 1.001}), Checkpoint(2_000_000, {"density": 1.0005})]
    assert stability_status(short, policy=_legacy_stability_policy()) == ConvergenceStatus.STABILITY_OBSERVED
    report = build_convergence_report(short, requested_steps=2_000_000)
    assert report.status == ConvergenceStatus.REFERENCE_NOT_AVAILABLE
    reference = [Checkpoint(20_000_000, {"density": 1.0})]
    compared = build_convergence_report(
        short, requested_steps=2_000_000, short_evidence=_evidence(short, FidelityMode.SHORT_HORIZON, 2_000_000),
        reference_evidence=_evidence(reference, FidelityMode.REFERENCE, 20_000_000),
        tolerance_policy=MetricTolerancePolicy(relative_tolerances={"density": 0.01}), stability_policy=_legacy_stability_policy(),
    )
    assert compared.status == ConvergenceStatus.VALIDATED_AGAINST_REFERENCE
    assert compared.available_metrics == ("density",)
    assert compared.stability_status == ConvergenceStatus.STABILITY_OBSERVED
    assert compared.reference_agreement_status == ReferenceAgreementStatus.REFERENCE_WITHIN_TOLERANCE


def test_convergence_rejects_far_or_unstable_short_runs() -> None:
    policy = MetricTolerancePolicy(default_relative_tolerance=0.01)
    stable = [Checkpoint(1, {"density": 1.0}), Checkpoint(2, {"density": 1.001}), Checkpoint(3, {"density": 1.0005})]
    far = build_convergence_report(stable, short_evidence=_evidence(stable, FidelityMode.SHORT_HORIZON, 3), reference_evidence=_evidence([Checkpoint(20_000_000, {"density": 2.0})], FidelityMode.REFERENCE, 20_000_000), tolerance_policy=policy, stability_policy=_legacy_stability_policy())
    assert far.status == ConvergenceStatus.REFERENCE_OUTSIDE_TOLERANCE
    unstable = [Checkpoint(1, {"density": 1.0}), Checkpoint(2, {"density": 1.4}), Checkpoint(3, {"density": 1.0})]
    close = build_convergence_report(unstable, short_evidence=_evidence(unstable, FidelityMode.SHORT_HORIZON, 3), reference_evidence=_evidence([Checkpoint(20_000_000, {"density": 1.0})], FidelityMode.REFERENCE, 20_000_000), tolerance_policy=policy, stability_policy=_legacy_stability_policy())
    assert close.status == ConvergenceStatus.NOT_STABLE
    assert close.reference_agreement_status == ReferenceAgreementStatus.REFERENCE_WITHIN_TOLERANCE


def test_convergence_rejects_missing_metrics_missing_short_reference_and_undefined_tolerance() -> None:
    short = [Checkpoint(1, {"density": 1.0}), Checkpoint(2, {"density": 1.0}), Checkpoint(3, {"density": 1.0})]
    missing = build_convergence_report(
        short, short_evidence=_evidence(short, FidelityMode.SHORT_HORIZON, 3), reference_evidence=_evidence([Checkpoint(20_000_000, {"density": 1.0, "pressure": 2.0})], FidelityMode.REFERENCE, 20_000_000),
        tolerance_policy=MetricTolerancePolicy(default_relative_tolerance=0.01, required_metrics=("density", "pressure")), stability_policy=_legacy_stability_policy(),
    )
    assert missing.status == ConvergenceStatus.REFERENCE_METRIC_MISSING
    assert missing.missing_metrics == ("pressure",)
    assert build_convergence_report(short, reference_checkpoints=[Checkpoint(19_000_000, {"density": 1.0})]).status == ConvergenceStatus.INCOMPATIBLE_REFERENCE_EVIDENCE
    assert build_convergence_report(short, reference_checkpoints=[Checkpoint(10, {"density": 1.0})], reference_steps=10).status == ConvergenceStatus.REFERENCE_NOT_AVAILABLE
    undefined = build_convergence_report(short, short_evidence=_evidence(short, FidelityMode.SHORT_HORIZON, 3), reference_evidence=_evidence([Checkpoint(20_000_000, {"density": 1.0})], FidelityMode.REFERENCE, 20_000_000), stability_policy=_legacy_stability_policy())
    assert undefined.status == ConvergenceStatus.REFERENCE_OUTSIDE_TOLERANCE


def test_stability_policy_reports_checkpoint_and_span_diagnostics_without_reference_claims() -> None:
    short = [Checkpoint(0, {"density": 1.0}), Checkpoint(10, {"density": 1.001}), Checkpoint(20, {"density": 1.0005})]
    assert stability_status(short) == ConvergenceStatus.INSUFFICIENT_CHECKPOINTS
    policy = StabilityPolicy(minimum_checkpoints=3, minimum_step_span=100, required_metrics=("density",))
    assert stability_status(short, policy=policy) == ConvergenceStatus.INSUFFICIENT_STEP_SPAN
    enough_span = [Checkpoint(0, {"density": 1.0}), Checkpoint(50, {"density": 1.001}), Checkpoint(100, {"density": 1.0005})]
    assert stability_status(enough_span, policy=policy) == ConvergenceStatus.STABILITY_OBSERVED
    unstable = [Checkpoint(0, {"density": 1.0}), Checkpoint(50, {"density": 1.5}), Checkpoint(100, {"density": 1.0})]
    assert stability_status(unstable, policy=policy) == ConvergenceStatus.NOT_STABLE
    report = build_convergence_report(enough_span, requested_steps=100, stability_policy=StabilityPolicy(minimum_checkpoints=3))
    assert report.stability_status == ConvergenceStatus.STABILITY_OBSERVED
    assert report.status == ConvergenceStatus.REFERENCE_NOT_AVAILABLE
    assert report.stability_policy["minimum_checkpoints"] == 3


def test_stability_span_uses_only_the_recent_window() -> None:
    checkpoints = [Checkpoint(0, {"density": 1.0}), Checkpoint(1_000_000, {"density": 1.0})]
    checkpoints.extend(Checkpoint(1_000_000 + index * 1_000, {"density": 1.0}) for index in range(10))
    policy = StabilityPolicy(minimum_checkpoints=10, minimum_step_span=100_000, required_metrics=("density",))
    assert stability_status(checkpoints, policy=policy) == ConvergenceStatus.INSUFFICIENT_STEP_SPAN


def test_convergence_uses_absolute_tolerance_for_zero_reference() -> None:
    policy = MetricTolerancePolicy(absolute_tolerances={"residual": 0.01}, required_metrics=("residual",))
    report = build_convergence_report(
        [Checkpoint(1, {"residual": 0.001}), Checkpoint(2, {"residual": 0.001}), Checkpoint(3, {"residual": 0.001})],
        short_evidence=_evidence([Checkpoint(1, {"residual": 0.001}), Checkpoint(2, {"residual": 0.001}), Checkpoint(3, {"residual": 0.001})], FidelityMode.SHORT_HORIZON, 3),
        reference_evidence=_evidence([Checkpoint(20_000_000, {"residual": 0.0})], FidelityMode.REFERENCE, 20_000_000), tolerance_policy=policy, stability_policy=_legacy_stability_policy(),
    )
    comparison = report.comparisons[0]
    assert report.status == ConvergenceStatus.VALIDATED_AGAINST_REFERENCE
    assert comparison.error_rule == "absolute" and comparison.relative_error is None and comparison.within_tolerance


def test_convergence_requires_compatible_typed_run_evidence() -> None:
    short = [Checkpoint(1, {"density": 1.0}), Checkpoint(2, {"density": 1.0}), Checkpoint(3, {"density": 1.0})]
    reference = [Checkpoint(20_000_000, {"density": 1.0})]
    report = build_convergence_report(
        short, short_evidence=_evidence(short, FidelityMode.SHORT_HORIZON, 3, recipe="short-recipe"),
        reference_evidence=_evidence(reference, FidelityMode.REFERENCE, 20_000_000, recipe="reference-recipe"),
        tolerance_policy=MetricTolerancePolicy(default_relative_tolerance=0.01), stability_policy=_legacy_stability_policy(),
    )
    assert report.status == ConvergenceStatus.INCOMPATIBLE_REFERENCE_EVIDENCE
    with pytest.raises(ValueError, match="cannot claim the reference horizon"):
        _evidence(short, FidelityMode.SHORT_HORIZON, 20_000_000)


def test_recipe_validation_and_actual_minimization_mapping() -> None:
    with pytest.raises(ValueError, match="sum to one"): SlurryRecipe(2, (5.0,) * 10, (0.4, 0.4) + (0.0,) * 8, 1.0, 0.5, 0.1, 0.9, 0.1, 0, 0.5)
    assert CalenderingRecipe(0.2, 0, False, True).template_values()["minimize"] == 1
    assert CalenderingRecipe(0.2, 0, False, False).template_values()["minimize"] == 0


def test_particle_preflight_uses_upstream_microgram_semantics_and_fails_closed(pinned_source: Path, tmp_path: Path) -> None:
    estimate = estimate_particles(_recipe())
    assert (estimate.n_am_nominal, estimate.n_am_by_type, estimate.n_am_created_total, estimate.n_cbd, estimate.total_particles) == (296, (296,), 296, 21221, 21517)
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=pinned_source, output_root=tmp_path / "runs", max_particle_count=20_000))
    with pytest.raises(ParticleCountSafetyError, match="exceeds configured safety threshold"):
        simulator.preflight(_recipe())
    with pytest.raises(ParticleCountSafetyError, match="exceeds configured safety threshold"):
        simulator.prepare(_recipe(), run_id="blocked-before-render")
    assert not (tmp_path / "runs" / "blocked-before-render").exists()
    assert ArtisticSimulator(ArtisticRunConfig(source_root=pinned_source, max_particle_count=20_000, allow_unsafe_particle_count=True)).preflight(_recipe()).total_particles == 21517


def test_lammps_round_is_nearest_not_python_bankers_or_floor() -> None:
    assert _lammps_round(1.5) == 2
    assert _lammps_round(2.5) == 3
    assert _lammps_round(2.49) == 2


def test_particle_guard_uses_sum_of_per_type_rounds(pinned_source: Path, tmp_path: Path) -> None:
    slurry = SlurryRecipe(2, (23.0, 23.0) + (5.0,) * 8, (0.5, 0.5) + (0.0,) * 8, 1.0, 0.5, 0.1, 0.9, 0.1, 0, 0.5)
    recipe = ArtisticRecipe(slurry)
    estimate = estimate_particles(recipe)
    assert estimate.n_am_nominal == 3
    assert estimate.n_am_by_type == (2, 2)
    assert estimate.n_am_created_total == 4
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=pinned_source, output_root=tmp_path / "runs", max_particle_count=estimate.n_am_nominal + estimate.n_cbd))
    with pytest.raises(ParticleCountSafetyError):
        simulator.preflight(recipe)


def test_lammps_version_skips_blank_banner_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "\nLAMMPS test\n", ""))
    assert _version("lmp") == "LAMMPS test"


def test_artistic_hashing_streams_without_path_read_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "large-output.data"
    path.write_bytes(b"x" * (5 * 1024 * 1024))
    monkeypatch.setattr(Path, "read_bytes", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hashing must stream files")))
    assert runner_module._sha256(path) == hashlib.sha256(b"x" * (5 * 1024 * 1024)).hexdigest()


def test_invoke_streams_stage_log_and_records_wall_seconds(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "emit.py").write_text("print('stage output')\n", encoding="utf-8")
    commands: list[dict[str, object]] = []
    ArtisticSimulator(ArtisticRunConfig())._invoke(workspace, "stream", "emit.py", commands, python=True)
    assert "stage output" in (workspace / "stream.log").read_text(encoding="utf-8")
    assert commands[0]["wall_seconds"] >= 0


def test_thermo_progress_parser_reports_only_printed_steps(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "slurry.log").write_text("Step Temp\n0 300\n250000 301\nLoop time of 1 on 1 procs\n", encoding="utf-8")
    parsed = parse_thermo_checkpoints(workspace)
    assert tuple(item.raw_step for item in parsed) == (0, 250000)
    assert parsed[1].metrics == {"Temp": 301.0} and parsed[1].stage == "slurry" and parsed[1].source_log == "slurry.log"


def test_thermo_progress_ignores_other_stage_logs_and_reports_malformed_rows(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "slurry.log").write_text("Step Temp Press\n0 300 1\n250000 301 1.1\n500000 nan 1.2\n750000 303\n", encoding="utf-8")
    (workspace / "drying.log").write_text("Step Temp\n9000000 999\n", encoding="utf-8")
    parsed = parse_thermo_log(workspace, stage="slurry")
    assert [item.raw_step for item in parsed.checkpoints] == [0, 250000]
    assert any("non-finite" in item for item in parsed.diagnostics)
    assert any("malformed" in item for item in parsed.diagnostics)
    progress = runner_module._progress(workspace, [{"stage": "slurry", "wall_seconds": 2.0}], 1_000_000, 250_000, None)
    assert progress["completed_steps"] == 250_000 and progress["checkpoints"][0]["metrics"] == {"Temp": 300.0, "Press": 1.0}
    assert progress["thermo_parse_diagnostics"]


def test_thermo_progress_uses_structure_for_dynamics_relative_steps(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "slurry.log").write_text(
        "Step Temp\n0 300\n50000 301\nMinimization stats:\nconverged\nStep Temp\n90000 302\n340000 303\n590000 304\n",
        encoding="utf-8",
    )
    parsed = parse_thermo_log(workspace, minimization_expected=True)
    assert [(item.raw_step, item.dynamics_step, item.phase) for item in parsed.checkpoints] == [
        (0, None, "minimization"), (50000, None, "minimization"),
        (90000, 0, "dynamics"), (340000, 250000, "dynamics"), (590000, 500000, "dynamics"),
    ]
    progress = runner_module._progress(
        workspace, [{"stage": "slurry", "wall_seconds": 2.0}], 20_000_000, 1_000_000, None,
        minimization_expected=True,
    )
    assert progress["completed_slurry_steps"] == 500_000
    assert progress["last_raw_lammps_step"] == 590_000
    assert progress["checkpoint_steps"] == [0, 250_000, 500_000]
    assert progress["steps_per_second"] is None and progress["estimated_remaining_seconds"] is None


def test_thermo_progress_handles_no_minimization_and_multiple_headers(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "slurry.log").write_text(
        "Step Temp\n0 300\nStep Temp\n250000 301\n500000 302\n", encoding="utf-8",
    )
    parsed = parse_thermo_log(workspace)
    assert [(item.raw_step, item.dynamics_step, item.phase) for item in parsed.checkpoints] == [
        (0, 0, "dynamics"), (250000, 250000, "dynamics"), (500000, 500000, "dynamics"),
    ]


def test_thermo_progress_refuses_ambiguous_minimization_boundary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "slurry.log").write_text("Step Temp\n0 300\nStep Temp\n10 301\n", encoding="utf-8")
    parsed = parse_thermo_log(workspace, minimization_expected=True)
    assert parsed.checkpoints == () and any("ambiguous" in item for item in parsed.diagnostics)


def test_thermo_progress_accepts_exact_dynamics_relative_reference_horizon(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "slurry.log").write_text(
        "Step Temp\n100 300\nMinimization stats:\nStep Temp\n90000 301\n20090000 302\n", encoding="utf-8",
    )
    parsed = parse_thermo_log(workspace, minimization_expected=True)
    assert parsed.checkpoints[-1].dynamics_step == 20_000_000


def test_convergence_evidence_can_be_loaded_from_a_run_manifest() -> None:
    manifest = {
        "run_id": "reference", "recipe_fingerprint": "recipe", "checked_out_commit": "commit",
        "source_tree_hash": "tree", "physics_config_fingerprint": "physics", "fidelity_mode": "REFERENCE",
        "fidelity_identity": "identity", "requested_slurry_steps": 20_000_000, "dump_interval_steps": 1_000_000,
        "status": "Success", "simulation_manifest_hash": "manifest",
        "checkpoints": [{"step": None, "dynamics_step": None, "raw_step": 90000, "phase": "minimization", "metrics": {}},
                        {"step": 20_000_000, "dynamics_step": 20_000_000, "raw_step": 20_090_000, "phase": "dynamics", "metrics": {"density": 1.0}}],
    }
    evidence = ConvergenceRunEvidence.from_manifest(manifest)
    assert evidence.fidelity_mode == FidelityMode.REFERENCE and evidence.checkpoints[0].raw_step == 20_090_000


def test_invoke_records_general_launch_oserror(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    commands: list[dict[str, object]] = []

    def fail_launch(*args: object, **kwargs: object) -> None:
        raise PermissionError("launch denied")

    monkeypatch.setattr(runner_module.subprocess, "Popen", fail_launch)
    with pytest.raises(PermissionError, match="launch denied"):
        ArtisticSimulator(ArtisticRunConfig())._invoke(workspace, "launch", "missing.run", commands)
    assert commands[0]["returncode"] == "environment_error"
    assert commands[0]["error_phase"] == "launch"
    assert commands[0]["error_type"] == "PermissionError"
    assert commands[0]["wall_seconds"] >= 0


def test_execute_classifies_launch_oserror_and_writes_command_manifest(pinned_source: Path, tmp_path: Path) -> None:
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=pinned_source, output_root=tmp_path / "runs", lammps_command="__missing_lammps_for_test__", confirm_reference_execution=True))
    result = simulator.execute(_recipe(), run_id="launch-error")
    manifest = json.loads((tmp_path / "runs" / "launch-error" / "manifest.json").read_text(encoding="utf-8"))
    assert result.status == SimulationStatus.ENVIRONMENT_ERROR
    assert manifest["status"] == SimulationStatus.ENVIRONMENT_ERROR.value
    assert manifest["commands"][0]["returncode"] == "environment_error"
    assert manifest["commands"][0]["error_phase"] == "launch"


def test_invoke_timeout_cleanup_is_bounded_and_diagnostic(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sleep.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    commands: list[dict[str, object]] = []
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        ArtisticSimulator(ArtisticRunConfig(timeout_seconds=1, cleanup_timeout_seconds=0.2))._invoke(workspace, "sleep", "sleep.py", commands, python=True)
    assert time.monotonic() - started < 5
    assert commands[0]["returncode"] == "timeout"
    assert commands[0]["cleanup"]["cleanup_complete"] is True


def test_mpi_validation_proves_one_distributed_job(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {"mpiexec": r"C:\mpi\mpiexec.exe", "lmp": r"C:\lammps\lmp.exe"}
    monkeypatch.setattr(config_module.shutil, "which", lambda command: paths.get(command))

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "-in" in command:
            return subprocess.CompletedProcess(command, 0, "2 by 1 by 1 MPI processor grid\n", "")
        return subprocess.CompletedProcess(command, 0, "MPI/LAMMPS test\n", "")

    monkeypatch.setattr(config_module.subprocess, "run", fake_run)
    metadata = validate_mpi_environment(ArtisticRunConfig(execution_mode=ExecutionMode.MPI, mpi_processes=2, mpi_launcher="mpiexec", lammps_command="lmp"))
    assert metadata["validated"] is True
    assert metadata["mpi_task_count"] == 2
    assert metadata["resolved_mpi_launcher"] == paths["mpiexec"]
    assert metadata["resolved_lammps_path"] == paths["lmp"]


def test_mpi_validation_fails_closed_on_grid_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {"mpiexec": r"C:\mpi\mpiexec.exe", "lmp": r"C:\lammps\lmp.exe"}
    monkeypatch.setattr(config_module.shutil, "which", lambda command: paths.get(command))
    monkeypatch.setattr(config_module.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "1 by 1 by 1 MPI processor grid\n", "") if "-in" in command else subprocess.CompletedProcess(command, 0, "version\n", ""))
    with pytest.raises(MPIEnvironmentError, match="reported 1 tasks"):
        validate_mpi_environment(ArtisticRunConfig(execution_mode=ExecutionMode.MPI, mpi_processes=2, mpi_launcher="mpiexec", lammps_command="lmp"))


def test_mpi_validation_rejects_unresolved_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module.shutil, "which", lambda command: None if command == "mpiexec" else r"C:\lammps\lmp.exe")
    with pytest.raises(MPIEnvironmentError, match="launcher does not resolve"):
        validate_mpi_environment(ArtisticRunConfig(execution_mode=ExecutionMode.MPI, mpi_processes=2, mpi_launcher="mpiexec", lammps_command="lmp"))


def test_mpi_environment_error_prevents_workspace_start(pinned_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(config: ArtisticRunConfig) -> dict[str, object]:
        raise MPIEnvironmentError("smoke unavailable", {"execution_mode": "mpi", "validated": False})

    monkeypatch.setattr(runner_module, "validate_mpi_environment", fail)
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=pinned_source, output_root=tmp_path / "runs", execution_mode=ExecutionMode.MPI, mpi_processes=2, confirm_reference_execution=True))
    result = simulator.execute(ArtisticRecipe(slurry=_slurry()), run_id="mpi-blocked")
    assert result.status == SimulationStatus.ENVIRONMENT_ERROR
    assert not (tmp_path / "runs" / "mpi-blocked" / "workspace").exists()
    manifest = json.loads((tmp_path / "runs" / "mpi-blocked" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mpi_environment"]["validated"] is False


def test_artistic_cli_runs_directly_from_repository_root() -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run([sys.executable, "scripts/run_artistic_simulator.py", "--help"], cwd=root, capture_output=True, text=True, check=False)
    assert completed.returncode == 0


def test_output_validation_rejects_nan_missing_and_particle_loss(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"; workspace.mkdir()
    (workspace / "density_slurry.out").write_text("nan", encoding="utf-8")
    (workspace / "coord_in.data").write_text(_data(10), encoding="utf-8"); (workspace / "coord_out_slurry.data").write_text(_data(9), encoding="utf-8")
    (workspace / "slurry.log").write_text("Lost atoms: original 10 current 9", encoding="utf-8")
    errors = output_errors(ArtisticRecipe(slurry=_slurry()), workspace, parse_artistic_output(workspace), 0.0)
    assert any("non-finite" in error for error in errors) and any("particle loss" in error for error in errors) and any("missing" in error for error in errors)


def test_output_validation_rejects_missing_required_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"; workspace.mkdir()
    (workspace / "coord_in.data").write_text(_data(), encoding="utf-8")
    assert any("missing expected source output: density_slurry.out" in error for error in output_errors(ArtisticRecipe(slurry=_slurry()), workspace, parse_artistic_output(workspace), 0.0))


def test_final_thickness_uses_particle_extent_not_fixed_simulation_box(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"; workspace.mkdir()
    (workspace / "coord_out_cal.data").write_text(_data(z=12), encoding="utf-8")
    (workspace / "Cal_electrode.atom").write_text(_dump(2.0, 6.0), encoding="utf-8")
    first = parse_artistic_output(workspace)
    (workspace / "Cal_electrode.atom").write_text(_dump(2.0, 9.0), encoding="utf-8")
    second = parse_artistic_output(workspace)
    assert first.diagnostics["simulation_box_z_length"] == second.diagnostics["simulation_box_z_length"] == 12.0
    assert first.final_kpis["calendered_electrode_thickness"] == 6.0
    assert second.final_kpis["calendered_electrode_thickness"] == 9.0


def test_validation_uses_file_size_for_large_data_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"; workspace.mkdir()
    (workspace / "density_slurry.out").write_text("1", encoding="utf-8")
    (workspace / "coord_out_slurry.data").write_text(_data(), encoding="utf-8")
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("validation must not read data files")) if self.name == "coord_out_slurry.data" else original(self, *args, **kwargs))
    parsed = ParsedArtisticOutput({"slurry": {"slurry_density": 1.0}}, {}, {}, 10, 10, None)
    assert output_errors(ArtisticRecipe(slurry=_slurry()), workspace, parsed, 0.0) == ()


def test_runner_orders_postprocessing_and_records_lineage(pinned_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=pinned_source, output_root=tmp_path / "runs", confirm_reference_execution=True)); calls: list[str] = []
    def invoke(workspace: Path, stage: str, input_file: str, commands: list[dict[str, object]], *, python: bool = False) -> None:
        calls.append(stage)
        if stage == "slurry":
            for name, value in {"coord_in.data": _data(), "coord_out_slurry.data": _data(), "density_slurry.out": "1.0"}.items(): (workspace / name).write_text(value, encoding="utf-8")
        elif stage == "drying_homogeneous":
            (workspace / "coord_out_electrode.data").write_text(_data(), encoding="utf-8"); (workspace / "AM_loading.out").write_text("2.0", encoding="utf-8")
        elif stage == "drying_porosity":
            for name, value in {"porosity_bulk.out": "10", "porosity_all.out": "20", "check.txt": "ok"}.items(): (workspace / name).write_text(value, encoding="utf-8")
        elif stage == "calendering":
            for name, value in {"coord_out_cal.data": _data(z=8), "Cal_electrode.atom": _dump(3, 7, zhi=8), "new_CBD_nanoporosity": "0.4", "initial_lz": "12"}.items(): (workspace / name).write_text(value, encoding="utf-8")
        elif stage == "calendering_porosity":
            for name, value in {"porosity_cal_bulk.out": "5", "porosity_cal_all.out": "15", "check_cal.txt": "ok"}.items(): (workspace / name).write_text(value, encoding="utf-8")
        commands.append({"stage": stage, "command": [input_file], "returncode": 0})
    monkeypatch.setattr(simulator, "_invoke", invoke)
    result = simulator.execute(_recipe(), run_id="full")
    assert result.status == SimulationStatus.SUCCESS
    assert calls == ["slurry", "drying_homogeneous", "drying_porosity", "calendering_reformat", "calendering", "calendering_porosity"]
    assert [entry["boundary"] for entry in result.provenance["stage_lineage"]] == ["slurry_output", "slurry_to_drying", "drying_to_calendering"]
    assert "workspace/coord_out_cal.data" in result.provenance["output_hashes"]
    assert result.final_outputs["calendered_electrode_thickness"] == 7.0


@pytest.mark.parametrize(("exception", "status"), [(subprocess.TimeoutExpired(["lmp"], 1), SimulationStatus.TIMEOUT), (subprocess.CalledProcessError(1, ["lmp"]), SimulationStatus.NUMERICAL_FAILURE)])
def test_runner_never_promotes_timeout_or_numerical_failure(pinned_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exception: Exception, status: SimulationStatus) -> None:
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=pinned_source, output_root=tmp_path / "runs", confirm_reference_execution=True))
    def fail(*args: object, **kwargs: object) -> None: raise exception
    monkeypatch.setattr(simulator, "_invoke", fail)
    assert simulator.execute(ArtisticRecipe(slurry=_slurry()), run_id=f"{status}").status == status


def test_adapter_normalizes_only_pinned_success_without_target_alias(tmp_path: Path) -> None:
    recipe = _recipe()
    result = _successful_result(tmp_path, "success", recipe)
    run = ArtisticSimulationAdapter.from_simulation(result, recipe)
    assert not (set(run.final_kpis) & {name for stage in run.stages for name in stage.intermediate_properties})
    view = InformationHorizon(ProcessStage.CALENDERING).project(run)
    assert not (set(run.final_kpis) & set(view.intermediate_properties))
    cache = ArtisticSimulationAdapter.normalize_successful(result, recipe, root=tmp_path / "cache")
    assert cache.is_file() and ArtisticSimulationAdapter(tmp_path / "cache").load_runs()[0].provenance.evidence_kind == "SIMULATED_PHYSICS"
    assert ArtisticSimulationAdapter().metadata().multimodal_capable is False
    failed = SimulationResult(SimulationStatus.TIMEOUT, "failed", tmp_path / "failed")
    with pytest.raises(ValueError, match="not valid simulated physics"): ArtisticSimulationAdapter.normalize_successful(failed, recipe, root=tmp_path / "cache")


def test_adapter_requires_exact_reference_dynamics_horizon(tmp_path: Path) -> None:
    recipe = _recipe()
    result = _successful_result(tmp_path, "short-reference", recipe)
    result.provenance["completed_slurry_steps"] = 19_000_000
    with pytest.raises(ValueError, match="exactly 20,000,000 dynamics-relative"):
        ArtisticSimulationAdapter.from_simulation(result, recipe)


def test_short_horizon_cache_preserves_lower_fidelity_identity_and_evidence_kind(tmp_path: Path) -> None:
    recipe = _recipe()
    result = _successful_result(tmp_path, "short", recipe)
    result.provenance.update({"fidelity_mode": "SHORT_HORIZON", "requested_slurry_steps": 500_000, "completed_slurry_steps": 500_000, "fidelity_identity": "short-id", "reference_equivalence_status": "REFERENCE_NOT_AVAILABLE"})
    manifest_path = result.run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({key: result.provenance[key] for key in ("fidelity_mode", "requested_slurry_steps", "completed_slurry_steps", "fidelity_identity", "reference_equivalence_status")})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache = ArtisticSimulationAdapter.normalize_successful(result, recipe, root=tmp_path / "cache")
    run = ArtisticSimulationAdapter(tmp_path / "cache").load_runs()[0]
    aggregate = json.loads((tmp_path / "cache" / "manifest.json").read_text(encoding="utf-8"))
    assert run.provenance.evidence_kind == "SIMULATED_STRESS"
    assert aggregate["normalized_runs"][0]["fidelity_identity"] == "short-id"
    assert aggregate["normalized_runs"][0]["cache_identity"]
    assert cache.is_file()


def test_normalization_keeps_each_manifest_and_groups_repeated_recipes(tmp_path: Path) -> None:
    recipe = _recipe(); cache_root = tmp_path / "cache"
    ArtisticSimulationAdapter.normalize_successful(_successful_result(tmp_path, "repeat-one", recipe), recipe, root=cache_root)
    ArtisticSimulationAdapter.normalize_successful(_successful_result(tmp_path, "repeat-two", recipe), recipe, root=cache_root)
    runs = ArtisticSimulationAdapter(cache_root).load_runs()
    assert {run.run_id for run in runs} == {"repeat-one", "repeat-two"}
    assert len({run.batch_id for run in runs}) == 1
    aggregate = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    assert len(aggregate["normalized_runs"]) == 2
    for entry in aggregate["normalized_runs"]:
        assert (cache_root / entry["simulation_manifest"]).is_file()
        assert entry["recipe_fingerprint"] == runs[0].batch_id
        assert entry["physics_output_hashes"] and entry["executable_versions"]


def test_normalization_publishes_nothing_when_cache_write_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = _recipe()
    result = _successful_result(tmp_path, "failed-normalize", recipe)
    monkeypatch.setattr(ArtisticSimulationAdapter, "write_processed_cache", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected cache failure")))
    with pytest.raises(OSError, match="injected cache failure"):
        ArtisticSimulationAdapter.normalize_successful(result, recipe, root=tmp_path / "cache")
    assert not (tmp_path / "cache").exists()
    assert not list(tmp_path.glob(".cache.txn-*"))
