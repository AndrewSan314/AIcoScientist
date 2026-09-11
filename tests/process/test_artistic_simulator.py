from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import src.process.simulators.artistic.config as config_module
import src.process.simulators.artistic.provenance as provenance_module
from src.datasets.battery_process.artistic import ArtisticSimulationAdapter
from src.process.simulators.base import SimulationResult, SimulationStatus
from src.process.simulators.artistic import ArtisticRecipe, ArtisticRunConfig, ArtisticSimulator, CalenderingRecipe, DryingMode, SlurryRecipe
from src.process.simulators.artistic.config import PINNED_COMMIT, PINNED_SOURCE_TREE_HASH, SourcePinError
from src.process.simulators.artistic.parser import parse_artistic_output
from src.process.simulators.artistic.validation import output_errors
from src.process.simulators.artistic.runner import _version
from src.process.information_horizon import InformationHorizon
from src.process.stages import ProcessStage


def _slurry() -> SlurryRecipe:
    return SlurryRecipe(1, (5.0,) * 10, (1.0,) + (0.0,) * 9, 0.1, 0.5, 1.0, 0.9, 0.1, 0, 0.5)


def _recipe() -> ArtisticRecipe:
    return ArtisticRecipe(_slurry(), DryingMode.HOMOGENEOUS, None, CalenderingRecipe(0.25, 0.05, True, False))


def _data(atoms: int = 10, z: float = 12.0) -> str:
    return f"LAMMPS data\n\n{atoms} atoms\n\n0 10 xlo xhi\n0 10 ylo yhi\n0 {z} zlo zhi\n"


@pytest.fixture
def pinned_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source"; root = source / "NMC" / "Updated version"; slurry = root / "Slurry"; slurry.mkdir(parents=True)
    (slurry / "user_inputs.txt").write_text("variable nAM_part equal @nAM_part@\n", encoding="utf-8")
    (slurry / "init_structure.txt").write_text("\n".join(f"variable n_AM{i} equal round(v_n_AM*v_p_AM6)" for i in range(7, 11)), encoding="utf-8")
    (slurry / "in_slurry.run").write_text("# source fixture\n", encoding="utf-8")
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


def test_recipe_validation_and_actual_minimization_mapping() -> None:
    with pytest.raises(ValueError, match="sum to one"): SlurryRecipe(2, (5.0,) * 10, (0.4, 0.4) + (0.0,) * 8, 0.1, 0.5, 1.0, 0.9, 0.1, 0, 0.5)
    assert CalenderingRecipe(0.2, 0, False, True).template_values()["minimize"] == 1
    assert CalenderingRecipe(0.2, 0, False, False).template_values()["minimize"] == 0


def test_lammps_version_skips_blank_banner_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "\nLAMMPS test\n", ""))
    assert _version("lmp") == "LAMMPS test"


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


def test_runner_orders_postprocessing_and_records_lineage(pinned_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=pinned_source, output_root=tmp_path / "runs")); calls: list[str] = []
    def invoke(workspace: Path, stage: str, input_file: str, commands: list[dict[str, object]], *, python: bool = False) -> None:
        calls.append(stage)
        if stage == "slurry":
            for name, value in {"coord_in.data": _data(), "coord_out_slurry.data": _data(), "density_slurry.out": "1.0"}.items(): (workspace / name).write_text(value, encoding="utf-8")
        elif stage == "drying_homogeneous":
            (workspace / "coord_out_electrode.data").write_text(_data(), encoding="utf-8"); (workspace / "AM_loading.out").write_text("2.0", encoding="utf-8")
        elif stage == "drying_porosity":
            for name, value in {"porosity_bulk.out": "10", "porosity_all.out": "20", "check.txt": "ok"}.items(): (workspace / name).write_text(value, encoding="utf-8")
        elif stage == "calendering":
            for name, value in {"coord_out_cal.data": _data(z=8), "new_CBD_nanoporosity": "0.4", "initial_lz": "12"}.items(): (workspace / name).write_text(value, encoding="utf-8")
        elif stage == "calendering_porosity":
            for name, value in {"porosity_cal_bulk.out": "5", "porosity_cal_all.out": "15", "check_cal.txt": "ok"}.items(): (workspace / name).write_text(value, encoding="utf-8")
        commands.append({"stage": stage, "command": [input_file], "returncode": 0})
    monkeypatch.setattr(simulator, "_invoke", invoke)
    result = simulator.execute(_recipe(), run_id="full")
    assert result.status == SimulationStatus.SUCCESS
    assert calls == ["slurry", "drying_homogeneous", "drying_porosity", "calendering_reformat", "calendering", "calendering_porosity"]
    assert [entry["boundary"] for entry in result.provenance["stage_lineage"]] == ["slurry_output", "slurry_to_drying", "drying_to_calendering"]
    assert "workspace/coord_out_cal.data" in result.provenance["output_hashes"]


@pytest.mark.parametrize(("exception", "status"), [(subprocess.TimeoutExpired(["lmp"], 1), SimulationStatus.TIMEOUT), (subprocess.CalledProcessError(1, ["lmp"]), SimulationStatus.NUMERICAL_FAILURE)])
def test_runner_never_promotes_timeout_or_numerical_failure(pinned_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exception: Exception, status: SimulationStatus) -> None:
    simulator = ArtisticSimulator(ArtisticRunConfig(source_root=pinned_source, output_root=tmp_path / "runs"))
    def fail(*args: object, **kwargs: object) -> None: raise exception
    monkeypatch.setattr(simulator, "_invoke", fail)
    assert simulator.execute(ArtisticRecipe(slurry=_slurry()), run_id=f"{status}").status == status


def test_adapter_normalizes_only_pinned_success_without_target_alias(tmp_path: Path) -> None:
    recipe = _recipe()
    result = SimulationResult(SimulationStatus.SUCCESS, "success", tmp_path / "run", {"slurry": {"slurry_density": 1.0}, "drying": {"am_loading": 2.0, "drying_porosity_bulk_percent": 10.0, "drying_porosity_all_percent": 20.0}}, {"calendered_electrode_thickness": 8.0, "calendered_cbd_nanoporosity": 0.4, "calendered_porosity_bulk_percent": 5.0, "calendered_porosity_all_percent": 15.0}, {"checked_out_commit": PINNED_COMMIT, "source_tree_hash": PINNED_SOURCE_TREE_HASH, "rendered_source_file_hashes": {"source": "digest"}, "patches": []})
    run = ArtisticSimulationAdapter.from_simulation(result, recipe)
    assert not (set(run.final_kpis) & {name for stage in run.stages for name in stage.intermediate_properties})
    view = InformationHorizon(ProcessStage.CALENDERING).project(run)
    assert not (set(run.final_kpis) & set(view.intermediate_properties))
    cache = ArtisticSimulationAdapter.normalize_successful(result, recipe, root=tmp_path / "cache")
    assert cache.is_file() and ArtisticSimulationAdapter(tmp_path / "cache").load_runs()[0].provenance.evidence_kind == "SIMULATED_PHYSICS"
    assert ArtisticSimulationAdapter().metadata().multimodal_capable is False
    failed = SimulationResult(SimulationStatus.TIMEOUT, "failed", tmp_path / "failed")
    with pytest.raises(ValueError, match="not valid simulated physics"): ArtisticSimulationAdapter.normalize_successful(failed, recipe, root=tmp_path / "cache")
