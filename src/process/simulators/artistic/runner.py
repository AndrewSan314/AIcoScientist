from __future__ import annotations

import hashlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.process.simulators.base import SimulationResult, SimulationStatus

from .config import ArtisticRunConfig, ExecutionMode
from .parser import parse_artistic_output
from .provenance import source_provenance, write_json
from .renderer import ArtisticRenderer, RenderState
from .schemas import ArtisticRecipe, DryingMode, ParticleCountSafetyError, ParticleEstimate, estimate_particles, recipe_fingerprint
from .validation import output_errors


class ArtisticSimulator:
    """Portable execution bridge for the pinned ARTISTIC source tree."""

    def __init__(self, config: ArtisticRunConfig = ArtisticRunConfig()) -> None:
        self.config = config
        self.renderer = ArtisticRenderer(config)

    def prepare(self, recipe: ArtisticRecipe, *, run_id: str | None = None) -> Path:
        estimate = self.preflight(recipe)
        state = self.renderer.prepare_workspace(run_id or uuid.uuid4().hex)
        self.renderer.stage(state, "slurry", recipe.template_values)
        if recipe.drying_mode == DryingMode.HOMOGENEOUS:
            self.renderer.stage(state, "drying_homogeneous", recipe.template_values)
        elif recipe.drying_mode == DryingMode.HETEROGENEOUS:
            self.renderer.stage(state, "drying_heterogeneous", recipe.template_values)
        if recipe.calendering:
            self.renderer.stage(state, "calendering", recipe.template_values)
        self._write_manifest(state, recipe, particle_estimate=estimate, commands=[], status=None, diagnostics=())
        return state.workspace.parent

    def execute(self, recipe: ArtisticRecipe, *, run_id: str | None = None) -> SimulationResult:
        run_id = run_id or uuid.uuid4().hex
        self.config.verify_source_pin()
        estimate = self.preflight(recipe)
        state: RenderState | None = None
        commands: list[dict[str, object]] = []
        lineage: list[dict[str, object]] = []
        try:
            state = self.renderer.prepare_workspace(run_id)
            self.renderer.stage(state, "slurry", recipe.template_values)
            self._invoke(state.workspace, "slurry", "in_slurry.run", commands)
            slurry_outputs = _hashes(state.workspace, "coord_out_slurry.data", "density_slurry.out")
            lineage.append({"boundary": "slurry_output", "output_hashes": slurry_outputs})
            if recipe.drying_mode == DryingMode.HOMOGENEOUS:
                self.renderer.stage(state, "drying_homogeneous", recipe.template_values)
                _assert_hashes(state.workspace, slurry_outputs)
                lineage.append({"boundary": "slurry_to_drying", "upstream_output_hashes": slurry_outputs, "downstream_input_hashes": _hashes(state.workspace, *slurry_outputs)})
                self._invoke(state.workspace, "drying_homogeneous", "in_evap_hom.run", commands)
                self._invoke(state.workspace, "drying_porosity", "pores.py", commands, python=True)
            elif recipe.drying_mode == DryingMode.HETEROGENEOUS:
                self.renderer.stage(state, "drying_heterogeneous", recipe.template_values)
                _assert_hashes(state.workspace, slurry_outputs)
                lineage.append({"boundary": "slurry_to_cbd_generation", "upstream_output_hashes": slurry_outputs, "downstream_input_hashes": _hashes(state.workspace, *slurry_outputs)})
                self._invoke(state.workspace, "drying_cbd", "CBDs.txt", commands)
                cbd_outputs = _hashes(state.workspace, "coord_out_slurry_CBDs.data")
                lineage.append({"boundary": "cbd_generation_output", "output_hashes": cbd_outputs})
                _assert_hashes(state.workspace, cbd_outputs)
                lineage.append({"boundary": "cbd_generation_to_drying", "upstream_output_hashes": cbd_outputs, "downstream_input_hashes": _hashes(state.workspace, *cbd_outputs)})
                self._invoke(state.workspace, "drying_heterogeneous", "in_evaporation_freeze.run", commands)
                self._invoke(state.workspace, "drying_porosity", "pores.py", commands, python=True)
            if recipe.calendering:
                self.renderer.stage(state, "calendering", recipe.template_values)
                drying_outputs = _hashes(state.workspace, "coord_out_electrode.data", "AM_loading.out", "porosity_bulk.out", "porosity_all.out")
                _assert_hashes(state.workspace, {"coord_out_electrode.data": drying_outputs["coord_out_electrode.data"]})
                lineage.append({"boundary": "drying_to_calendering", "upstream_output_hashes": drying_outputs, "downstream_input_hashes": _hashes(state.workspace, "coord_out_electrode.data")})
                self._invoke(state.workspace, "calendering_reformat", "Reformatting_cal_electrode.py", commands, python=True)
                self._invoke(state.workspace, "calendering", "in_cal.run", commands)
                self._invoke(state.workspace, "calendering_porosity", "pores_cal.py", commands, python=True)
            parsed = parse_artistic_output(state.workspace)
            errors = output_errors(recipe, state.workspace, parsed, self.config.lost_particle_tolerance)
            status = SimulationStatus.INVALID_PHYSICS_RUN if any("particle loss" in error for error in errors) else SimulationStatus.NUMERICAL_FAILURE if errors else SimulationStatus.SUCCESS
            result = SimulationResult(status, run_id, state.workspace.parent, parsed.stages, parsed.final_kpis, diagnostics=errors)
        except FileNotFoundError as exc:
            result = SimulationResult(SimulationStatus.ENVIRONMENT_ERROR, run_id, state.workspace.parent if state else self.config.output_root / run_id, diagnostics=(str(exc),))
        except subprocess.TimeoutExpired as exc:
            result = SimulationResult(SimulationStatus.TIMEOUT, run_id, state.workspace.parent if state else self.config.output_root / run_id, diagnostics=(f"command timed out: {exc.cmd}",))
        except subprocess.CalledProcessError as exc:
            result = SimulationResult(SimulationStatus.NUMERICAL_FAILURE, run_id, state.workspace.parent if state else self.config.output_root / run_id, diagnostics=(f"command exited {exc.returncode}: {exc.cmd}",))
        except RuntimeError as exc:
            result = SimulationResult(SimulationStatus.NUMERICAL_FAILURE, run_id, state.workspace.parent if state else self.config.output_root / run_id, diagnostics=(str(exc),))
        provenance = self._write_manifest(state, recipe, run_id=run_id, particle_estimate=estimate, commands=commands, lineage=lineage, status=result.status, diagnostics=result.diagnostics)
        return SimulationResult(result.status, result.run_id, result.run_directory, result.stage_outputs, result.final_outputs, provenance, result.diagnostics)

    def preflight(self, recipe: ArtisticRecipe) -> ParticleEstimate:
        estimate = estimate_particles(recipe)
        if estimate.total_particles > self.config.max_particle_count and not self.config.allow_unsafe_particle_count:
            raise ParticleCountSafetyError(
                f"predicted ARTISTIC particle count {estimate.total_particles:,} exceeds configured safety threshold "
                f"{self.config.max_particle_count:,}; rerun only with explicit allow_unsafe_particle_count"
            )
        return estimate

    def _invoke(self, workspace: Path, stage: str, input_file: str, commands: list[dict[str, object]], *, python: bool = False) -> None:
        command = [sys.executable, input_file] if python else self._lammps_command(input_file)
        if self.config.execution_mode == ExecutionMode.SLURM:
            command = self._slurm_command(workspace, stage, command)
        started = datetime.now(UTC)
        try:
            process = subprocess.Popen(command, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            commands.append({"stage": stage, "command": command, "started_at": started.isoformat(), "ended_at": datetime.now(UTC).isoformat(), "returncode": "environment_error"})
            raise
        try:
            stdout, stderr = process.communicate(timeout=self.config.timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            stdout, stderr = process.communicate()
            log = workspace / f"{stage}.log"
            log.write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")
            commands.append({"stage": stage, "command": command, "started_at": started.isoformat(), "ended_at": datetime.now(UTC).isoformat(), "returncode": "timeout", "log_sha256": _sha256(log)})
            raise subprocess.TimeoutExpired(command, self.config.timeout_seconds, output=stdout, stderr=stderr)
        completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        log = workspace / f"{stage}.log"
        log.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
        commands.append({"stage": stage, "command": command, "started_at": started.isoformat(), "ended_at": datetime.now(UTC).isoformat(), "returncode": completed.returncode, "log_sha256": _sha256(log)})
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)

    def _lammps_command(self, input_file: str) -> list[str]:
        command = [self.config.lammps_command, "-in", input_file]
        if self.config.execution_mode == ExecutionMode.MPI:
            command = [self.config.mpi_launcher, "-n", str(self.config.mpi_processes), *command]
        return command

    def _slurm_command(self, workspace: Path, stage: str, command: list[str]) -> list[str]:
        script = workspace / f"{stage}.slurm.sh"
        script.write_text("#!/bin/sh\nset -eu\n" + " ".join(_shell_quote(part) for part in command) + "\n", encoding="utf-8", newline="\n")
        return [self.config.slurm_submit, "--wait", str(script)]

    def _write_manifest(self, state: RenderState | None, recipe: ArtisticRecipe, *, run_id: str | None = None, particle_estimate: ParticleEstimate, commands: list[dict[str, object]], lineage: list[dict[str, object]] | None = None, status: SimulationStatus | None, diagnostics: tuple[str, ...]) -> dict[str, object]:
        run_directory = state.workspace.parent if state else self.config.output_root / (run_id or "unprepared")
        run_directory.mkdir(parents=True, exist_ok=True)
        source = source_provenance(self.config.source_root)
        outputs = {str(path.relative_to(run_directory)).replace("\\", "/"): _sha256(path) for path in run_directory.rglob("*") if path.is_file() and path.name != "manifest.json"}
        payload: dict[str, object] = {
            **source, "license": "CC BY-NC-SA 4.0", "ai_co_scientist_commit": _git_head(),
            "recipe": _recipe_dict(recipe), "recipe_fingerprint": recipe_fingerprint(recipe), "particle_preflight": particle_estimate.as_dict(), "status": str(status) if status else "PREPARED_NOT_EXECUTED",
            "commands": commands, "executable_versions": {"python": sys.version, "numpy": _package_version("numpy"), "numba": _package_version("numba"), "lammps": _version(self.config.lammps_command)},
            "executable_identity": {"lammps_command": self.config.lammps_command, "lammps_path": shutil.which(self.config.lammps_command), "python_executable": sys.executable},
            "rendered_source_file_hashes": state.source_hashes if state else {}, "patches": state.patches if state else [],
            "output_hashes": outputs, "stage_lineage": lineage or [], "diagnostics": list(diagnostics),
        }
        write_json(run_directory / "manifest.json", payload)
        return payload


def _version(command: str) -> str:
    try:
        result = subprocess.run([command, "-h"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    lines = [line.strip() for line in (result.stdout or result.stderr).splitlines() if line.strip()]
    return lines[0] if lines else f"exit {result.returncode}"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _recipe_dict(recipe: ArtisticRecipe) -> dict[str, object]:
    return {"slurry": recipe.slurry.__dict__, "drying_mode": recipe.drying_mode, "heterogeneous_drying": recipe.heterogeneous_drying.__dict__ if recipe.heterogeneous_drying else None, "calendering": recipe.calendering.__dict__ if recipe.calendering else None}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shell_quote(part: str) -> str:
    return "'" + part.replace("'", "'\"'\"'") + "'"


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        process.kill()


def _hashes(workspace: Path, *names: str) -> dict[str, str]:
    missing = [name for name in names if not (workspace / name).is_file()]
    if missing:
        raise RuntimeError(f"missing stage-boundary output: {', '.join(missing)}")
    return {name: _sha256(workspace / name) for name in names}


def _assert_hashes(workspace: Path, expected: dict[str, str]) -> None:
    actual = _hashes(workspace, *expected)
    mismatched = [name for name, digest in expected.items() if actual[name] != digest]
    if mismatched:
        raise RuntimeError(f"stage-lineage hash mismatch: {', '.join(mismatched)}")
