from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.process.simulators.base import SimulationResult, SimulationStatus

from .config import ArtisticRunConfig, ExecutionMode, FidelityMode, MPIEnvironmentError, physics_config_fingerprint, validate_mpi_environment
from .parser import parse_artistic_output, parse_thermo_log
from .provenance import sha256 as _streaming_sha256
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
        if self.config.fidelity_mode == FidelityMode.REFERENCE and not self.config.confirm_reference_execution:
            raise ValueError("reference ARTISTIC execution requires explicit confirm_reference_execution=True")
        self.config.verify_source_pin()
        estimate = self.preflight(recipe)
        state: RenderState | None = None
        commands: list[dict[str, object]] = []
        lineage: list[dict[str, object]] = []
        mpi_environment = _initial_mpi_metadata(self.config)
        try:
            if self.config.execution_mode == ExecutionMode.MPI:
                mpi_environment = validate_mpi_environment(self.config)
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
        except MPIEnvironmentError as exc:
            mpi_environment = {**mpi_environment, **exc.metadata, "validation_error": str(exc), "validated": False}
            result = SimulationResult(SimulationStatus.ENVIRONMENT_ERROR, run_id, state.workspace.parent if state else self.config.output_root / run_id, diagnostics=(str(exc),))
        except OSError as exc:
            result = SimulationResult(SimulationStatus.ENVIRONMENT_ERROR, run_id, state.workspace.parent if state else self.config.output_root / run_id, diagnostics=(str(exc),))
        except subprocess.TimeoutExpired as exc:
            result = SimulationResult(SimulationStatus.TIMEOUT, run_id, state.workspace.parent if state else self.config.output_root / run_id, diagnostics=(f"command timed out: {exc.cmd}",))
        except subprocess.CalledProcessError as exc:
            result = SimulationResult(SimulationStatus.NUMERICAL_FAILURE, run_id, state.workspace.parent if state else self.config.output_root / run_id, diagnostics=(f"command exited {exc.returncode}: {exc.cmd}",))
        except RuntimeError as exc:
            result = SimulationResult(SimulationStatus.NUMERICAL_FAILURE, run_id, state.workspace.parent if state else self.config.output_root / run_id, diagnostics=(str(exc),))
        provenance = self._write_manifest(state, recipe, run_id=run_id, particle_estimate=estimate, commands=commands, lineage=lineage, status=result.status, diagnostics=result.diagnostics, mpi_environment=mpi_environment)
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
        started_monotonic = time.monotonic()
        log = workspace / f"{stage}.log"
        process_options = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        process: subprocess.Popen[str] | None = None
        try:
            with log.open("w", encoding="utf-8", newline="\n") as stream:
                process = subprocess.Popen(command, cwd=workspace, stdout=stream, stderr=subprocess.STDOUT, text=True, **process_options)
                try:
                    process.wait(timeout=self.config.timeout_seconds)
                except subprocess.TimeoutExpired:
                    cleanup = _cleanup_process(process, self.config.cleanup_timeout_seconds)
                    commands.append({"stage": stage, "command": command, "started_at": started.isoformat(), "ended_at": datetime.now(UTC).isoformat(), "wall_seconds": time.monotonic() - started_monotonic, "returncode": "timeout", "cleanup": cleanup, "log_sha256": _sha256(log)})
                    raise subprocess.TimeoutExpired(command, self.config.timeout_seconds, output=_tail(log))
        except OSError as exc:
            commands.append({"stage": stage, "command": command, "started_at": started.isoformat(), "ended_at": datetime.now(UTC).isoformat(), "wall_seconds": time.monotonic() - started_monotonic, "returncode": "environment_error", "error_phase": "launch" if process is None else "wait", "error_type": type(exc).__name__, "error": str(exc), "log_sha256": _sha256(log) if log.is_file() else None})
            raise
        commands.append({"stage": stage, "command": command, "started_at": started.isoformat(), "ended_at": datetime.now(UTC).isoformat(), "wall_seconds": time.monotonic() - started_monotonic, "returncode": process.returncode, "log_sha256": _sha256(log)})
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output=_tail(log))

    def _lammps_command(self, input_file: str) -> list[str]:
        command = [self.config.lammps_command, "-in", input_file]
        if self.config.execution_mode == ExecutionMode.MPI:
            command = [self.config.mpi_launcher, "-n", str(self.config.mpi_processes), *command]
        return command

    def _slurm_command(self, workspace: Path, stage: str, command: list[str]) -> list[str]:
        script = workspace / f"{stage}.slurm.sh"
        script.write_text("#!/bin/sh\nset -eu\n" + " ".join(_shell_quote(part) for part in command) + "\n", encoding="utf-8", newline="\n")
        return [self.config.slurm_submit, "--wait", str(script)]

    def _write_manifest(self, state: RenderState | None, recipe: ArtisticRecipe, *, run_id: str | None = None, particle_estimate: ParticleEstimate, commands: list[dict[str, object]], lineage: list[dict[str, object]] | None = None, status: SimulationStatus | None, diagnostics: tuple[str, ...], mpi_environment: dict[str, object] | None = None) -> dict[str, object]:
        run_directory = state.workspace.parent if state else self.config.output_root / (run_id or "unprepared")
        run_directory.mkdir(parents=True, exist_ok=True)
        source = source_provenance(self.config.source_root)
        outputs = {str(path.relative_to(run_directory)).replace("\\", "/"): _sha256(path) for path in run_directory.rglob("*") if path.is_file() and path.name != "manifest.json"}
        payload: dict[str, object] = {
            **source, "run_id": run_id or run_directory.name, "license": "CC BY-NC-SA 4.0", "ai_co_scientist_commit": _git_head(),
            "recipe": _recipe_dict(recipe), "recipe_fingerprint": recipe_fingerprint(recipe), "particle_preflight": particle_estimate.as_dict(), "status": str(status) if status else "PREPARED_NOT_EXECUTED",
            "commands": commands, "executable_versions": {"python": sys.version, "numpy": _package_version("numpy"), "numba": _package_version("numba"), "lammps": _version(self.config.lammps_command)},
            "executable_identity": {"lammps_command": self.config.lammps_command, "lammps_path": shutil.which(self.config.lammps_command), "python_executable": sys.executable},
            "mpi_environment": mpi_environment or _initial_mpi_metadata(self.config),
            "rendered_source_file_hashes": state.source_hashes if state else {}, "patches": state.patches if state else [],
            "physics_config_fingerprint": physics_config_fingerprint(
                recipe_fingerprint=recipe_fingerprint(recipe),
                source_commit=str(source.get("checked_out_commit", "")),
                source_tree_hash=str(source.get("source_tree_hash", "")),
                patches=state.patches if state else (),
            ),
            "output_hashes": outputs, "stage_lineage": lineage or [], "diagnostics": list(diagnostics),
        }
        progress = _progress(
            state.workspace if state else None, commands, self.config.requested_slurry_steps,
            self.config.dump_interval_steps, status,
            minimization_expected=_slurry_minimization_expected(state.workspace) if state else None,
        )
        payload.update({
            "fidelity_mode": self.config.fidelity_mode.value,
            "reference_slurry_steps": 20_000_000,
            "requested_slurry_steps": self.config.requested_slurry_steps,
            "completed_slurry_steps": progress["completed_slurry_steps"],
            "dump_interval_steps": self.config.dump_interval_steps,
            "fidelity_identity": self.config.fidelity_identity,
            "early_termination": progress["early_termination"],
            "progress": progress,
            "checkpoints": progress["checkpoints"],
            "reference_equivalence_status": "REFERENCE_NOT_AVAILABLE" if self.config.fidelity_mode == FidelityMode.SHORT_HORIZON else "NOT_EVALUATED",
        })
        payload["simulation_manifest_hash"] = _manifest_hash(payload)
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


def _initial_mpi_metadata(config: ArtisticRunConfig) -> dict[str, object]:
    launcher = shutil.which(config.mpi_launcher)
    return {
        "execution_mode": config.execution_mode.value,
        "requested_mpi_processes": config.mpi_processes,
        "mpi_launcher": config.mpi_launcher,
        "mpi_launcher_command": [config.mpi_launcher, "-n", str(config.mpi_processes)],
        "resolved_mpi_launcher": launcher,
        "resolved_lammps_path": shutil.which(config.lammps_command),
        "validated": config.execution_mode != ExecutionMode.MPI,
    }


def _recipe_dict(recipe: ArtisticRecipe) -> dict[str, object]:
    return {"slurry": recipe.slurry.__dict__, "drying_mode": recipe.drying_mode, "heterogeneous_drying": recipe.heterogeneous_drying.__dict__ if recipe.heterogeneous_drying else None, "calendering": recipe.calendering.__dict__ if recipe.calendering else None}


def _sha256(path: Path) -> str:
    return _streaming_sha256(path)


def _manifest_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _shell_quote(part: str) -> str:
    return "'" + part.replace("'", "'\"'\"'") + "'"


def _terminate_process_tree(process: subprocess.Popen[str], *, hard: bool, timeout_seconds: float) -> dict[str, object]:
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
            return {"method": "taskkill", "hard": hard, "returncode": completed.returncode}
        except (OSError, subprocess.TimeoutExpired) as exc:
            try:
                process.kill()
                return {"method": "process.kill", "hard": True, "fallback": True, "taskkill_error": str(exc)}
            except OSError as kill_exc:
                return {"method": "taskkill", "hard": True, "error_type": type(kill_exc).__name__, "error": str(kill_exc), "taskkill_error": str(exc)}
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL if hard else signal.SIGTERM)
        return {"method": "process_group", "hard": hard}
    except (OSError, ProcessLookupError) as exc:
        try:
            process.kill()
            return {"method": "process", "hard": True, "fallback": True}
        except OSError as kill_exc:
            return {"method": "process", "hard": True, "fallback": True, "error_type": type(kill_exc).__name__, "error": str(kill_exc), "initial_error": str(exc)}


def _cleanup_process(process: subprocess.Popen[str], timeout_seconds: float) -> dict[str, object]:
    initial = _terminate_process_tree(process, hard=False, timeout_seconds=timeout_seconds)
    try:
        return {"initial": initial, "cleanup_complete": process.wait(timeout=timeout_seconds) is not None, "returncode": process.returncode}
    except subprocess.TimeoutExpired:
        hard = _terminate_process_tree(process, hard=True, timeout_seconds=timeout_seconds)
        try:
            return {"initial": initial, "hard": hard, "cleanup_complete": process.wait(timeout=timeout_seconds) is not None, "returncode": process.returncode}
        except subprocess.TimeoutExpired:
            return {"initial": initial, "hard": hard, "cleanup_complete": False, "returncode": process.poll()}


def _tail(path: Path, limit: int = 64 * 1024) -> str:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace")


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


def _progress(
    workspace: Path | None,
    commands: list[dict[str, object]],
    requested_steps: int,
    dump_interval_steps: int,
    status: SimulationStatus | None,
    *,
    minimization_expected: bool | None = None,
) -> dict[str, object]:
    parsed = parse_thermo_log(
        workspace / "slurry.log", stage="slurry", minimization_expected=minimization_expected,
    ) if workspace else None
    checkpoints = list(parsed.checkpoints) if parsed else []
    records = [{
        "step": item.dynamics_step,
        "raw_step": item.raw_step,
        "dynamics_step": item.dynamics_step,
        "phase": item.phase,
        "metrics": dict(item.metrics),
        "stage": item.stage,
        "source_log": item.source_log,
    } for item in checkpoints]
    dynamic_checkpoints = [item for item in checkpoints if item.dynamics_step is not None and item.phase == "dynamics"]
    completed = max((item.dynamics_step for item in dynamic_checkpoints), default=0)
    raw_completed = max((item.raw_step for item in checkpoints), default=None)
    wall_seconds = sum(float(command.get("wall_seconds", 0.0)) for command in commands if command.get("stage") == "slurry")
    return {
        "requested_slurry_steps": requested_steps,
        "requested_steps": requested_steps,
        "completed_slurry_steps": completed,
        "completed_steps": completed,
        "checkpoint_steps": [item.dynamics_step for item in dynamic_checkpoints],
        "checkpoint_raw_steps": [item.raw_step for item in checkpoints],
        "checkpoints": records,
        "thermo_parse_diagnostics": list(parsed.diagnostics) if parsed else [],
        "dump_interval_steps": dump_interval_steps,
        "last_raw_lammps_step": raw_completed,
        "last_thermo_step": raw_completed,
        "last_thermo_step_semantics": "raw_lammps_step",
        "wall_seconds": wall_seconds,
        "dynamics_wall_seconds": None,
        "steps_per_second": None,
        "estimated_remaining_seconds": None,
        "eta_limitation": "slurry command timing cannot be separated between minimization and dynamics",
        "early_termination": status is not None and completed < requested_steps,
    }


def _slurry_minimization_expected(workspace: Path) -> bool | None:
    script = workspace / "in_slurry.run"
    if not script.is_file():
        return None
    return bool(re.search(r"(?m)^\s*minimize(?:\s|$)", script.read_text(encoding="utf-8", errors="replace")))
