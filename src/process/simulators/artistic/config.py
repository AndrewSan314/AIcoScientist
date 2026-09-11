from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


PINNED_COMMIT = "5af9e0345673fac557c479ee8f4a0727e442c1fa"
PINNED_SOURCE_TREE_HASH = "e870c018acf8696424ae3ac9c36163f5bd3afb56"
SOURCE_URL = "https://github.com/ravinsingh166/Manufacturing-Model-Codes.git"


class SourcePinError(RuntimeError):
    """The local checkout is not the immutable source audited for ARTISTIC."""


class MPIEnvironmentError(RuntimeError):
    """MPI/LAMMPS compatibility could not be proven before a real run."""

    def __init__(self, message: str, metadata: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


class ExecutionMode(StrEnum):
    LOCAL = "local"
    MPI = "mpi"
    SLURM = "slurm"


class FidelityMode(StrEnum):
    REFERENCE = "REFERENCE"
    SHORT_HORIZON = "SHORT_HORIZON"


REFERENCE_SLURRY_STEPS = 20_000_000


def _default_mpi_launcher() -> str:
    if os.name != "nt":
        return "mpirun"
    installed = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft MPI" / "Bin" / "mpiexec.exe"
    return str(installed) if installed.is_file() else "mpiexec"


@dataclass(frozen=True)
class ArtisticRunConfig:
    source_root: Path = Path("data/external/artistic/source") / f"Manufacturing-Model-Codes-{PINNED_COMMIT}"
    output_root: Path = Path("outputs/artistic_runs")
    lammps_command: str = "lmp"
    execution_mode: ExecutionMode = ExecutionMode.LOCAL
    mpi_launcher: str = _default_mpi_launcher()
    mpi_processes: int = 1
    slurm_submit: str = "sbatch"
    timeout_seconds: int = 86_400
    cleanup_timeout_seconds: float = 5.0
    lost_particle_tolerance: float = 0.0
    apply_verified_patches: bool = True
    max_particle_count: int = 1_000_000
    allow_unsafe_particle_count: bool = False
    fidelity_mode: FidelityMode = FidelityMode.REFERENCE
    slurry_steps: int | None = None
    dump_interval_steps: int = 1_000_000
    confirm_reference_execution: bool = False

    def __post_init__(self) -> None:
        mode = FidelityMode(self.fidelity_mode)
        object.__setattr__(self, "fidelity_mode", mode)
        if self.mpi_processes < 1 or self.timeout_seconds < 1 or self.cleanup_timeout_seconds <= 0 or self.max_particle_count < 1:
            raise ValueError("mpi_processes, timeout_seconds, cleanup_timeout_seconds, and max_particle_count must be positive")
        if not 0 <= self.lost_particle_tolerance <= 1:
            raise ValueError("lost_particle_tolerance must be in [0, 1]")
        if self.dump_interval_steps < 1:
            raise ValueError("dump_interval_steps must be positive")
        if mode == FidelityMode.REFERENCE:
            if self.slurry_steps is not None and self.slurry_steps != REFERENCE_SLURRY_STEPS:
                raise ValueError(f"reference ARTISTIC fidelity requires exactly {REFERENCE_SLURRY_STEPS:,} slurry steps")
            if self.dump_interval_steps != 1_000_000:
                raise ValueError("reference ARTISTIC fidelity requires dump_interval_steps=1,000,000")
        elif self.slurry_steps is None or not 0 < self.slurry_steps < REFERENCE_SLURRY_STEPS:
            raise ValueError(f"short-horizon ARTISTIC fidelity requires explicit slurry_steps below {REFERENCE_SLURRY_STEPS:,}")

    @property
    def source_tree(self) -> Path:
        return self.source_root / "NMC" / "Updated version"

    @property
    def requested_slurry_steps(self) -> int:
        return REFERENCE_SLURRY_STEPS if self.fidelity_mode == FidelityMode.REFERENCE else int(self.slurry_steps)

    @property
    def fidelity_identity(self) -> str:
        payload = {
            "mode": self.fidelity_mode.value,
            "reference_slurry_steps": REFERENCE_SLURRY_STEPS,
            "requested_slurry_steps": self.requested_slurry_steps,
            "dump_interval_steps": self.dump_interval_steps,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def verify_source_pin(self) -> None:
        if not self.source_tree.is_dir():
            raise SourcePinError(f"missing ARTISTIC source tree: {self.source_tree}")
        def git(*args: str) -> str:
            try:
                result = subprocess.run(["git", "-C", str(self.source_root), *args], capture_output=True, text=True, check=False)
            except OSError as exc:
                raise SourcePinError(f"cannot verify ARTISTIC source identity: {exc}") from exc
            if result.returncode:
                raise SourcePinError(f"cannot verify ARTISTIC source identity: {' '.join(args)}")
            return result.stdout.strip()

        commit = git("rev-parse", "HEAD")
        tree = git("rev-parse", "HEAD:NMC/Updated version")
        dirty = git("status", "--porcelain", "--untracked-files=all")
        if commit != PINNED_COMMIT:
            raise SourcePinError(f"ARTISTIC commit mismatch: expected {PINNED_COMMIT}, got {commit}")
        if tree != PINNED_SOURCE_TREE_HASH:
            raise SourcePinError(f"ARTISTIC source tree mismatch: expected {PINNED_SOURCE_TREE_HASH}, got {tree}")
        if dirty:
            raise SourcePinError("ARTISTIC source checkout is modified or contains untracked files")


def _resolve_executable(command: str) -> str | None:
    path = Path(command)
    return str(path.resolve()) if path.is_file() else shutil.which(command)


def _command_version(command: str) -> str:
    for flag in ("--version", "-h", "-help"):
        try:
            result = subprocess.run([command, flag], capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = [line.strip() for line in (result.stdout or result.stderr).splitlines() if line.strip()]
        usable = [line for line in lines if not line.lower().startswith(("error", "unknown option", "invalid command-line"))]
        if usable:
            return usable[0]
    return "unavailable"


def physics_config_fingerprint(
    *, recipe_fingerprint: str, source_commit: str, source_tree_hash: str,
    patches: list[dict[str, object]] | tuple[dict[str, object], ...] = (),
) -> str:
    """Identity for physics-affecting inputs; horizon and dump-only patches are excluded."""
    excluded = {"short_horizon_slurry_steps", "short_horizon_checkpoint_interval"}
    normalized_patches = [
        patch for patch in patches
        if str(patch.get("id", "")) not in excluded
    ]
    payload = {
        "recipe_fingerprint": recipe_fingerprint, "source_commit": source_commit,
        "source_tree_hash": source_tree_hash, "physics_patches": normalized_patches,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _processor_grids(output: str) -> list[tuple[int, int, int]]:
    patterns = (
        r"(\d+)\s+by\s+(\d+)\s+by\s+(\d+)\s+MPI processor grid",
        r"Processor grid\s*=\s*(\d+)\s+(\d+)\s+(\d+)",
    )
    return [tuple(map(int, match)) for pattern in patterns for match in re.findall(pattern, output, re.IGNORECASE)]


def validate_mpi_environment(config: ArtisticRunConfig) -> dict[str, object]:
    """Prove that the requested launcher runs one distributed LAMMPS job."""
    resolved_launcher = _resolve_executable(config.mpi_launcher)
    resolved_lammps = _resolve_executable(config.lammps_command)
    metadata: dict[str, object] = {
        "execution_mode": config.execution_mode.value,
        "requested_mpi_processes": config.mpi_processes,
        "mpi_launcher": config.mpi_launcher,
        "resolved_mpi_launcher": resolved_launcher,
        "resolved_lammps_path": resolved_lammps,
        "mpi_implementation_version": _command_version(resolved_launcher) if resolved_launcher else "unavailable",
        "lammps_version": _command_version(resolved_lammps) if resolved_lammps else "unavailable",
        "validated": False,
    }
    if not resolved_launcher:
        raise MPIEnvironmentError(f"MPI launcher does not resolve: {config.mpi_launcher}", metadata)
    if not resolved_lammps:
        raise MPIEnvironmentError(f"LAMMPS executable does not resolve: {config.lammps_command}", metadata)

    command = [resolved_launcher, "-n", str(config.mpi_processes), resolved_lammps, "-log", "none", "-in", "smoke.in"]
    metadata["mpi_launcher_command"] = command
    smoke_input = "\n".join((
        "clear", "units lj", "atom_style atomic", "region box block 0 2 0 2 0 2", "create_box 1 box",
        "create_atoms 1 single 1 1 1", "mass 1 1.0", "run 0", "",
    ))
    try:
        with tempfile.TemporaryDirectory(prefix="artistic-mpi-smoke-") as directory:
            smoke_dir = Path(directory)
            (smoke_dir / "smoke.in").write_text(smoke_input, encoding="utf-8", newline="\n")
            result = subprocess.run(command, cwd=smoke_dir, capture_output=True, text=True, timeout=min(config.timeout_seconds, 120), check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MPIEnvironmentError(f"MPI/LAMMPS smoke run failed: {exc}", metadata) from exc
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    metadata["smoke_returncode"] = result.returncode
    grids = _processor_grids(output)
    metadata["mpi_processor_grids"] = [list(grid) for grid in grids]
    if result.returncode:
        raise MPIEnvironmentError(f"MPI/LAMMPS smoke run exited {result.returncode}", metadata)
    if len(grids) != 1:
        raise MPIEnvironmentError("MPI smoke run did not report exactly one LAMMPS processor grid", metadata)
    task_count = grids[0][0] * grids[0][1] * grids[0][2]
    metadata["mpi_task_count"] = task_count
    if task_count != config.mpi_processes:
        raise MPIEnvironmentError(f"MPI smoke run reported {task_count} tasks; requested {config.mpi_processes}", metadata)
    metadata["validated"] = True
    return metadata
