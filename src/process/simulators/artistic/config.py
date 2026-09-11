from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path
import subprocess


PINNED_COMMIT = "5af9e0345673fac557c479ee8f4a0727e442c1fa"
PINNED_SOURCE_TREE_HASH = "e870c018acf8696424ae3ac9c36163f5bd3afb56"
SOURCE_URL = "https://github.com/ravinsingh166/Manufacturing-Model-Codes.git"


class SourcePinError(RuntimeError):
    """The local checkout is not the immutable source audited for ARTISTIC."""


class ExecutionMode(StrEnum):
    LOCAL = "local"
    MPI = "mpi"
    SLURM = "slurm"


@dataclass(frozen=True)
class ArtisticRunConfig:
    source_root: Path = Path("data/external/artistic/source") / f"Manufacturing-Model-Codes-{PINNED_COMMIT}"
    output_root: Path = Path("outputs/artistic_runs")
    lammps_command: str = "lmp"
    execution_mode: ExecutionMode = ExecutionMode.LOCAL
    mpi_launcher: str = "mpiexec" if os.name == "nt" else "mpirun"
    mpi_processes: int = 1
    slurm_submit: str = "sbatch"
    timeout_seconds: int = 86_400
    lost_particle_tolerance: float = 0.0
    apply_verified_patches: bool = True
    max_particle_count: int = 1_000_000
    allow_unsafe_particle_count: bool = False

    def __post_init__(self) -> None:
        if self.mpi_processes < 1 or self.timeout_seconds < 1 or self.max_particle_count < 1:
            raise ValueError("mpi_processes, timeout_seconds, and max_particle_count must be positive")
        if not 0 <= self.lost_particle_tolerance <= 1:
            raise ValueError("lost_particle_tolerance must be in [0, 1]")

    @property
    def source_tree(self) -> Path:
        return self.source_root / "NMC" / "Updated version"

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
