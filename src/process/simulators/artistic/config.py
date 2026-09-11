from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


PINNED_COMMIT = "5af9e0345673fac557c479ee8f4a0727e442c1fa"
SOURCE_URL = "https://github.com/ravinsingh166/Manufacturing-Model-Codes.git"


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
    mpi_launcher: str = "mpirun"
    mpi_processes: int = 1
    slurm_submit: str = "sbatch"
    timeout_seconds: int = 86_400
    lost_particle_tolerance: float = 0.0
    apply_verified_patches: bool = True

    def __post_init__(self) -> None:
        if self.mpi_processes < 1 or self.timeout_seconds < 1:
            raise ValueError("mpi_processes and timeout_seconds must be positive")
        if not 0 <= self.lost_particle_tolerance <= 1:
            raise ValueError("lost_particle_tolerance must be in [0, 1]")

    @property
    def source_tree(self) -> Path:
        return self.source_root / "NMC" / "Updated version"
