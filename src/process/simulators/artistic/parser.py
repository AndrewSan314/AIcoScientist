from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_ATOMS = re.compile(r"^\s*(\d+)\s+atoms\s*$")
_BOUNDS = re.compile(r"^\s*(\S+)\s+(\S+)\s+[xyz]lo\s+[xyz]hi\s*$")
_LOST = re.compile(r"Lost atoms:\s*original\s+(\d+)\s+current\s+(\d+)", re.I)


@dataclass(frozen=True)
class ParsedArtisticOutput:
    stages: dict[str, dict[str, float]]
    final_kpis: dict[str, float]
    initial_atoms: int | None
    final_atoms: int | None
    log_lost_fraction: float | None

    @property
    def lost_fraction(self) -> float | None:
        if self.initial_atoms and self.final_atoms is not None:
            return max(0.0, (self.initial_atoms - self.final_atoms) / self.initial_atoms)
        return self.log_lost_fraction


def parse_artistic_output(workspace: Path) -> ParsedArtisticOutput:
    stages: dict[str, dict[str, float]] = {}
    slurry = _read_number(workspace / "density_slurry.out")
    if slurry is not None:
        stages["slurry"] = {"slurry_density": slurry}
    drying = _read_number(workspace / "AM_loading.out")
    if drying is not None:
        stages["drying"] = {"am_loading": drying}
    calendering: dict[str, float] = {}
    initial_thickness = _read_number(workspace / "initial_lz")
    if initial_thickness is not None:
        calendering["initial_thickness"] = initial_thickness
    nanoporosity = _read_number(workspace / "new_CBD_nanoporosity")
    if nanoporosity is not None:
        calendering["cbd_nanoporosity"] = nanoporosity
    final_box = _lammps_box(workspace / "coord_out_cal.data")
    if final_box and "z" in final_box:
        calendering["electrode_thickness"] = final_box["z"]
    if calendering:
        stages["calendering"] = calendering
    initial_atoms = _lammps_atoms(workspace / "coord_in.data")
    final_atoms = _lammps_atoms(workspace / ("coord_out_cal.data" if (workspace / "coord_out_cal.data").is_file() else "coord_out_electrode.data"))
    lost = _lost_from_logs(workspace)
    final_kpis = dict(calendering)
    return ParsedArtisticOutput(stages=stages, final_kpis=final_kpis, initial_atoms=initial_atoms, final_atoms=final_atoms, log_lost_fraction=lost)


def _read_number(path: Path) -> float | None:
    if not path.is_file():
        return None
    match = _NUMBER.search(path.read_text(encoding="utf-8", errors="replace"))
    return float(match.group()) if match else None


def _lammps_atoms(path: Path) -> int | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _ATOMS.match(line)
        if match:
            return int(match.group(1))
    return None


def _lammps_box(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _BOUNDS.match(line)
        if match:
            axis = line.split()[-2][0]
            values[axis] = float(match.group(2)) - float(match.group(1))
    return values


def _lost_from_logs(workspace: Path) -> float | None:
    fractions: list[float] = []
    for path in workspace.glob("*.log"):
        for original, current in _LOST.findall(path.read_text(encoding="utf-8", errors="replace")):
            if int(original):
                fractions.append(max(0.0, (int(original) - int(current)) / int(original)))
    return max(fractions) if fractions else None
