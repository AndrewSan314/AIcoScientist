from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


SOURCE_TO_ROLE = {
    "slurry_density": "intermediate_state", "am_loading": "intermediate_state",
    "drying_porosity_bulk_percent": "intermediate_state", "drying_porosity_all_percent": "intermediate_state",
    "initial_thickness": "diagnostic", "simulation_box_z_length": "diagnostic", "calendered_electrode_thickness": "final_kpi",
    "calendered_cbd_nanoporosity": "final_kpi", "calendered_porosity_bulk_percent": "final_kpi",
    "calendered_porosity_all_percent": "final_kpi",
}
_ATOMS = re.compile(r"^\s*(\d+)\s+atoms\s*$")
_BOUNDS = re.compile(r"^\s*(\S+)\s+(\S+)\s+[xyz]lo\s+[xyz]hi\s*$")
_LOST = re.compile(r"Lost atoms:\s*original\s+(\d+)\s+current\s+(\d+)", re.I)
_THERMO_HEADER = re.compile(r"^\s*Step(?:\s|$)", re.I)
_THERMO_ROW = re.compile(r"^\s*(\d+)(?:\s|$)")


@dataclass(frozen=True)
class ParsedArtisticOutput:
    stages: dict[str, dict[str, float]]
    final_kpis: dict[str, float]
    diagnostics: dict[str, float]
    initial_atoms: int | None
    final_atoms: int | None
    log_lost_fraction: float | None
    parse_errors: tuple[str, ...] = ()

    @property
    def lost_fraction(self) -> float | None:
        if self.initial_atoms and self.final_atoms is not None:
            return max(0.0, (self.initial_atoms - self.final_atoms) / self.initial_atoms)
        return self.log_lost_fraction


def parse_artistic_output(workspace: Path) -> ParsedArtisticOutput:
    errors: list[str] = []
    slurry = _scalar(workspace / "density_slurry.out", errors)
    drying_loading = _scalar(workspace / "AM_loading.out", errors)
    drying_bulk = _scalar(workspace / "porosity_bulk.out", errors)
    drying_all = _scalar(workspace / "porosity_all.out", errors)
    initial_thickness = _scalar(workspace / "initial_lz", errors)
    cbd_nanoporosity = _scalar(workspace / "new_CBD_nanoporosity", errors)
    cal_bulk = _scalar(workspace / "porosity_cal_bulk.out", errors)
    cal_all = _scalar(workspace / "porosity_cal_all.out", errors)
    final_box = _lammps_box(workspace / "coord_out_cal.data", errors)
    final_thickness = _electrode_thickness(workspace / "Cal_electrode.atom", errors)
    stages: dict[str, dict[str, float]] = {}
    if slurry is not None:
        stages["slurry"] = {"slurry_density": slurry}
    drying = _present(am_loading=drying_loading, drying_porosity_bulk_percent=drying_bulk, drying_porosity_all_percent=drying_all)
    if drying:
        stages["drying"] = drying
    final_kpis = _present(calendered_electrode_thickness=final_thickness, calendered_cbd_nanoporosity=cbd_nanoporosity, calendered_porosity_bulk_percent=cal_bulk, calendered_porosity_all_percent=cal_all)
    diagnostics = _present(initial_thickness=initial_thickness, simulation_box_z_length=final_box.get("z") if final_box else None)
    initial_atoms = _lammps_atoms(workspace / "coord_in.data", errors)
    final_path = workspace / ("coord_out_cal.data" if (workspace / "coord_out_cal.data").is_file() else "coord_out_electrode.data")
    final_atoms = _lammps_atoms(final_path, errors)
    return ParsedArtisticOutput(stages, final_kpis, diagnostics, initial_atoms, final_atoms, _lost_from_logs(workspace), tuple(errors))


def parse_thermo_checkpoints(workspace: Path) -> tuple[int, ...]:
    """Return only step numbers actually printed by LAMMPS thermo logs."""
    steps: set[int] = set()
    for path in workspace.glob("*.log"):
        in_thermo = False
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if _THERMO_HEADER.search(line):
                in_thermo = True
                continue
            if in_thermo:
                match = _THERMO_ROW.match(line)
                if match:
                    steps.add(int(match.group(1)))
                elif line.strip().startswith(("Loop time", "ERROR", "Per MPI rank")):
                    in_thermo = False
    return tuple(sorted(steps))


def _present(**values: float | None) -> dict[str, float]:
    return {name: value for name, value in values.items() if value is not None}


def _scalar(path: Path, errors: list[str]) -> float | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        value = float(text)
    except ValueError:
        errors.append(f"malformed scalar output: {path.name}")
        return None
    if not math.isfinite(value):
        errors.append(f"non-finite scalar output: {path.name}")
        return None
    return value


def _lammps_atoms(path: Path, errors: list[str]) -> int | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _ATOMS.match(line)
            if match:
                return int(match.group(1))
    errors.append(f"malformed LAMMPS data output: {path.name}")
    return None


def _lammps_box(path: Path, errors: list[str]) -> dict[str, float]:
    if not path.is_file():
        return {}
    values: dict[str, float] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _BOUNDS.match(line)
            if match:
                try:
                    value = float(match.group(2)) - float(match.group(1))
                except ValueError:
                    errors.append(f"malformed box bound in {path.name}")
                    return {}
                if not math.isfinite(value):
                    errors.append(f"non-finite box bound in {path.name}")
                    return {}
                values[line.split()[-2][0]] = value
                if "z" in values:
                    return values
    if "z" not in values:
        errors.append(f"missing z bounds in {path.name}")
    return values


def _electrode_thickness(path: Path, errors: list[str]) -> float | None:
    """Match pores_cal.py: final particle zmax measured from the lower z boundary."""
    if not path.is_file():
        return None
    zlo: float | None = None
    z_index: int | None = None
    maximum: float | None = None
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("ITEM: BOX BOUNDS"):
                    bounds = [next(handle).split() for _ in range(3)]
                    zlo = float(bounds[2][0])
                elif line.startswith("ITEM: ATOMS "):
                    columns = line.split()[2:]
                    if "z" not in columns:
                        errors.append(f"missing z particle coordinate in {path.name}")
                        return None
                    z_index = columns.index("z")
                    for atom in handle:
                        fields = atom.split()
                        if not fields:
                            continue
                        if fields[0] == "ITEM:":
                            break
                        if len(fields) <= z_index:
                            errors.append(f"malformed particle coordinate in {path.name}")
                            return None
                        z = float(fields[z_index])
                        if not math.isfinite(z):
                            errors.append(f"non-finite particle coordinate in {path.name}")
                            return None
                        maximum = z if maximum is None else max(maximum, z)
    except (OSError, StopIteration, ValueError):
        errors.append(f"malformed particle dump: {path.name}")
        return None
    if zlo is None or z_index is None or maximum is None:
        errors.append(f"missing particle extent in {path.name}")
        return None
    thickness = maximum - zlo
    if not math.isfinite(thickness):
        errors.append(f"non-finite particle extent in {path.name}")
        return None
    return thickness


def _lost_from_logs(workspace: Path) -> float | None:
    fractions: list[float] = []
    for path in workspace.glob("*.log"):
        for original, current in _LOST.findall(path.read_text(encoding="utf-8", errors="replace")):
            if int(original):
                fractions.append(max(0.0, (int(original) - int(current)) / int(original)))
    return max(fractions) if fractions else None
