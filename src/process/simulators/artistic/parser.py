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


@dataclass(frozen=True)
class ThermoCheckpoint:
    raw_step: int
    metrics: dict[str, float]
    stage: str
    source_log: str
    dynamics_step: int | None = None
    phase: str = "dynamics"


@dataclass
class _ThermoBlock:
    headers: tuple[str, ...]
    start_line: int
    after_loop: bool
    rows: list[tuple[int, dict[str, float]]]


@dataclass(frozen=True)
class ThermoParseResult:
    checkpoints: tuple[ThermoCheckpoint, ...]
    diagnostics: tuple[str, ...] = ()


def parse_thermo_log(
    path: Path, *, stage: str = "slurry", minimization_expected: bool | None = None,
) -> ThermoParseResult:
    """Parse raw LAMMPS thermo rows and derive dynamics-relative steps from structure."""
    log = path / f"{stage}.log" if path.is_dir() else path
    if not log.is_file():
        return ThermoParseResult((), (f"missing thermo log: {log.name}",))
    diagnostics: list[str] = []
    blocks: list[_ThermoBlock] = []
    current: _ThermoBlock | None = None
    minimization_marker: int | None = None
    loop_since_header = False
    for line_number, line in enumerate(log.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if minimization_marker is None and re.search(r"\bMinimization stats\b", line, re.I):
            minimization_marker = line_number
        if _THERMO_HEADER.match(line):
            columns = tuple(line.split())
            headers = columns[1:]
            if not headers:
                diagnostics.append(f"{log.name}:{line_number}: thermo header has no metrics")
                current = None
            else:
                current = _ThermoBlock(headers, line_number, loop_since_header, [])
                blocks.append(current)
            loop_since_header = False
            continue
        if line.strip().startswith("Loop time"):
            loop_since_header = True
            current = None
            continue
        if current is None:
            continue
        match = _THERMO_ROW.match(line)
        if not match:
            continue
        fields = line.split()
        if len(fields) != len(current.headers) + 1:
            diagnostics.append(f"{log.name}:{line_number}: malformed thermo row for step {fields[0]}")
            continue
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError:
            diagnostics.append(f"{log.name}:{line_number}: malformed thermo metric row for step {fields[0]}")
            continue
        if not all(math.isfinite(value) for value in values):
            diagnostics.append(f"{log.name}:{line_number}: non-finite thermo metric row for step {fields[0]}")
            continue
        current.rows.append((int(fields[0]), dict(zip(current.headers, values))))

    if minimization_marker is not None:
        boundary = next((index for index, block in enumerate(blocks) if block.start_line > minimization_marker), len(blocks))
    elif minimization_expected:
        candidates = [index for index, block in enumerate(blocks) if block.after_loop]
        if len(candidates) != 1:
            return ThermoParseResult((), tuple(diagnostics) + (
                f"{log.name}: minimization/dynamics boundary is ambiguous; refusing dynamics-relative progress",
            ))
        boundary = candidates[0]
    else:
        boundary = 0

    rows: list[tuple[int, dict[str, float], str]] = []
    for index, block in enumerate(blocks):
        phase = "dynamics" if index >= boundary else "minimization"
        rows.extend((raw_step, metrics, phase) for raw_step, metrics in block.rows)
    dynamics_start = next((raw_step for raw_step, _, phase in rows if phase == "dynamics"), None)
    checkpoints: list[ThermoCheckpoint] = []
    last_dynamic: int | None = None
    for raw_step, metrics, phase in rows:
        dynamics_step = None
        if phase == "dynamics" and dynamics_start is not None:
            candidate = raw_step - dynamics_start
            if candidate < 0 or (last_dynamic is not None and candidate < last_dynamic):
                diagnostics.append(f"{log.name}: dynamics raw step moved backwards; refusing relative step")
            else:
                dynamics_step = candidate
                last_dynamic = candidate
        checkpoints.append(ThermoCheckpoint(raw_step, metrics, stage, log.name, dynamics_step, phase))
    if minimization_expected and minimization_marker is None and boundary == len(blocks):
        diagnostics.append(f"{log.name}: minimization completed without a dynamics thermo block")
    return ThermoParseResult(tuple(checkpoints), tuple(diagnostics))


def parse_thermo_checkpoints(
    path: Path, *, stage: str = "slurry", minimization_expected: bool | None = None,
) -> tuple[ThermoCheckpoint, ...]:
    return parse_thermo_log(path, stage=stage, minimization_expected=minimization_expected).checkpoints


def parse_thermo_steps(path: Path, *, stage: str = "slurry") -> tuple[int, ...]:
    """Compatibility projection for callers that only need printed step numbers."""
    return tuple(item.raw_step for item in parse_thermo_checkpoints(path, stage=stage))


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
