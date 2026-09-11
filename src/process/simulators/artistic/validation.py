from __future__ import annotations

import math
from pathlib import Path

from .parser import ParsedArtisticOutput
from .schemas import ArtisticRecipe


def output_errors(recipe: ArtisticRecipe, workspace: Path, parsed: ParsedArtisticOutput, lost_tolerance: float) -> tuple[str, ...]:
    expected = {"density_slurry.out", "coord_out_slurry.data"}
    required_values = {"slurry_density"}
    if recipe.drying_mode:
        expected.update({"AM_loading.out", "coord_out_electrode.data", "porosity_bulk.out", "porosity_all.out", "check.txt"})
        required_values.update({"am_loading", "drying_porosity_bulk_percent", "drying_porosity_all_percent"})
    if recipe.calendering:
        expected.update({"coord_out_cal.data", "Cal_electrode.atom", "new_CBD_nanoporosity", "porosity_cal_bulk.out", "porosity_cal_all.out", "check_cal.txt"})
        required_values.update({"calendered_electrode_thickness", "calendered_cbd_nanoporosity", "calendered_porosity_bulk_percent", "calendered_porosity_all_percent"})
    errors = [f"missing expected source output: {name}" for name in sorted(expected) if not (workspace / name).is_file()]
    errors.extend(f"empty source output: {name}" for name in sorted(expected) if (workspace / name).is_file() and (workspace / name).stat().st_size == 0)
    values = {**parsed.stages.get("slurry", {}), **parsed.stages.get("drying", {}), **parsed.final_kpis}
    errors.extend(parsed.parse_errors)
    errors.extend(f"missing parsed source quantity: {name}" for name in sorted(required_values - set(values)))
    for name, value in values.items():
        if not math.isfinite(value):
            errors.append(f"non-finite parsed source quantity: {name}")
    for name in ("slurry_density", "am_loading", "calendered_electrode_thickness"):
        if name in values and values[name] <= 0:
            errors.append(f"non-positive physical output: {name}")
    for name in ("drying_porosity_bulk_percent", "drying_porosity_all_percent", "calendered_porosity_bulk_percent", "calendered_porosity_all_percent"):
        if name in values and not 0 <= values[name] <= 100:
            errors.append(f"out-of-range porosity percent: {name}")
    if "calendered_cbd_nanoporosity" in values and not 0 <= values["calendered_cbd_nanoporosity"] <= 1:
        errors.append("out-of-range CBD nanoporosity")
    if parsed.lost_fraction is not None and parsed.lost_fraction > lost_tolerance:
        errors.append(f"particle loss {parsed.lost_fraction:.6g} exceeds tolerance {lost_tolerance:.6g}")
    return tuple(dict.fromkeys(errors))
