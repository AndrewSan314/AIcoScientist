from __future__ import annotations

from pathlib import Path

from .parser import ParsedArtisticOutput
from .schemas import ArtisticRecipe


def output_errors(recipe: ArtisticRecipe, workspace: Path, parsed: ParsedArtisticOutput, lost_tolerance: float) -> tuple[str, ...]:
    expected = ["density_slurry.out", "coord_out_slurry.data"]
    if recipe.drying_mode:
        expected.extend(["AM_loading.out", "coord_out_electrode.data"])
    if recipe.calendering:
        expected.extend(["coord_out_cal.data", "new_CBD_nanoporosity"])
    errors = [f"missing expected source output: {name}" for name in expected if not (workspace / name).is_file()]
    if parsed.lost_fraction is not None and parsed.lost_fraction > lost_tolerance:
        errors.append(f"particle loss {parsed.lost_fraction:.6g} exceeds tolerance {lost_tolerance:.6g}")
    return tuple(errors)
